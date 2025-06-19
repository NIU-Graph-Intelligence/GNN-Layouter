#!/usr/bin/env python3
import os, sys
import math
import torch
import argparse
from torch_geometric.loader import DataLoader
from torch.nn import MSELoss
import pandas as pd  # For a nice summary table


# Project-specific imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models_PPI import IGNN
from models.GCNFR import ForceGNN
from utils import get_spectral_rad
from data.dataset import data_loader  # Add this import


def orthogonal_procrustes_torch(A, B):
    # Be clever with transposes, with the intention to save memory.
    A_device = A.device
    B_copy = B.clone().to(A_device)

    input = torch.transpose(torch.matmul(torch.transpose(B_copy, 0, 1), A), 0, 1)
    u, w, vt = torch.svd(input)
    # u, w, vt = torch.svd(torch.transpose(torch.matmul(torch.transpose(B,0,1),A),0,1))
    R = torch.matmul(u, torch.transpose(vt, 0, 1))
    scale = torch.sum(w)
    return R, scale


def criterion_procrustes(data1, data2):
    device = data1.device
    mtx1 = data1
    mtx2 = data2.clone().to(device)

    # translate all the data to the origin
    mtx3 = mtx1 - torch.mean(mtx1, 0)
    mtx4 = mtx2 - torch.mean(mtx2, 0)

    norm1 = torch.norm(mtx3)
    norm2 = torch.norm(mtx4)

    if norm1 == 0:
        norm1 = 1e-16
    if norm2 == 0:
        norm2 = 1e-16
    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx3 = mtx3 / norm1
    mtx4 = mtx4 / norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes_torch(mtx3, mtx4)
    mtx4 = torch.matmul(mtx4, torch.transpose(R, 0, 1)) * s

    # measure the dissimilarity between the two datasets
    disparity = torch.sum((mtx3 - mtx4) ** 2)

    return disparity

def fr_net_force_metric(positions, edge_index, area=None):

    N = positions.shape[0]
    if area is None:
        area = 1.0

    k = (area / N) ** 0.5

    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=-1) + 1e-8
    rep_force = k * k * diff / dist_sq.unsqueeze(-1)
    rep_force = rep_force.sum(dim=1)

    att_force = torch.zeros_like(positions)
    src, dst = edge_index
    edge_vec = positions[src] - positions[dst]
    edge_dist = (edge_vec ** 2).sum(dim=1).sqrt() + 1e-8
    att = - (edge_dist ** 2 / k).unsqueeze(-1) * (edge_vec / edge_dist.unsqueeze(-1))
    att_force = att_force.index_add(0, src, att)

    net_force = rep_force + att_force
    net_force_magnitude = net_force.norm(dim=1)
    return net_force_magnitude.mean().item()

# ---------- Model Registry ----------
MODEL_REGISTRY = {
    "IGNN": {
        "class": IGNN,
        "checkpoint": "IGNN_checkpoints/IGNN_bs16_ep1000HC64.pt",
        "data_path": "data/processed/modelInput_FR1024.pt",
        "feature_func": lambda data: torch.cat([data.x, data.init_coords], dim=1).T,
        "model_args":{
            "nfeat": 42,  # Example values
            "nhid": 64,
            "nclass": 2,
            "num_node": 40,
            "dropout": 0.2,
            "kappa": 0.8,
        },
        "forward_func": lambda model, data, adj: model(torch.cat([data.x, data.init_coords], dim=1).T, adj),
    },
    "ForceGNN": {
        "class": ForceGNN,
        "checkpoint": "results/metrics/ForceGNN/FinalWeights_ForceGNN_FR_batch128.pt",
        "data_path": "data/processed/modelInput_FR1024.pt",
        "feature_func": lambda data: torch.cat([data.x, data.init_coords], dim=1),
        "model_args": {
            "in_feat": 42,
            "hidden_dim": 128,
            "out_feat": 2,
            "num_layers": 3,
        },
        "forward_func": lambda model, data, adj: model(
            data.x, data.edge_index, data.init_coords if hasattr(data, "init_coords") else None
        )
    },
}

def evaluate_model(model_class, checkpoint_path, dataset_path, feature_func, device, model_args, forward_func, batch_size=128):
    # Load the full dataset
    data_dict = torch.load(dataset_path)
    full_dataset = data_dict['dataset']  # Extract the dataset from the dictionary
    
    # Get data loaders with consistent splits
    train_loader, val_loader, test_loader = data_loader(
        dataset=full_dataset,
        batch_size=batch_size,
        splits=(0.7, 0.15, 0.15),
        random_state=42
    )

    # Instantiate model
    model = model_class(**model_args).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    total_procrustes = 0.0
    total_fr_force = 0.0
    num_graphs = 0

    with torch.no_grad():
        for data in test_loader:
            num_edges = data.edge_index.size(1)
            edge_weight = torch.ones(num_edges, dtype=torch.float).to(device)
            data = data.to(device)

            # Adjacency setup (use edge_attr if available)
            adj = torch.sparse.FloatTensor(
                data.edge_index,
                edge_weight,
                (data.num_nodes, data.num_nodes)
            ).coalesce()

            # Spectral radius update if model has it
            if hasattr(model, "adj_rho"):
                model.adj_rho = get_spectral_rad(adj)

            pred = forward_func(model, data, adj)

            # ---- Metric 1: Procrustes (matrix R2) ----
            procrustes_loss = criterion_procrustes(pred, data.y).item()
            total_procrustes += procrustes_loss

            # ---- Metric 2: FR Net Force ----
            # For all models, pred is [N,2], data.edge_index is [2,E]
            fr_force = fr_net_force_metric(pred, data.edge_index)
            total_fr_force += fr_force

            num_graphs += 1

    avg_procrustes = total_procrustes / num_graphs
    avg_fr_force = total_fr_force / num_graphs
    return avg_procrustes, avg_fr_force

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple GNN models on test dataset")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (cuda:0 or cpu)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {}

    for model_name, config in MODEL_REGISTRY.items():
        print(f"\nEvaluating {model_name}...")
        avg_procrustes, avg_fr_force = evaluate_model(
            model_class=config["class"],
            checkpoint_path=config["checkpoint"],
            dataset_path=config["data_path"],
            feature_func=config["feature_func"],
            device=device,
            model_args=config["model_args"],
            forward_func=config["forward_func"],
            batch_size=args.batch_size
        )
        results[model_name] = {
            "Procrustes Static": avg_procrustes,
            "FR Net Force": avg_fr_force,
        }
        print(f"  {model_name} Procrustes Static: {avg_procrustes:.6f}")
        print(f"  {model_name} FR Net Force:      {avg_fr_force:.6f}")

    # Present summary table
    df = pd.DataFrame(results).T
    print("\n--- Evaluation Summary ---")
    print(df)

if __name__ == "__main__":
    main()
