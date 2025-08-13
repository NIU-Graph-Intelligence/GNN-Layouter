#!/usr/bin/env python3
import os, sys
import math
import torch
import argparse
from torch_geometric.loader import DataLoader
from torch.nn import MSELoss
import pandas as pd 


# Project-specific imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models_PPI import IGNN
from models.GCNFR import ForceGNN
from utils import get_spectral_rad
from data.dataset import force_directed_data_loader
from config_utils.config_manager import get_config


def orthogonal_procrustes_torch(A, B):
    """
    Compute the orthogonal Procrustes transformation between A and B.
    """

    
    # Be clever with transposes, with the intention to save memory.
    A_device = A.device
    B_copy = B.clone().to(A_device)

    # Compute correlation matrix
    input_matrix = torch.matmul(torch.transpose(B_copy, 0, 1), A)
    # print(f"Correlation matrix shape: {input_matrix.shape}")
    
    # SVD
    u, w, vt = torch.svd(input_matrix)
    # print(f"SVD shapes - U: {u.shape}, w: {w.shape}, Vt: {vt.shape}")
    
    # Compute rotation matrix
    R = torch.matmul(u, torch.transpose(vt, 0, 1))
    # print(f"Rotation matrix R shape: {R.shape}")
    
    scale = torch.sum(w)
    # print(f"Scale factor: {scale.item():.4f}")
    
    return R, scale


def criterion_procrustes(pred, target):
   
  
    # If pred is batched, take the first graph
    if len(pred.shape) == 3:
        pred = pred[0]  # Now shape is [nodes_per_graph, 2]
    
    device = pred.device
    mtx1 = pred
    mtx2 = target.clone().to(device)

    # Center the coordinates
    mtx3 = mtx1 - torch.mean(mtx1, dim=0, keepdim=True)
    mtx4 = mtx2 - torch.mean(mtx2, dim=0, keepdim=True)

    # Normalize to unit scale
    norm1 = torch.norm(mtx3)
    norm2 = torch.norm(mtx4)

    if norm1 < 1e-8:
        norm1 = 1e-8
    if norm2 < 1e-8:
        norm2 = 1e-8

    mtx3 = mtx3 / norm1
    mtx4 = mtx4 / norm2

    # Compute optimal rotation and scale
    R, s = orthogonal_procrustes_torch(mtx3, mtx4)
    
    # Apply transformation
    mtx4 = torch.matmul(mtx4, torch.transpose(R, 0, 1)) * s

    # Compute disparity
    disparity = torch.sum((mtx3 - mtx4) ** 2)
    
    return disparity

def fr_net_force_metric(positions, edge_index, area=None):
    
    # Handle batched input - take first graph if batched
    if len(positions.shape) == 3:
        positions = positions[0]  # Now shape is [num_nodes, 2]
    
    N = positions.shape[0]
    if area is None:
        area = 1.0
    
    k = (area / N) ** 0.5
    
    # Compute repulsive forces
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [N, N, 2]
    dist_sq = (diff ** 2).sum(dim=-1) + 1e-8  # [N, N]
    rep_force = k * k * diff / dist_sq.unsqueeze(-1)  # [N, N, 2]
    rep_force = rep_force.sum(dim=1)  # [N, 2]
    
    # Compute attractive forces
    src, dst = edge_index
    edge_vec = positions[src] - positions[dst]  # [E, 2]
    edge_dist = (edge_vec ** 2).sum(dim=1).sqrt() + 1e-8  # [E]
    att = - (edge_dist ** 2 / k).unsqueeze(-1) * (edge_vec / edge_dist.unsqueeze(-1))  # [E, 2]
    
    # Sum attractive forces for each node
    att_force = torch.zeros_like(positions)  # [N, 2]
    att_force = att_force.index_add(0, src, att)
    
    # Compute net force
    net_force = rep_force + att_force  # [N, 2]
    net_force_magnitude = net_force.norm(dim=1)  # [N]
    
    return net_force_magnitude.mean().item()

# ---------- Model Registry ----------
def get_model_registry():
    """Generate model registry from configuration"""
    config = get_config()
    
    # Get model configurations
    forcegnn_config = config.get_model_config('ForceGNN')
    
    MODEL_REGISTRY = {
        "IGNN": {
            "class": IGNN,
            "checkpoint": "IGNN_checkpoints/CustomOneHotOnlyIGNN_bs128_ep500_HC64.pt",
            "data_path": config.get_data_path('force_directed'),
            "feature_func": lambda data: torch.cat([data.x, data.init_coords], dim=1).T,
            "model_args":{
                "nfeat": 44,  # Example values
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
            "checkpoint": config.get_model_path('force_forcegnn'),
            "data_path": config.get_data_path('force_directed'),
            "feature_func": lambda data: torch.cat([data.x, data.init_coords], dim=1),
            "model_args": {
                "in_feat": 44,  # 40 (one-hot) + 2 (coords) - could be calculated dynamically
                "hidden_dim": forcegnn_config.get('hidden_dim', 32),
                "out_feat": forcegnn_config.get('out_feat', 2),
                "num_layers": forcegnn_config.get('num_layers', 4),
            },
            "forward_func": lambda model, data, adj: model(
                data.x,
                data.edge_index,
                data.batch,  # Add batch information
                data.init_coords if hasattr(data, "init_coords") else None
            )
        },
    }
    
    return MODEL_REGISTRY

def evaluate_model(model_class, checkpoint_path, dataset_path, feature_func, device, model_args, forward_func, batch_size=None):
    
    # Load config for defaults
    config = get_config()
    data_config = config.get_data_config()
    eval_config = config.get_evaluation_config()
    
    # Use config defaults if not provided
    if batch_size is None:
        batch_size = eval_config.get('batch_size', 1)
    
    data_dict = torch.load(dataset_path)
    full_dataset = data_dict['dataset']
    
    # Get data loaders with config settings
    splits = data_config.get('splits', [0.8, 0.1, 0.1])
    random_state = data_config.get('random_state', 42)
    
    train_loader, val_loader, test_loader = force_directed_data_loader(
        dataset=full_dataset,
        batch_size=batch_size,
        splits=splits,
        random_state=random_state
    )
    
    # Instantiate model
    model = model_class(**model_args).to(device)

    # Load weights - Modified this part
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        # If checkpoint contains full training state
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If checkpoint contains only model state
        model.load_state_dict(checkpoint)
    model.eval()

    total_procrustes = 0.0
    total_fr_force = 0.0
    num_graphs = 0

    print("\nStarting evaluation loop...")
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            
            num_edges = data.edge_index.size(1)
            edge_weight = torch.ones(num_edges, dtype=torch.float).to(device)
            data = data.to(device)

            # Adjacency setup
            adj = torch.sparse.FloatTensor(
                data.edge_index,
                edge_weight,
                (data.num_nodes, data.num_nodes)
            ).coalesce()

            # Update spectral radius if needed
            if hasattr(model, "adj_rho"):
                model.adj_rho = get_spectral_rad(adj)

            # Forward pass
            pred = forward_func(model, data, adj)

            # Calculate metrics
            procrustes_loss = criterion_procrustes(pred, data.y).item()
            fr_force = fr_net_force_metric(pred, data.edge_index)

            total_procrustes += procrustes_loss
            total_fr_force += fr_force
            num_graphs += 1

    avg_procrustes = total_procrustes / num_graphs
    avg_fr_force = total_fr_force / num_graphs
    
    return avg_procrustes, avg_fr_force

def main():
    config = get_config()
    eval_config = config.get_evaluation_config()
    
    parser = argparse.ArgumentParser(description="Evaluate multiple GNN models on test dataset")
    parser.add_argument("--device", type=str, 
                       default=eval_config.get('device', 'cuda:0'), 
                       help="Torch device (cuda:0 or cpu)")
    parser.add_argument("--batch-size", type=int, 
                       default=eval_config.get('batch_size', 1), 
                       help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = {}

    # Get model registry from configuration
    MODEL_REGISTRY = get_model_registry()
    
    for model_name, model_config in MODEL_REGISTRY.items():
        avg_procrustes, avg_fr_force = evaluate_model(
            model_class=model_config["class"],
            checkpoint_path=model_config["checkpoint"],
            dataset_path=model_config["data_path"],
            feature_func=model_config["feature_func"],
            device=device,
            model_args=model_config["model_args"],
            forward_func=model_config["forward_func"],
            batch_size=args.batch_size
        )
        results[model_name] = {
            "Procrustes Static": avg_procrustes,
            "FR Net Force": avg_fr_force,
        }

    # Present summary table
    df = pd.DataFrame(results).T
    print("\n--- Evaluation Summary ---")
    print(df)

if __name__ == "__main__":
    main()
