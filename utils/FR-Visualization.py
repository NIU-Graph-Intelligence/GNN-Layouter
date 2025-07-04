import os
import sys
import pickle
import argparse
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# allow imports from your project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(PROJECT_ROOT)

# import your three model classes
from models_PPI         import IGNN
from utils             import get_spectral_rad
from models.GCNFR import ForceGNN



def visualize_all(
    ignn_weights,
    forcegnn_weights,
    data_path,
    num_samples,
    output_dir,
    seed,
    raw_dataset="data/raw/graph_dataset/community_graphs_dataset1024.pkl"
):
    # ─── Setup ───────────────────────────────────────────────────────────────
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load processed dataset
    data_dict = torch.load(data_path, map_location='cpu')
    graphs    = data_dict['dataset']
    max_nodes = data_dict['max_nodes']
    print(f"Loaded {len(graphs)} graphs; max_nodes={max_nodes}")

    # load raw community assignments
    with open(raw_dataset, 'rb') as f:
        raw = pickle.load(f)

    comm_map = {g['graph_id']: g['community'] for g in raw}

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ─── Instantiate & load models ──────────────────────────────────────────
    feature_dim = graphs[0].x.shape[1] + graphs[0].init_coords.shape[1]

    print(feature_dim)
    # IGNN
    ignn = IGNN(
        nfeat    = feature_dim,
        nhid     = 64,
        nclass   = 2,
        num_node = max_nodes,
        dropout  = 0.2,
        kappa    = 0.8
    ).to(device)
    ignn.load_state_dict(torch.load(ignn_weights, map_location=device))
    ignn.eval()
    print("Loaded IGNN from", ignn_weights)


    #ForceGNN
    print("\nInitializing ForceGNN:")
    print(f"in_feat={feature_dim}, hidden_dim=64, out_feat=2, num_layers=4")
    forcegnn = ForceGNN(
        in_feat=feature_dim,
        hidden_dim=64,
        out_feat=2,
        num_layers=4,
    ).to(device)
    print("Loading weights from:", forcegnn_weights)
    forcegnn.load_state_dict(torch.load(forcegnn_weights, map_location=device))
    forcegnn.eval()
    print("Loaded ForceGNN from", forcegnn_weights)


    models      = [ignn, forcegnn]
    model_names = ["IGNN", "ForceGNN"]

    # ─── Sample graphs ──────────────────────────────────────────────────────
    num_samples = min(num_samples, len(graphs))
    indices     = np.random.choice(len(graphs), num_samples)
    os.makedirs(output_dir, exist_ok=True)
    # selected_ids = ["graph_14", "graph_168", "graph_365", "graph_954", "graph_1138"]
    # id_to_idx = {g.graph_id: i for i, g in enumerate(graphs)}
    # indices = [id_to_idx[gid] for gid in selected_ids if gid in id_to_idx]
    # if len(indices) < len(selected_ids):
    #     missing = set(selected_ids) - set(id_to_idx.keys())
    #     print(f"Warning: The following IDs were not found in the loaded dataset: {missing}")
    # num_samples = len(indices)
    # os.makedirs(output_dir, exist_ok=True)


    # ─── Setup figure ───────────────────────────────────────────────────────
    cols = 1 + len(models)
    fig  = plt.figure(figsize=(5*cols, 5*num_samples))
    gs   = GridSpec(num_samples, cols, figure=fig)

    # Create a batch tensor for single graph visualization
    batch = torch.zeros(max_nodes, dtype=torch.long, device=device)

    for row, idx in enumerate(indices):
        print(f"\nProcessing graph {idx}")
        
        data = graphs[idx].to(device)
        print(f"Graph data: nodes={data.num_nodes}, edges={data.edge_index.size(1)}")
        
        num_edges = data.edge_index.size(1)
        edge_weight = torch.ones(num_edges, dtype=torch.float).to(device)

        # build adjacency
        adj = torch.sparse.FloatTensor(
            data.edge_index,
            edge_weight,
            (data.num_nodes, data.num_nodes)
        ).coalesce()
        rho = get_spectral_rad(adj)
        for m in models:
            m.adj_rho = rho

        # ground-truth coords
        true = data.y.cpu().numpy()
        print(f"Ground truth shape: {true.shape}")

        # community coloring
        communities = comm_map[data.graph_id]
        orig_ids = data.orig_node_ids.cpu().tolist()
        comm_labels = [communities[n] for n in orig_ids]
        
        newlist = []
        for n in comm_labels:
            newlist.append(int(n))
        
        uniq = sorted(set(newlist))
        distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        c2i = {c:i for i,c in enumerate(uniq)}
        node_colors = [distinct_colors[c2i[c]] for c in newlist]

        # build NX graph
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        G.add_edges_from(data.edge_index.cpu().numpy().T)
        print(f"NetworkX graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        # Ground Truth visualization
        ax = fig.add_subplot(gs[row, 0])
        pos_gt = {i: tuple(true[i]) for i in range(data.num_nodes)}
        nx.draw(G, pos=pos_gt, ax=ax,
                node_color=node_colors, edge_color='gray',
                node_size=80, width=0.5, alpha=0.8,
                with_labels=False)
        ax.set_title(f"{data.graph_id}\nGround Truth")
        ax.set_aspect('equal')
        ax.axis('off')

        # Model predictions
        for col, (m, name) in enumerate(zip(models, model_names), start=1):
            print(f"\nProcessing model: {name}")
            try:
                if name == "IGNN":
                    feat = torch.cat([data.x, data.init_coords], dim=1).T
                    pred = m(feat, adj)
                    print(f"IGNN prediction shape: {pred.shape}")
                    pred = pred.detach().cpu().numpy()
                elif name == "ForceGNN":
                    # Debug input shapes
                    print(f"Input shapes:")
                    print(f"x shape: {data.x.shape}")
                    print(f"edge_index shape: {data.edge_index.shape}")
                    print(f"batch shape: {batch.shape}")
                    print(f"init_coords shape: {data.init_coords.shape}")
                    
                    pred = m(data.x, data.edge_index, batch, data.init_coords)
                    print(f"ForceGNN raw prediction shape: {pred.shape}")
                    pred = pred.squeeze(0).detach().cpu().numpy()
                    print(f"ForceGNN final prediction shape: {pred.shape}")

                # normalize for display
                p = pred - pred.mean(axis=0, keepdims=True)
                print(f"After centering shape: {p.shape}")
                
                s = max(p[:,0].ptp(), p[:,1].ptp())
                if s>1e-6: 
                    p /= s
                print(f"Final normalized shape: {p.shape}")
                
                pos_p = {i: tuple(p[i]) for i in range(data.num_nodes)}
                print(f"Created position dictionary with {len(pos_p)} positions")

                axp = fig.add_subplot(gs[row, col])
                nx.draw(G, pos=pos_p, ax=axp,
                        node_color=node_colors, edge_color='gray',
                        node_size=80, width=0.5, alpha=0.8,
                        with_labels=False)
                axp.set_title(name)
                axp.set_aspect('equal')
                axp.axis('off')
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Create an empty subplot with error message
                axp = fig.add_subplot(gs[row, col])
                axp.text(0.5, 0.5, f"Error: {str(e)}", 
                        ha='center', va='center', wrap=True)
                axp.set_title(f"{name} (Failed)")
                axp.axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "PEComparisonReadoutsNode40.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved comparison_grid.png in", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare GT vs IGNN vs FORCEGNN"
    )
    parser.add_argument('--ignn',   type=str, default='IGNN_checkpoints/PEIGNN_bs16_ep1000_HC64WithoutInputOneHot.pt',
                        help="Path to IGNN weights")
    parser.add_argument('--forcegnn', type=str, default='results/metrics/ForceGNN/PEWeights_ForceGNN_FR_batch16.pt',
                        help="Path to ForceGNN weights")
    parser.add_argument('--data',   type=str, default='data/processed/modelInput_FRgraphs1024_40Nodes_withPE.pt',
                        help="Processed .pt dataset")
    parser.add_argument('--samples',type=int, default=5,
                        help="Number of graphs to visualize")
    parser.add_argument('--outdir', type=str, default='comparisons',
                        help="Where to save the image")
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    visualize_all(
        ignn_weights=args.ignn,
        forcegnn_weights=args.forcegnn,
        data_path=args.data,
        num_samples=args.samples,
        output_dir=args.outdir,
        seed=args.seed
    )
