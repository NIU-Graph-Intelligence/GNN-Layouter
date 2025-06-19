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
    frgnn_weights,
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
    feature_dim = graphs[0].x.shape[1] + 2

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
    forcegnn = ForceGNN(
        in_feat=feature_dim,
        hidden_dim=128,
        out_feat=2,
        num_layers=3,
    ).to(device)
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

    for row, idx in enumerate(indices):

        data = graphs[idx].to(device)
        num_edges = data.edge_index.size(1)
        edge_weight = torch.ones(num_edges, dtype=torch.float).to(device)

        # build adjacency
        adj = torch.sparse.FloatTensor(
            data.edge_index,
            edge_weight,
            # data.edge_attr.view(-1),
            (data.num_nodes, data.num_nodes)
        ).coalesce()
        rho = get_spectral_rad(adj)
        for m in models:
            m.adj_rho = rho

        # ground-truth coords
        true = data.y.cpu().numpy()
        print(data)
        # community coloring
        communities = comm_map[data.graph_id]
        orig_ids    = data.orig_node_ids.cpu().tolist()
        comm_labels = [communities[n] for n in orig_ids]   # shift from 0‐based to 1‐based

        newlist = []
        for n in comm_labels:
            newlist.append(int(n))

        uniq = sorted(set(newlist))

        # Use explicit distinct colors for better visualization
        distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        c2i = {c:i for i,c in enumerate(uniq)}
        node_colors = [distinct_colors[c2i[c]] for c in newlist]

        # build NX graph
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        G.add_edges_from(data.edge_index.cpu().numpy().T)

        # ── Column 0: Ground Truth ────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 0])
        pos_gt = {i: tuple(true[i]) for i in range(data.num_nodes)}
        nx.draw(G, pos=pos_gt, ax=ax,
                node_color=node_colors, edge_color='gray',
                node_size=80, width=0.5, alpha=0.8,
                with_labels=False)
        ax.set_title(f"{data.graph_id}\nGround Truth")
        ax.set_aspect('equal'); ax.axis('off')

        # ── Columns 1–3: predictions ─────────────────────────────────────────
        for col, (m, name) in enumerate(zip(models, model_names), start=1):

            if name == "IGNN":
                feat = torch.cat([data.x, data.init_coords], dim=1).T
                pred = m(feat, adj).detach().cpu().numpy()
            elif name == "ForceGNN":
                # feat = torch.cat([data.x, data.init_coords], dim=1).T
                pred = m(data.x, data.edge_index, data.init_coords).detach().cpu().numpy()
            else:
                # FRGNN and FRGAT want the whole Data object
                pred = m(data).detach().cpu().numpy()


            # normalize for display
            p = pred - pred.mean(axis=0, keepdims=True)
            s = max(p[:,0].ptp(), p[:,1].ptp())
            if s>1e-6: p /= s
            pos_p = {i: tuple(p[i]) for i in range(data.num_nodes)}

            axp = fig.add_subplot(gs[row, col])
            nx.draw(G, pos=pos_p, ax=axp,
                    node_color=node_colors, edge_color='gray',
                    node_size=80, width=0.5, alpha=0.8,
                    with_labels=False)
            axp.set_title(name)
            axp.set_aspect('equal'); axp.axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "HC64comparison_grid.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved comparison_grid.png in", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare GT vs IGNN vs FRGNN vs FRGAT"
    )
    parser.add_argument('--ignn',   type=str, default='IGNN_checkpoints/IGNN_bs16_ep1000HC64.pt',
                        help="Path to IGNN weights")
    parser.add_argument('--frgnn',  type=str, default='results/metrics/FRGNN/FinalWeights_FRGNN_FR_batch128.pt',
                        help="Path to FRGNN weights")
    # parser.add_argument('--frgat',  type=str, default='results/metrics/FRGAT/FinalWeights_FRGAT_FR_batch128.pt',
    #                     help="Path to FRGAT weights")
    parser.add_argument('--forcegnn', type=str, default='results/metrics/ForceGNN/Weights_ForceGNN_FR_batch128.pt',
                        help="Path to ForceGNN weights")
    parser.add_argument('--data',   type=str, default='data/processed/modelInput_FR1024.pt',
                        help="Processed .pt dataset")
    parser.add_argument('--samples',type=int, default=5,
                        help="Number of graphs to visualize")
    parser.add_argument('--outdir', type=str, default='comparisons',
                        help="Where to save the image")
    parser.add_argument('--seed',   type=int, default=42)
    args = parser.parse_args()

    visualize_all(
        ignn_weights=args.ignn,
        frgnn_weights=args.frgnn,
        forcegnn_weights=args.forcegnn,
        data_path=args.data,
        num_samples=args.samples,
        output_dir=args.outdir,
        seed=args.seed
    )
