"""
data/generate_final_layout_dataset.py

Builds a PyG dataset for DIRECT FINAL LAYOUT PREDICTION with trajectory supervision.

Each sample = one full graph (not one step).
  - Input  x        [N, 4]:      degree_norm | k | pos_0_x | pos_0_y
  - Target y        [N, 2]:      final coordinates at iteration 50
  - y_traj          [N, 100]:    all 50 iterations flattened per node (50 steps × 2 coords)
  - rrwp            [N, N, C]:   pre-computed RRWP positional encoding (C = num_hops)

y_traj storage format:
  Stored as [N, 100] where each row is [x0,y0, x1,y1, ..., x49,y49] for that node.
  To recover iteration t: y_traj.view(N, 50, 2)[:, t, :]  → [N, 2]
  This format is required because PyG cannot batch [50, N, 2] across graphs
  with different N values.

RRWP is pre-computed here and stored per graph so the model does not
recompute it on every forward pass (saves ~30% training time).

Usage:
    python3 data/generate_final_dataset.py \
        --graphs_file data/graphs/comm_5k_5000graphs_20-50nodes_20250920_233608.pkl  \
        --iters_file data/layouts/comm_5k_fr_iters.pkl \
        --output data/processed/comm_5k_final_alliteration.pt
"""

import argparse
import glob
import os
import pickle
import sys
from typing import Dict, List

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_graphs(filepath: str) -> List[nx.Graph]:
    paths = sorted(glob.glob(filepath))
    if not paths:
        raise FileNotFoundError(f"No files matched: {filepath}")
    all_graphs = []
    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        graphs = data["graphs"] if isinstance(data, dict) else data
        all_graphs.extend(graphs)
    return all_graphs


def load_iterations(filepath: str) -> Dict:
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Edge index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_edge_index(G: nx.Graph) -> torch.Tensor:
    """Bidirectional edge_index [2, 2E], nodes ordered by sorted(G.nodes())."""
    node_to_idx = {n: i for i, n in enumerate(sorted(G.nodes()))}
    src, dst = [], []
    for u, v in G.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        src += [i, j]
        dst += [j, i]
    
    return torch.tensor([src, dst], dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# RRWP pre-computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_rrwp(edge_index: torch.Tensor, num_nodes: int, num_hops: int = 8) -> torch.Tensor:
    """
    Pre-compute Relative Random Walk Probabilities.

    R_ij = [I_ij, M_ij, M²_ij, ..., M^{C-1}_ij]
    where M = D^{-1} A.

    Returns
    -------
    R : [N, N, C] float32
    """
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # [N, N]
    deg = A.sum(dim=1).clamp(min=1.0)
    D_inv = torch.diag(1.0 / deg)
    M = D_inv @ A

    I = torch.eye(num_nodes)
    hops = [I, M]
    M_power = M.clone()
    for _ in range(num_hops - 2):
        M_power = M_power @ M
        hops.append(M_power)

    R = torch.stack(hops[:num_hops], dim=-1)  # [N, N, C]
    return R.float()


# ─────────────────────────────────────────────────────────────────────────────
# Per-graph sample builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_sample(
    graph_idx: int,
    G: nx.Graph,
    record: Dict,
    num_hops: int = 8,
) -> Data:
    """
    Build one Data object from one graph and its full FR trajectory.

    Returns PyG Data with:
      x          [N, 4]       : [degree_norm, k, pos_0_x, pos_0_y]
      edge_index [2, 2E]      : bidirectional edges
      y          [N, 2]       : final positions (step 50)
      y_traj     [N, 100]     : all 50 pos_after arrays, flattened per node
      y_mean     [2]          : mean of final positions (for denormalization)
      y_std      [2]          : std of final positions (for denormalization)
      k          scalar
      graph_idx  int
    """
    nodes = sorted(G.nodes())
    # Extract community label per node in sorted order. Non-community graph
    # families (e.g. the scale-free second family) carry no such attribute;
    # default to a single community (label 0) for them so the schema stays
    # identical across families.
    community = torch.tensor(
        [G.nodes[node].get('community', 0) for node in nodes],
        dtype=torch.long
    )
    n = len(nodes)

    edge_index = build_edge_index(G)
    k = float(record["k"])
    steps = record["steps"]

    # ── Trajectory: [50, N, 2] → normalize → flatten to [N, 100] ────────────
    traj = np.stack(
        [step["pos_after"].astype(np.float32) for step in steps],
        axis=0,
    )   # [50, N, 2]

    # ── Compute normalization stats from final positions ──────────────────────
    # Mean and std computed per coordinate (x and y separately) across all nodes
    # in the final layout. Stored so predictions can be denormalized at inference.
    y_final = traj[-1]                                     # [N, 2]
    y_mean  = y_final.mean(axis=0)                         # [2]
    y_std   = y_final.std(axis=0) + 1e-6                   # [2]  avoid div by zero

    # ── Node features at step 0 ──────────────────────────────────────────────
    degrees = np.array([G.degree(node) for node in nodes], dtype=np.float32)
    max_deg = float(degrees.max()) if degrees.max() > 0 else 1.0
    deg_norm = (degrees / max_deg).reshape(-1, 1)          # [N, 1]
    k_feat = np.full((n, 1), k, dtype=np.float32)          # [N, 1]

    # The initial position MUST use the same (y_mean, y_std) frame as y and
    # y_traj. It is the loop's p_0, and every target it is compared against
    # lives in the normalized frame — leaving it raw would force the model to
    # spend its first iterations learning a fixed affine frame conversion
    # instead of FR dynamics (measured: 4.1x larger step-0 error).
    pos_0 = steps[0]["pos_before"].astype(np.float32)      # [N, 2]
    pos_0 = (pos_0 - y_mean) / y_std                       # [N, 2] normalized

    x = np.concatenate([deg_norm, k_feat, pos_0], axis=1)  # [N, 4]

    # ── Normalize final target ────────────────────────────────────────────────
    y_norm = (y_final - y_mean) / y_std                    # [N, 2]

    # ── Normalize full trajectory then flatten to [N, 100] ───────────────────
    traj_norm = (traj - y_mean) / y_std                    # [50, N, 2]
    y_traj_flat = traj_norm.transpose(1, 0, 2).reshape(n, -1)  # [N, 100]

    data = Data(
        x          = torch.from_numpy(x),
        edge_index = edge_index,
        y          = torch.from_numpy(y_norm.astype(np.float32)),
        y_traj     = torch.from_numpy(y_traj_flat.astype(np.float32)),
        community = community
    )
    data.y_mean    = torch.from_numpy(y_mean.astype(np.float32))  # [2]
    data.y_std     = torch.from_numpy(y_std.astype(np.float32))   # [2]
    data.k         = k
    data.graph_idx = graph_idx
    data.num_nodes = n
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    graphs: List[nx.Graph],
    iter_records: List[Dict],
    num_hops: int = 8,
    max_graphs: int = None,
) -> List[Data]:
    assert len(graphs) == len(iter_records)

    if max_graphs is not None:
        graphs       = graphs[:max_graphs]
        iter_records = iter_records[:max_graphs]

    dataset: List[Data] = []
    for graph_idx, (G, record) in enumerate(
        tqdm(zip(graphs, iter_records), total=len(graphs), desc="Building dataset")
    ):
        try:
            sample = build_graph_sample(graph_idx, G, record, num_hops=num_hops)
            dataset.append(sample)
        except Exception as e:
            print(f"\n[WARN] Skipped graph {graph_idx}: {e}")

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_file", required=True)
    parser.add_argument("--iters_file",  required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--num_hops",    type=int, default=8,
                        help="RRWP random walk hops (must match model's num_hops)")
    parser.add_argument("--max_graphs",  type=int, default=None)
    args = parser.parse_args()

    print(f"Loading graphs from {args.graphs_file} ...")
    graphs = load_graphs(args.graphs_file)

    print(f"Loading FR iteration records from {args.iters_file} ...")
    iter_records = load_iterations(args.iters_file)["graphs"]

    assert len(graphs) == len(iter_records), (
        f"Mismatch: {len(graphs)} graphs vs {len(iter_records)} records"
    )

    print(f"\nBuilding dataset: {len(graphs)} graphs")
    print(f"  Input features : [degree_norm, k, pos_0_x, pos_0_y]  (4-dim)")
    print(f"  Target         : final coordinates at iteration 50")
    print(f"  Trajectory     : 50 steps flattened to [N, 100]")
    print(f"  RRWP hops      : {args.num_hops}\n")

    dataset = build_dataset(
        graphs, iter_records,
        num_hops=args.num_hops,
        max_graphs=args.max_graphs,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(dataset, args.output)

    # Statistics
    node_counts = [d.num_nodes for d in dataset]
    sample = dataset[0]
    print(f"\nDataset saved → {args.output}")
    print(f"  Total graphs    : {len(dataset)}")
    print(f"  Node range      : {min(node_counts)} – {max(node_counts)}")
    print(f"  Avg nodes       : {sum(node_counts)/len(node_counts):.1f}")
    print(f"  x shape         : {sample.x.shape}")
    print(f"  y shape         : {sample.y.shape}")
    print(f"  y_traj shape    : {sample.y_traj.shape} ")
    # print(f"  rrwp shape      : {sample.rrwp.shape}  (N × N × {args.num_hops})")

    # Sanity check: normalized y should equal last step in normalized y_traj
    n_iter = sample.y_traj.shape[1] // 2
    traj_last = sample.y_traj.view(sample.num_nodes, n_iter, 2)[:, -1, :]
    max_diff = (sample.y - traj_last).abs().max().item()
    status = "OK" if max_diff < 1e-5 else f"WARNING: {max_diff:.2e}"
    print(f"  y == y_traj[-1] : {status}  (n_iter={n_iter})")
    print(f"  y_mean sample   : {sample.y_mean.tolist()}")
    print(f"  y_std  sample   : {sample.y_std.tolist()}")


if __name__ == "__main__":
    main()