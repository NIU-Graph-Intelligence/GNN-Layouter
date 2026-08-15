"""
data/generate_fr_iterations.py

Runs Fruchterman-Reingold step-by-step and saves every intermediate state.
Each graph contributes n_iter training samples of the form:
    (pos_t, graph, k, temperature) -> pos_{t+1}

The FR implementation mirrors NetworkX exactly:
  - unweighted adjacency matrix (weight=None)
  - grid initial positions in [-1, 1]²
  - linear temperature cooling: t0 / (n_iter + 1) per step
  - t0 = 0.1 * max coordinate span of initial positions

Output (pickle):
    {
      'graphs': [{
          'graph_idx': int,
          'graph_id':  str,
          'n_nodes':   int,
          'k':         float,   # sqrt(1/N)
          't0':        float,   # initial temperature
          'steps': [{
              'step':        int,
              'pos_before':  np.ndarray [N, 2],  float64
              'pos_after':   np.ndarray [N, 2],  float64
              'temperature': float,
          }, ...]
      }, ...],
      'n_iter': int,
      'seed':   int,
    }

Usage:
    python data/generate_fr_iterations.py \
        --graphs_file "data/graphs/*.pkl" \
        --output      data/layouts/comm_5k_fr_iters.pkl \
        --n_iter 50 --verify
"""

import argparse
import glob
import os
import pickle
import sys
from typing import Dict, List

import networkx as nx
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# FR single step
# ─────────────────────────────────────────────────────────────────────────────

def fr_single_step(
    pos: np.ndarray,
    A:   np.ndarray,
    k:   float,
    temperature: float,
    eps: float = 0.01,
) -> np.ndarray:
    """One step of Fruchterman-Reingold (mirrors NetworkX exactly).

    Parameters
    ----------
    pos         : [N, 2]  current positions (float64)
    A           : [N, N]  unweighted adjacency matrix (0/1, float64)
    k           : float   optimal distance = sqrt(1/N)
    temperature : float   current temperature
    eps         : float   minimum distance clip (NetworkX default: 0.01)
    """
    delta    = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]    # [N, N, 2]
    distance = np.linalg.norm(delta, axis=-1)                   # [N, N]
    np.clip(distance, eps, None, out=distance)

    force_mag    = k * k / distance ** 2 - A * distance / k    # [N, N]
    displacement = np.einsum("ijk,ij->ik", delta, force_mag)    # [N, 2]

    length = np.linalg.norm(displacement, axis=-1)              # [N]
    length = np.where(length < eps, 0.1, length)
    delta_pos = np.einsum("ij,i->ij", displacement, temperature / length)

    return pos + delta_pos


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grid_initial_positions(G: nx.Graph) -> np.ndarray:
    """Grid init in [-1, 1]², the repo-wide initial frame."""
    nodes     = sorted(G.nodes())
    n         = len(nodes)
    grid_size = int(np.ceil(np.sqrt(n)))
    rows      = int(np.ceil(n / grid_size))

    pos = np.zeros((n, 2), dtype=np.float64)
    for idx in range(n):
        u = (idx % grid_size) / max(grid_size - 1, 1)
        v = (idx // grid_size) / max(rows - 1, 1)
        pos[idx, 0] = 2 * u - 1
        pos[idx, 1] = 2 * v - 1
    return pos


def run_fr_with_snapshots(G: nx.Graph, n_iter: int = 50) -> Dict:
    """Run FR for n_iter steps, saving a snapshot after every step."""
    nodes = sorted(G.nodes())
    n     = len(nodes)

    # Unweighted adjacency — weight=None ignores LFR edge weights
    A   = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64, weight=None)
    pos = _grid_initial_positions(G)

    k          = np.sqrt(1.0 / n)
    coord_span = pos.max(axis=0) - pos.min(axis=0)
    t0         = float(np.max(coord_span) * 0.1)
    t0         = max(t0, 1e-4)
    dt         = t0 / (n_iter + 1)

    steps       = []
    temperature = t0

    for step in range(n_iter):
        pos_before = pos.copy()
        pos        = fr_single_step(pos, A, k, temperature)

        steps.append({
            "step":        step,
            "pos_before":  pos_before,   # float64 — no precision loss
            "pos_after":   pos.copy(),   # float64
            "temperature": float(temperature),
        })

        temperature = max(temperature - dt, 1e-6)

    return {"n_nodes": n, "k": float(k), "t0": float(t0), "steps": steps}


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_against_networkx(G: nx.Graph, n_iter: int = 50) -> float:
    """Returns max absolute deviation from NetworkX spring_layout."""
    nodes  = sorted(G.nodes())
    result = run_fr_with_snapshots(G, n_iter=n_iter)
    our_final = result["steps"][-1]["pos_after"]

    init_pos_dict = {
        node: _grid_initial_positions(G)[i].tolist()
        for i, node in enumerate(nodes)
    }
    nx_pos = nx.spring_layout(
        G, pos=init_pos_dict, iterations=n_iter,
        seed=42, scale=None, threshold=0,
    )
    nx_final = np.array([nx_pos[node] for node in nodes], dtype=np.float64)
    return float(np.max(np.abs(our_final - nx_final)))


# ─────────────────────────────────────────────────────────────────────────────
# I/O
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
        print(f"  Loaded {len(graphs)} graphs from {os.path.basename(path)}")
    return all_graphs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_file", required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--n_iter",      type=int, default=50)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--verify",      action="store_true",
                        help="Compare first graph against NetworkX spring_layout.")
    parser.add_argument("--max_graphs",  type=int, default=None)
    args = parser.parse_args()

    all_graphs = load_graphs(args.graphs_file)
    if args.max_graphs:
        all_graphs = all_graphs[: args.max_graphs]

    print(f"\nTotal graphs: {len(all_graphs)}  |  Iterations: {args.n_iter}")
    print(f"Total training samples: {len(all_graphs) * args.n_iter}\n")

    if args.verify and all_graphs:
        print("Verifying against NetworkX spring_layout …")
        dev = verify_against_networkx(all_graphs[0], n_iter=args.n_iter)
        status = "OK" if dev < 1e-4 else "WARNING: larger than expected"
        print(f"  Max absolute deviation: {dev:.2e}  [{status}]\n")

    records = []
    for idx, G in enumerate(tqdm(all_graphs, desc="Generating FR trajectories")):
        try:
            result              = run_fr_with_snapshots(G, n_iter=args.n_iter)
            result["graph_idx"] = idx
            result["graph_id"]  = G.graph.get("id", f"G_{idx:05d}")
            records.append(result)
        except Exception as e:
            print(f"\n[WARN] Skipped graph {idx}: {e}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"graphs": records, "n_iter": args.n_iter, "seed": args.seed}, f)

    sizes = [r["n_nodes"] for r in records]
    print(f"\nSaved {len(records)} graph trajectories → {args.output}")
    print(f"  Node count range       : {min(sizes)} – {max(sizes)}")
    print(f"  Total training samples : {sum(len(r['steps']) for r in records)}")


if __name__ == "__main__":
    main()
