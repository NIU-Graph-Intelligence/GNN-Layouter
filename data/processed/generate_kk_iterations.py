"""
data/processed/generate_kk_iterations.py

Runs a stepped Kamada-Kawai solver and saves every intermediate state,
mirroring generate_fr_iterations.py / generate_fa2_iterations.py in
structure and output schema.

Design decision (flagged, not glossed over): unlike
FR and FA2, classical Kamada-Kawai has no fixed-step iteration at all --
networkx's kamada_kawai_layout is a single black-box scipy.optimize call
over the whole stress function, returning only the endpoint. To get a
trajectory, this file solves the SAME objective (graph-theoretic stress
majorization) by steepest descent instead of BFGS, and takes fixed steps
under FR's own step-size rule -- normalize the descent direction, scale by a
shared linear-cooling temperature -- rather than a line search. That is a
real substitution of solution method, not a detail: it changes convergence
behavior and possibly the fixed point reached vs. scipy's BFGS. It's
defensible here because it's what makes KK an "operator" at all in this
paper's sense (X_{t+1} = Phi(G, X_t, tau_t)), and because it keeps KK in the
same state-space/temperature convention as FR and FA2 --
but it is a paper-relevant modeling choice, not an implementation detail, and
should be described as such wherever this dataset is used.

Objective (classical Kamada-Kawai stress):
    E = sum_{i<j} w_ij * (||p_i - p_j|| - d_ij)^2
    d_ij = k * shortest_path_length(i, j)     -- k = sqrt(1/N), same
                                                  ideal-length convention as
                                                  FR/FA2
    w_ij = 1 / d_ij^2                          -- standard KK weighting

Gradient descent step (this file's substitution for BFGS):
    displacement_i = -sum_j 2*w_ij*(||p_i-p_j|| - d_ij) * (p_i-p_j)/||p_i-p_j||
    pos_i += displacement_i * (temperature / ||displacement_i||)   -- FR's
                                                                       step rule

Usage:
    python data/processed/generate_kk_iterations.py \
        --graphs_file "data/graphs/*.pkl" \
        --output      data/layouts/comm_5k_kk_iters.pkl \
        --n_iter 50 --seed 42
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
# KK single step
# ─────────────────────────────────────────────────────────────────────────────

def kk_single_step(
    pos: np.ndarray,
    d_target: np.ndarray,
    weight: np.ndarray,
    temperature: float,
    eps: float = 0.01,
) -> np.ndarray:
    """One steepest-descent step on the Kamada-Kawai stress objective, under
    FR's step-size rule.

    Parameters
    ----------
    pos         : [N, 2]  current positions
    d_target    : [N, N]  target (graph-distance-scaled) pairwise distances,
                          zero diagonal
    weight      : [N, N]  1/d_target^2, zero diagonal
    temperature : float   current temperature (step-size cap, shared
                          schedule with FR/FA2)

    Returns
    -------
    new_pos : [N, 2]
    """
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # [N, N, 2]
    dist = np.linalg.norm(delta, axis=-1)                   # [N, N]
    np.clip(dist, eps, None, out=dist)

    diff = dist - d_target                                  # [N, N]
    # Negative gradient direction: pulls together when dist > target
    # (diff > 0), pushes apart when dist < target (diff < 0).
    signed_force_mag = -2.0 * weight * diff / dist           # [N, N]

    displacement = np.einsum("ijk,ij->ik", delta, signed_force_mag)  # [N, 2]

    length = np.linalg.norm(displacement, axis=-1)
    length = np.where(length < eps, 0.1, length)
    delta_pos = np.einsum("ij,i->ij", displacement, temperature / length)

    return pos + delta_pos


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (duplicated from generate_fr_iterations.py -- see that file's note)
# ─────────────────────────────────────────────────────────────────────────────

def _grid_initial_positions(G: nx.Graph) -> np.ndarray:
    """Grid init in [-1, 1]^2, identical to generate_fr_iterations.py."""
    nodes = sorted(G.nodes())
    n     = len(nodes)
    grid_size = int(np.ceil(np.sqrt(n)))
    rows      = int(np.ceil(n / grid_size))

    pos = np.zeros((n, 2), dtype=np.float64)
    for idx in range(n):
        u = (idx % grid_size) / max(grid_size - 1, 1)
        v = (idx // grid_size) / max(rows - 1, 1)
        pos[idx, 0] = 2 * u - 1
        pos[idx, 1] = 2 * v - 1
    return pos


def _shortest_path_matrix(G: nx.Graph, nodes: list) -> np.ndarray:
    """All-pairs unweighted shortest path length, [N, N]. Disconnected pairs
    get N (the graph's own node count) as a finite fallback distance rather
    than inf -- these are small connected-by-construction community graphs,
    but this keeps the generator from producing NaNs if that ever isn't true."""
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    D = np.full((n, n), float(n), dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    for source, lengths in nx.all_pairs_shortest_path_length(G):
        i = idx[source]
        for target, d in lengths.items():
            D[i, idx[target]] = d
    return D


def run_kk_with_snapshots(G: nx.Graph, n_iter: int = 50) -> Dict:
    """Run stepped KK for n_iter steps, saving a snapshot at every step."""
    nodes = sorted(G.nodes())
    n     = len(nodes)

    sp = _shortest_path_matrix(G, nodes)                     # [N, N]
    k  = np.sqrt(1.0 / n)                                     # same ideal-length convention as FR/FA2
    d_target = k * sp
    weight = np.zeros_like(d_target)
    nz = d_target > 0
    weight[nz] = 1.0 / (d_target[nz] ** 2)

    pos = _grid_initial_positions(G)
    coord_span = pos.max(axis=0) - pos.min(axis=0)
    t0 = float(np.max(coord_span) * 0.1)
    t0 = max(t0, 1e-4)
    dt = t0 / (n_iter + 1)

    steps       = []
    temperature = t0

    for step in range(n_iter):
        pos_before = pos.copy()
        pos        = kk_single_step(pos, d_target, weight, temperature)

        steps.append({
            "step":        step,
            "pos_before":  pos_before.astype(np.float32),
            "pos_after":   pos.astype(np.float32),
            "temperature": float(temperature),
        })

        temperature = max(temperature - dt, 1e-6)

    return {"n_nodes": n, "k": float(k), "t0": float(t0), "steps": steps}


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


def save_iterations(records: List[Dict], output_path: str, n_iter: int, seed: int):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"graphs": records, "n_iter": n_iter, "seed": seed}, f)
    print(f"Saved {len(records)} graph trajectories → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_file", required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--n_iter",  type=int, default=50)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--max_graphs", type=int, default=None)
    args = parser.parse_args()

    all_graphs = load_graphs(args.graphs_file)
    if args.max_graphs:
        all_graphs = all_graphs[: args.max_graphs]

    print(f"\nTotal graphs: {len(all_graphs)}  |  Iterations: {args.n_iter}")
    print(f"Total training samples: {len(all_graphs) * args.n_iter}\n")

    records = []
    for idx, G in enumerate(tqdm(all_graphs, desc="Generating KK trajectories")):
        try:
            result             = run_kk_with_snapshots(G, n_iter=args.n_iter)
            result["graph_idx"] = idx
            result["graph_id"]  = G.graph.get("id", f"G_{idx:05d}")
            records.append(result)
        except Exception as e:
            print(f"\n[WARN] Skipped graph {idx}: {e}")

    save_iterations(records, args.output, n_iter=args.n_iter, seed=args.seed)

    sizes = [r["n_nodes"] for r in records]
    print(f"\nSummary:")
    print(f"  Graphs processed       : {len(records)}")
    print(f"  Node count range       : {min(sizes)} – {max(sizes)}")
    print(f"  Total training samples : {sum(len(r['steps']) for r in records)}")


if __name__ == "__main__":
    main()
