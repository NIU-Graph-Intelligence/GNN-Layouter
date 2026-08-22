"""
data/processed/generate_fa2_iterations.py

Runs ForceAtlas2 step-by-step and saves every intermediate state, mirroring
generate_fr_iterations.py exactly in structure and output schema so
generate_final_dataset.py can consume either without modification.

FA2 is the same operator form as FR with different force laws:
Barnes-Hut changes how the repulsive sum is computed, not
what the state is. This generator keeps FR's exact machinery (grid init,
linear-cooling temperature schedule, the same
displacement = displacement_vector * (temperature / ||displacement_vector||)
step rule) and only swaps the force law:

    repulsion (i,j): k_r * mass_i * mass_j / distance   (mass = degree + 1)
    attraction (edge i,j): distance                      (linear, non-LinLog)
    gravity: k_g * mass_i, pulling toward the origin

k_r=2.0, k_g=1.0 are Gephi's own scalingRatio/gravity defaults, kept as-is
rather than re-tuned for this graph scale -- if trajectories look degenerate
in the sanity check below, that's the first thing to revisit.

Deliberately NOT implemented: Gephi's adaptive global-speed algorithm
(swinging/traction-based per-iteration speed control, Jacomy et al. 2014
Algorithm 1). Reusing FR's linear-cooling schedule instead keeps every
teacher in the same state-space/temperature convention, which is what makes
multi-teacher conditioning cost almost nothing in formalization --
mismatched per-teacher temperature semantics would
undercut exactly the thing this generator exists to support.

Output schema: identical to generate_fr_iterations.py's pickle (n_nodes, k,
t0, steps: [{step, pos_before, pos_after, temperature}]).

Usage:
    python data/processed/generate_fa2_iterations.py \
        --graphs_file "data/graphs/*.pkl" \
        --output      data/layouts/comm_5k_fa2_iters.pkl \
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
# FA2 single step
# ─────────────────────────────────────────────────────────────────────────────

def fa2_single_step(
    pos: np.ndarray,
    A: np.ndarray,
    degrees: np.ndarray,
    temperature: float,
    k_r: float = 2.0,
    k_g: float = 1.0,
    eps: float = 0.01,
) -> np.ndarray:
    """One step of ForceAtlas2, in FR's step-rule shape.

    Parameters
    ----------
    pos         : [N, 2]  current positions
    A           : [N, N]  UNWEIGHTED adjacency matrix (0/1)
    degrees     : [N]     node degree, used for FA2's per-node mass
    temperature : float   current temperature (step-size cap, shared
                           schedule with FR -- see module docstring)
    k_r, k_g    : ForceAtlas2 scalingRatio / gravity constants

    Returns
    -------
    new_pos : [N, 2]
    """
    delta    = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]   # [N, N, 2]
    distance = np.linalg.norm(delta, axis=-1)                  # [N, N]
    np.clip(distance, eps, None, out=distance)

    mass       = degrees + 1.0                                 # [N]
    mass_outer = mass[:, np.newaxis] * mass[np.newaxis, :]      # [N, N]

    # repulsion: k_r * mass_i * mass_j / distance  (all pairs)
    # attraction: distance  (connected pairs only, unweighted, linear)
    force_mag = k_r * mass_outer / distance - A * distance      # [N, N]

    displacement = np.einsum("ijk,ij->ik", delta, force_mag)    # [N, 2]

    # gravity: pulls every node toward the origin with magnitude k_g * mass_i
    pos_norm = np.linalg.norm(pos, axis=-1, keepdims=True)
    np.clip(pos_norm, eps, None, out=pos_norm)
    gravity = -k_g * mass[:, np.newaxis] * pos / pos_norm       # [N, 2]
    displacement = displacement + gravity

    length = np.linalg.norm(displacement, axis=-1)              # [N]
    length = np.where(length < eps, 0.1, length)
    delta_pos = np.einsum("ij,i->ij", displacement, temperature / length)

    return pos + delta_pos


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (duplicated from generate_fr_iterations.py rather than imported --
# data/processed/ has no __init__.py, and every generator script in this
# directory is already self-contained this way)
# ─────────────────────────────────────────────────────────────────────────────

def _grid_initial_positions(G: nx.Graph) -> np.ndarray:
    """Grid init in [-1, 1]^2, identical to generate_fr_iterations.py --
    same init means the two teachers' trajectories start from the same
    state, which is required for them to be comparable steps of one operator."""
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


def run_fa2_with_snapshots(
    G: nx.Graph, n_iter: int = 50, k_r: float = 2.0, k_g: float = 1.0
) -> Dict:
    """Run FA2 for n_iter steps, saving a snapshot at every step. k is stored
    for schema compatibility with the FR pickle even though FA2's force law
    doesn't use it directly -- generate_final_dataset.py reads record["k"]."""
    nodes = sorted(G.nodes())
    n     = len(nodes)

    A       = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float64, weight=None)
    degrees = A.sum(axis=1)
    pos     = _grid_initial_positions(G)

    k  = np.sqrt(1.0 / n)  # schema compatibility only; FA2 doesn't use k
    coord_span = pos.max(axis=0) - pos.min(axis=0)
    t0 = float(np.max(coord_span) * 0.1)
    t0 = max(t0, 1e-4)
    dt = t0 / (n_iter + 1)

    steps       = []
    temperature = t0

    for step in range(n_iter):
        pos_before = pos.copy()
        pos        = fa2_single_step(pos, A, degrees, temperature, k_r=k_r, k_g=k_g)

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
    parser.add_argument("--k_r", type=float, default=2.0, help="FA2 scalingRatio")
    parser.add_argument("--k_g", type=float, default=1.0, help="FA2 gravity")
    parser.add_argument("--max_graphs", type=int, default=None)
    args = parser.parse_args()

    all_graphs = load_graphs(args.graphs_file)
    if args.max_graphs:
        all_graphs = all_graphs[: args.max_graphs]

    print(f"\nTotal graphs: {len(all_graphs)}  |  Iterations: {args.n_iter}")
    print(f"Total training samples: {len(all_graphs) * args.n_iter}\n")

    records = []
    for idx, G in enumerate(tqdm(all_graphs, desc="Generating FA2 trajectories")):
        try:
            result             = run_fa2_with_snapshots(G, n_iter=args.n_iter, k_r=args.k_r, k_g=args.k_g)
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
