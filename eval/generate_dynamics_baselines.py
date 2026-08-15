"""
eval/generate_dynamics_baselines.py

Q27 (the experiment queue): simple dynamics baselines -- trajectory-
smoothness controls that need NO training. They check the trained models are
not exploiting the fact that FR trajectories are smooth (small per-step
moves) instead of actually learning attraction / repulsion / the temperature
schedule. All scored by the same `eval/score_predictions.py` path as the real
models, using the recorded FR trajectories from the canonical dataset.

Baselines (per graph, from its recorded trajectory y_traj):
  noop      -- X_hat_t = X_0 for all t: predicts the fixed initial layout.
               Upper bound on how bad "do nothing" is.
  linvel    -- linear velocity extrapolation from the first two recorded
               states: X_hat_t = X_0 + t*(X_1 - X_0). Exploits smoothness
               naively; beats noop but has no curvature.
  conststep -- last-displacement extrapolation: X_hat_t = X_0 + t*(X_50 - X_49)
               repeated. Exploits the small late-step magnitude.
A model that only matches these is not learning the algorithm.

Emits the same .npz format as eval/predictions.py (source_indices, n_nodes,
positions, trajectory, meta) so score_predictions.py scores it identically.

Usage:
    .venv/bin/python eval/generate_dynamics_baselines.py \
        --dataset_path data/processed/comm_5k_v2_with_encodings.pt
"""

import argparse
import json
import os

import numpy as np
import torch


def build_baseline_trajectory(data, baseline: str, T: int = 50):
    """Return predicted trajectory [T, N, 2] for one graph."""
    n = int(data.num_nodes)
    y_traj = data.y_traj.view(n, -1, 2)  # [N, 50, 2]
    traj = y_traj.transpose(0, 1).numpy()  # [50, N, 2]
    x0 = traj[0]  # [N, 2]

    if baseline == "noop":
        return np.broadcast_to(x0[None], (T, n, 2)).copy()

    if baseline == "linvel":
        v = traj[1] - traj[0]  # first-step velocity
        return x0[None] + np.arange(T)[:, None, None] * v[None]

    if baseline == "conststep":
        step = traj[-1] - traj[-2]  # last-step displacement (small near converged)
        return x0[None] + np.arange(T)[:, None, None] * step[None]

    raise ValueError(baseline)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default="data/processed/comm_5k_v2_with_encodings.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="eval/predictions")
    args = ap.parse_args()

    # same split as score_predictions uses: reuse the executor loader's split
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from glide.executor.data import load_split

    full = torch.load(args.dataset_path, weights_only=False)
    gidx_to_src = {int(g.graph_idx): i for i, g in enumerate(full)}
    test = load_split(args.dataset_path, seed=args.seed)["test"]

    for baseline in ("noop", "linvel", "conststep"):
        src_indices, n_nodes, traj_chunks = [], [], []
        for d in test:
            n = int(d.num_nodes)
            tr = build_baseline_trajectory(d, baseline, T=50)
            src_indices.append(gidx_to_src[int(d.graph_idx)])
            n_nodes.append(n)
            traj_chunks.append(tr)
        T = traj_chunks[0].shape[0]
        positions = np.concatenate([t[-1] for t in traj_chunks]).astype(np.float32)
        trajectory = np.concatenate(traj_chunks, axis=1).astype(np.float32)
        meta = {"name": f"dynamics_{baseline}", "checkpoint": None,
                "family": "dynamics_baseline", "scale_free": False, "T": T,
                "frame": "normalized FR frame (shares y_mean/y_std with the target)"}

        os.makedirs(args.out_dir, exist_ok=True)
        dest = os.path.join(args.out_dir, f"dynamics_{baseline}.npz")
        np.savez_compressed(
            dest,
            source_indices=np.asarray(src_indices, dtype=np.int64),
            n_nodes=np.asarray(n_nodes, dtype=np.int64),
            positions=positions,
            trajectory=trajectory,
            meta=json.dumps(meta),
        )
        print(f"wrote {dest}  ({len(src_indices)} graphs, T={T})")


if __name__ == "__main__":
    main()
