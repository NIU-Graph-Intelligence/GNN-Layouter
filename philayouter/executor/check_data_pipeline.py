"""
philayouter/executor/check_data_pipeline.py

Verifies philayouter/executor/data.py's raw-frame reconstruction and temperature
schedule against the recorded trajectories in comm_5k_v2_with_encodings.pt,
before trusting either as a training target.

FR's step is displacement * (temperature / ||displacement||), so
||pos_after - pos_before|| == temperature exactly for any node whose raw
displacement length is above fr_single_step's eps=0.01 clip. Per-step median
node displacement should therefore match temperature_schedule()'s output
almost exactly -- this is what's checked here, across many graphs and steps,
not asserted from the formula alone.

Run: .venv/bin/python -m philayouter.executor.check_data_pipeline
"""

import argparse

import numpy as np
import torch

from .data import graph_to_raw_trajectory, load_split, temperature_schedule

TOLERANCE = 1e-3  # relative error


def check(dataset_path: str, n_graphs: int = 200):
    split = load_split(dataset_path)
    train = split["train"]
    print(f"train/val/test sizes: {len(train)}/{len(split['val'])}/{len(split['test'])}")

    max_rel_err = 0.0
    k_mismatches = 0
    for data in train[:n_graphs]:
        n = int(data.num_nodes)
        pos0_raw, traj_raw, k = graph_to_raw_trajectory(data)

        expected_k = float(np.sqrt(1.0 / n))
        if abs(k - expected_k) / expected_k > 1e-4:
            k_mismatches += 1

        n_iter = traj_raw.shape[0]
        temps = temperature_schedule(pos0_raw, n_iter)

        full = np.concatenate([pos0_raw[None], traj_raw], axis=0)  # [T+1, N, 2]
        for t in range(n_iter):
            step_mag = np.linalg.norm(full[t + 1] - full[t], axis=-1)
            med = float(np.median(step_mag))
            rel_err = abs(med - temps[t]) / temps[t]
            max_rel_err = max(max_rel_err, rel_err)

    print(f"graphs checked: {min(n_graphs, len(train))}")
    print(f"k == sqrt(1/N) mismatches: {k_mismatches}")
    print(f"max relative error, median step magnitude vs reconstructed temperature: {max_rel_err:.2e}")

    ok = k_mismatches == 0 and max_rel_err < TOLERANCE
    print(f"-> {'PASS' if ok else 'FAIL'} (tolerance {TOLERANCE:.0e})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="data/processed/comm_5k_v2_with_encodings.pt")
    parser.add_argument("--n_graphs", type=int, default=200)
    args = parser.parse_args()
    check(args.dataset_path, args.n_graphs)
