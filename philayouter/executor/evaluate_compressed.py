"""
philayouter/executor/evaluate_compressed.py

Q11 (the experiment queue): quality-vs-k curve for the compressed
executor Phi_k. One forward pass = k algorithm iterations (stride as a
conditioning input, dial-able at inference per PAPER_PLAN.md §5), so the
model reaches FR step 50 in ceil(50/k) forwards.

For each candidate stride k, roll the trained Phi_k model out for
ceil(T/k) forwards and measure Procrustes rollout_error at the T=50 mark
against the recorded FR trajectory -- the same metric/convention
evaluate.py uses, so k=1 is directly comparable to Q4/Q5's Phi_1 numbers.
Also reports the sequential-depth win: forwards_needed = ceil(T/k).

The honest empirical answer -- including k where quality collapses -- is the
deliverable; do not cherry-pick k values after the fact.

Usage:
    .venv/bin/python -m philayouter.executor.evaluate_compressed \
        --checkpoint checkpoints/phi_k_v1/executor_best.pt \
        --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
        --k_list 1 2 4 8 16
"""

import argparse

import numpy as np
import torch

from eval.metrics import rollout_error
from .data import graph_to_raw_trajectory, load_split, temperature_schedule
from .evaluate import load_checkpoint, rollout_model


def quality_vs_k(encoder, model, test_graphs, device, k_list, T=50):
    """Per-k step-50 Procrustes fidelity + forwards needed. Returns dict
    k -> (mean_aligned_step50, forwards)."""
    results = {}
    for k in k_list:
        n_forwards = int(np.ceil(T / k))
        reached = min(n_forwards * k, T)  # algorithm step the last forward lands on
        errs = np.zeros(len(test_graphs))
        n = 0
        for i, data in enumerate(test_graphs):
            pos0_raw, traj_raw, kk = graph_to_raw_trajectory(data)
            temps = temperature_schedule(pos0_raw, T)
            pred = rollout_model(encoder, model, data, device, n_forwards, temps, stride=k)
            # pred has n_forwards rows; its last row is the state after
            # `reached` algorithm iterations. Compare against the recorded
            # trajectory at that same step index (matched step index, same
            # convention as evaluate.py's step-50 number).
            true_at_reached = traj_raw[reached - 1 : reached]  # [1, N, 2]
            errs[i] = float(rollout_error(pred[-1:], true_at_reached, aligned=True)[0])
            n += 1
        results[k] = (float(errs.mean()), n_forwards)
    return results


def main():
    parser = argparse.ArgumentParser(description="Q11: quality-vs-k curve for Phi_k")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", default="data/processed/comm_5k_v2_with_encodings.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_list", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--T", type=int, default=50)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    encoder, model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"loaded checkpoint from epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.6f}")
    print(f"checkpoint strides: {ckpt.get('strides')}")

    test_graphs = load_split(args.dataset_path, seed=args.seed)["test"]
    print(f"test graphs: {len(test_graphs)}, T={args.T}\n")

    res = quality_vs_k(encoder, model, test_graphs, device, args.k_list, T=args.T)

    print(f"{'k':>3} | {'forwards (T/k)':>14} | {'step-50 Procrustes':>18} | compression win")
    print("-" * 70)
    base = res[1][0]  # k=1 is the reference
    for k in args.k_list:
        err, n_fwd = res[k]
        win = f"{base/err:.2f}x quality at {n_fwd}/{args.T} depth" if k > 1 else "reference"
        print(f"{k:>3} | {n_fwd:>14} | {err:>18.6f} | {win}")


if __name__ == "__main__":
    main()
