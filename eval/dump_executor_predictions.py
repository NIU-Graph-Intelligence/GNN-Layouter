"""dump_executor_predictions.py - dump executor (and ablation) rollouts to
eval/predictions/<name>.npz in the dataset's normalized frame (shares
y_mean/y_std with d.y), so eval/score_predictions.py can score them through
the same metrics as every baseline. Optionally includes the full 50-step
trajectory, which is what the qualitative rollout figure needs.

Usage:
    .venv/bin/python eval/dump_executor_predictions.py \\
        --checkpoint checkpoints/phi1_v1/executor_best.pt \\
        --name phi1_v1
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from philayouter.executor.data import graph_to_raw_trajectory, temperature_schedule
from philayouter.executor.evaluate import load_checkpoint, rollout_model


def test_split_indices(dataset_path: str, seed: int = 42, ratios=(0.8, 0.1, 0.1)):
    """Replicate load_split's shuffle exactly and return (original indices of
    the requested split, dataset objects) so source_indices index the raw list."""
    dataset = torch.load(dataset_path, weights_only=False)
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    n = len(dataset)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    split = indices[n_train + n_val :]
    return split, [dataset[i] for i in split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", required=True, help="output stem -> eval/predictions/<name>.npz")
    ap.add_argument("--dataset_path", default="data/processed/comm_5k_v2_with_encodings.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--teacher_id", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    enc, model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"loaded {args.checkpoint} (epoch {ckpt['epoch']})")

    src_idx, graphs = test_split_indices(args.dataset_path, seed=args.seed)
    T = 50

    src = []
    nn = []
    pos_all = []
    traj_all = []
    for gi, data in enumerate(graphs):
        pos0_raw, _, k = graph_to_raw_trajectory(data)
        temps = temperature_schedule(pos0_raw, T)
        pred = rollout_model(enc, model, data, device, T, temps,
                             teacher_id=args.teacher_id, stride=args.stride)
        y_mean = data.y_mean.numpy()
        y_std = data.y_std.numpy()
        stored = (pred - y_mean) / y_std          # -> dataset normalized frame
        n = int(data.num_nodes)
        src.append(int(src_idx[gi]))
        nn.append(n)
        pos_all.append(stored[-1])
        traj_all.append(stored)

    meta = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "family": "phi-layouter",
        "scale_free": False,
        "split": args.split,
        "teacher_id": args.teacher_id,
        "stride": args.stride,
        "frame": "normalized FR frame (shares y_mean/y_std with the target)",
        "epoch": ckpt["epoch"],
        "val_loss": float(ckpt["val_loss"]),
    }
    positions = np.concatenate(pos_all, axis=0).astype(np.float32)
    trajectory = np.concatenate(traj_all, axis=1).astype(np.float32)  # [T, sum(n), 2]
    out = os.path.join(HERE, "predictions", f"{args.name}.npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        source_indices=np.array(src, dtype=np.int64),
        n_nodes=np.array(nn, dtype=np.int64),
        positions=positions,
        trajectory=trajectory,
        meta=np.array(json.dumps(meta)),
    )
    print(f"wrote {out}  ({len(src)} graphs, T={T}, {trajectory.shape[1]} rows)")


if __name__ == "__main__":
    main()
