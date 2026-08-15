"""
philayouter/executor/train_compressed.py

Q11 (the experiment queue): train Phi_k, the compressed executor. One
forward pass = k algorithm iterations, with k supplied as a conditioning
input so it is dial-able at inference (PAPER_PLAN.md §5).

Design (recorded in EXPERIMENT_QUEUE.md Q11, 2026-08-13):
- ONE model, `use_stride=True`, trained on mixed strides. Each graph is
  assigned a stride sampled (per epoch) from the candidate set -- stride=1
  anchors the model to exact Phi_1 behaviour while larger strides teach it
  compression. The stride value is the conditioning input, so at inference
  any k in/near the trained range is a valid setting; no retraining per k.
- Targets come from strided views of the already-recorded trajectories
  (philayouter/executor/data.py::iter_steps(stride=...)) -- no new data collection.
- Otherwise identical to train.py: one graph = one training example (all its
  steps summed into one loss/update), no cross-graph batching, same
  k-unit normalization and temperature reconstruction.

Usage:
    .venv/bin/python -m philayouter.executor.train_compressed \
        --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
        --output_dir checkpoints/phi_k_v1 --strides 1 2 4 8 \
        --num_epochs 50 --device cuda:1
"""

import argparse
import csv
import os
import random
import time

import torch
import torch.nn.functional as F

from .data import iter_steps, load_split
from .model import EquivariantExecutor
from .structural import StructuralEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Q11: train the compressed executor (Phi_k)")
    parser.add_argument("--dataset_path", default="data/processed/comm_5k_v2_with_encodings.pt")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k_geo", type=int, default=10)
    parser.add_argument("--lap_k", type=int, default=10)
    parser.add_argument("--rw_k", type=int, default=16)

    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="candidate compression strides to sample from per graph")

    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_graphs", type=int, default=None, help="debug: cap train set size")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def graph_loss(encoder, model, data, stride, device):
    """Sum of per-step MSE(dX_pred, dX_target) over every recorded FR
    iteration of THIS graph at the given stride, in the k-normalized frame."""
    edge_index = data.edge_index.to(device)
    node_feat = encoder.encode_precomputed(
        data.lap_pe.to(device), data.rrwp_node.to(device)
    )

    total = 0.0
    n_steps = 0
    stride_t = torch.tensor(float(stride), device=device)
    for step in iter_steps(data, stride=stride):
        pos_before = step.pos_before.to(device)
        pos_after = step.pos_after.to(device)
        tau = step.tau.to(device)

        dX_pred = model(node_feat, edge_index, pos_before, tau, stride=stride_t)
        dX_target = pos_after - pos_before
        total = total + F.mse_loss(dX_pred, dX_target, reduction="mean")
        n_steps += 1

    return total / max(n_steps, 1)


def run_epoch(encoder, model, graphs, strides, device, optimizer=None, grad_clip=1.0):
    train = optimizer is not None
    encoder.train(train)
    model.train(train)

    total_loss = 0.0
    for data in graphs:
        # sample a stride per graph per epoch: every epoch a graph is seen at
        # a (possibly different) compression, so no stride is starved.
        stride = random.choice(strides)
        loss = graph_loss(encoder, model, data, stride, device)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(model.parameters()), grad_clip
            )
            optimizer.step()
        total_loss += loss.item()

    return total_loss / len(graphs)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    split = load_split(args.dataset_path, seed=args.seed)
    train_graphs, val_graphs = split["train"], split["val"]
    if args.max_graphs is not None:
        train_graphs = train_graphs[: args.max_graphs]
        val_graphs = val_graphs[: max(1, args.max_graphs // 8)]
    print(f"train/val graphs: {len(train_graphs)}/{len(val_graphs)}")
    print(f"strides: {args.strides}")

    encoder = StructuralEncoder(out_dim=args.hidden_dim, lap_k=args.lap_k, rw_k=args.rw_k).to(device)
    model = EquivariantExecutor(
        node_feat_dim=args.hidden_dim, hidden_dim=args.hidden_dim, k_geo=args.k_geo,
        num_teachers=1, use_stride=True,
    ).to(device)

    n_params = sum(p.numel() for p in list(encoder.parameters()) + list(model.parameters()) if p.requires_grad)
    print(f"parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "executor_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "seconds"])

    best_val = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(encoder, model, train_graphs, args.strides, device, optimizer, args.grad_clip)
        with torch.no_grad():
            val_loss = run_epoch(encoder, model, val_graphs, args.strides, device, optimizer=None)
        dt = time.time() - t0

        if epoch % args.log_every == 0 or epoch == args.num_epochs:
            print(f"epoch {epoch:4d}  train_loss {train_loss:.6f}  val_loss {val_loss:.6f}  ({dt:.1f}s)")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, dt])

        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "encoder_state_dict": encoder.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "strides": args.strides,
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, "executor_best.pt"))

    print(f"done. best val_loss {best_val:.6f}")


if __name__ == "__main__":
    main()
