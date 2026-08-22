"""
philayouter/executor/train_mpnn.py

Train the MPNN-max Neural Executor baseline.

IDENTICAL to train.py (same data, same seed-42 split, same per-graph training
shape, same k-unit normalization, same temperature reconstruction, same loss
and optimizer) EXCEPT the model is MpnnExecutor -- a plain MPNN with max
aggregation, no equivariant readout, no geometric rewiring. That is the whole
point: it isolates what algorithmic alignment buys, holding everything else
constant.

Usage:
    .venv/bin/python -m philayouter.executor.train_mpnn \
        --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
        --output_dir checkpoints/mpnn_v1 --num_epochs 50 --device cuda:0
"""

import argparse
import csv
import os
import random
import time

import torch
import torch.nn.functional as F

from .data import iter_steps, load_split
from .mpnn import MpnnExecutor
from .structural import StructuralEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="train the MPNN-max neural executor baseline")
    parser.add_argument("--dataset_path", default="data/processed/comm_5k_v2_with_encodings.pt")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lap_k", type=int, default=10)
    parser.add_argument("--rw_k", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=3)

    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_graphs", type=int, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def graph_loss(encoder, model, data, device):
    edge_index = data.edge_index.to(device)
    node_feat = encoder.encode_precomputed(data.lap_pe.to(device), data.rrwp_node.to(device))

    total = 0.0
    n_steps = 0
    for step in iter_steps(data):
        pos_before = step.pos_before.to(device)
        pos_after = step.pos_after.to(device)
        tau = step.tau.to(device)

        dX_pred = model(node_feat, edge_index, pos_before, tau)
        dX_target = pos_after - pos_before
        total = total + F.mse_loss(dX_pred, dX_target, reduction="mean")
        n_steps += 1

    return total / n_steps


def run_epoch(encoder, model, graphs, device, optimizer=None, grad_clip=1.0):
    train = optimizer is not None
    encoder.train(train)
    model.train(train)

    total_loss = 0.0
    for data in graphs:
        loss = graph_loss(encoder, model, data, device)
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

    split = load_split(args.dataset_path, seed=args.split_seed)
    train_graphs, val_graphs = split["train"], split["val"]
    if args.max_graphs is not None:
        train_graphs = train_graphs[: args.max_graphs]
        val_graphs = val_graphs[: max(1, args.max_graphs // 8)]
    print(f"train/val graphs: {len(train_graphs)}/{len(val_graphs)}")

    encoder = StructuralEncoder(out_dim=args.hidden_dim, lap_k=args.lap_k, rw_k=args.rw_k).to(device)
    model = MpnnExecutor(node_feat_dim=args.hidden_dim, hidden_dim=args.hidden_dim,
                         num_layers=args.num_layers).to(device)

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
        train_loss = run_epoch(encoder, model, train_graphs, device, optimizer, args.grad_clip)
        with torch.no_grad():
            val_loss = run_epoch(encoder, model, val_graphs, device, optimizer=None)
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
            "model_type": "MpnnExecutor",
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, "executor_best.pt"))

    print(f"done. best val_loss {best_val:.6f}")


if __name__ == "__main__":
    main()
