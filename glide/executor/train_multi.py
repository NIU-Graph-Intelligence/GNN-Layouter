"""
glide/executor/train_multi.py

Q7 (the experiment queue): train Phi_1 with a 3-teacher conditioning
vector -- FR, FA2, Kamada-Kawai as values of the `teacher_id` embedding
`EquivariantExecutor` has carried unused since Q2.

The three datasets (comm_5k_v2_with_encodings.pt, comm_5k_fa2_with_encodings.pt,
comm_5k_kk_with_encodings.pt) are the same 5000 graphs in the same order, so
calling glide.executor.data.load_split(path, seed=42) independently on each
produces bit-identical train/val/test index partitions (same seed, same N ->
same random.shuffle output) without needing a dedicated multi-dataset split
routine. Each epoch trains on the union of all three teachers' train graphs,
shuffled together, each tagged with its teacher_id.

Otherwise identical to train.py: one graph = one training example (all its
recorded steps summed into one loss/update), no cross-graph batching (same
kNN-contamination reason as train.py).

Usage:
    .venv/bin/python -m glide.executor.train_multi \
        --output_dir checkpoints/phi1_multiteacher_v1 --num_epochs 50 --device cuda:0
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

TEACHERS = {
    0: ("fr", "data/processed/comm_5k_v2_with_encodings.pt"),
    1: ("fa2", "data/processed/comm_5k_fa2_with_encodings.pt"),
    2: ("kk", "data/processed/comm_5k_kk_with_encodings.pt"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Q7: train Phi_1 with 3-teacher conditioning")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k_geo", type=int, default=10)
    parser.add_argument("--lap_k", type=int, default=10)
    parser.add_argument("--rw_k", type=int, default=16)

    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_graphs", type=int, default=None, help="debug: cap train set size PER teacher")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def load_all_teachers(seed: int, max_graphs=None):
    """Returns {'train': [(data, teacher_id), ...], 'val': [...], 'test': [...]}."""
    splits = {"train": [], "val": [], "test": []}
    for teacher_id, (name, path) in TEACHERS.items():
        split = load_split(path, seed=seed)
        for part in ("train", "val", "test"):
            graphs = split[part]
            if max_graphs is not None:
                cap = max_graphs if part == "train" else max(1, max_graphs // 8)
                graphs = graphs[:cap]
            splits[part].extend((g, teacher_id) for g in graphs)
        print(f"  {name}: train {len(split['train'])}, val {len(split['val'])}, test {len(split['test'])}")
    return splits


def graph_loss(encoder, model, data, teacher_id, device):
    edge_index = data.edge_index.to(device)
    node_feat = encoder.encode_precomputed(data.lap_pe.to(device), data.rrwp_node.to(device))

    total = 0.0
    n_steps = 0
    for step in iter_steps(data, teacher_id=teacher_id):
        pos_before = step.pos_before.to(device)
        pos_after = step.pos_after.to(device)
        tau = step.tau.to(device)
        teacher_id_t = step.teacher_id.to(device)

        dX_pred = model(node_feat, edge_index, pos_before, tau, teacher_id=teacher_id_t)
        dX_target = pos_after - pos_before
        total = total + F.mse_loss(dX_pred, dX_target, reduction="mean")
        n_steps += 1

    return total / n_steps


def run_epoch(encoder, model, graph_teacher_pairs, device, optimizer=None, grad_clip=1.0):
    train = optimizer is not None
    encoder.train(train)
    model.train(train)

    total_loss = 0.0
    per_teacher_loss = {tid: [0.0, 0] for tid in TEACHERS}
    for data, teacher_id in graph_teacher_pairs:
        loss = graph_loss(encoder, model, data, teacher_id, device)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(model.parameters()), grad_clip
            )
            optimizer.step()

        total_loss += loss.item()
        per_teacher_loss[teacher_id][0] += loss.item()
        per_teacher_loss[teacher_id][1] += 1

    breakdown = {TEACHERS[tid][0]: (s / max(n, 1)) for tid, (s, n) in per_teacher_loss.items()}
    return total_loss / len(graph_teacher_pairs), breakdown


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading all 3 teachers (same seed-42 split, same graph order):")
    splits = load_all_teachers(args.seed, args.max_graphs)
    train_pairs, val_pairs = splits["train"], splits["val"]

    random.Random(args.seed).shuffle(train_pairs)  # interleave teachers, not blocked

    print(f"total train/val examples (all teachers): {len(train_pairs)}/{len(val_pairs)}")

    encoder = StructuralEncoder(out_dim=args.hidden_dim, lap_k=args.lap_k, rw_k=args.rw_k).to(device)
    model = EquivariantExecutor(
        node_feat_dim=args.hidden_dim, hidden_dim=args.hidden_dim, k_geo=args.k_geo,
        num_teachers=len(TEACHERS),
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
    teacher_names = [TEACHERS[i][0] for i in sorted(TEACHERS)]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "seconds"]
            + [f"train_{n}" for n in teacher_names]
            + [f"val_{n}" for n in teacher_names]
        )

    best_val = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        train_loss, train_breakdown = run_epoch(encoder, model, train_pairs, device, optimizer, args.grad_clip)
        with torch.no_grad():
            val_loss, val_breakdown = run_epoch(encoder, model, val_pairs, device, optimizer=None)
        dt = time.time() - t0

        if epoch % args.log_every == 0 or epoch == args.num_epochs:
            tb = " ".join(f"{k}={v:.4f}" for k, v in train_breakdown.items())
            vb = " ".join(f"{k}={v:.4f}" for k, v in val_breakdown.items())
            print(f"epoch {epoch:4d}  train {train_loss:.6f} ({tb})  val {val_loss:.6f} ({vb})  ({dt:.1f}s)")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, val_loss, dt]
                + [train_breakdown[n] for n in teacher_names]
                + [val_breakdown[n] for n in teacher_names]
            )

        checkpoint = {
            "epoch": epoch,
            "val_loss": val_loss,
            "encoder_state_dict": encoder.state_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "teachers": TEACHERS,
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, "executor_best.pt"))

    print(f"done. best val_loss {best_val:.6f}")


if __name__ == "__main__":
    main()
