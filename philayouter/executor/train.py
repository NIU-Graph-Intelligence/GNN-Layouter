"""
philayouter/executor/train.py

Q4 (the experiment queue): train Phi_1, the equivariant executor, on
the existing comm_5k_v2_with_encodings.pt trajectories -- no new data
collection, reusing the FR trajectories already validated to cosine 1.0
against FR's own update direction (eval/validate_metrics.py).

One graph = one training example: all recorded steps for that graph are
summed into a single loss and a single optimizer step, rather than a
per-step update, so gradient noise reflects a whole trajectory rather than
one step of one graph. Multiple graphs are NOT batched into one PyG Batch --
the geometric kNN in EquivariantExecutor operates over cdist(pos, pos), and
batching would connect nodes across unrelated graphs unless the kNN
construction is made batch-aware first. Cheaper to train per-graph for this
first run than to add that now; revisit if throughput becomes the bottleneck
(graphs here are N=20-50, two RTX 4090s -- unlikely to be the bottleneck at
this scale).

Usage:
    .venv/bin/python -m philayouter.executor.train \
        --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
        --output_dir checkpoints/phi1_v1 --num_epochs 50 --device cuda:0
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
    parser = argparse.ArgumentParser(description="Q4: train the equivariant executor (Phi_1)")
    parser.add_argument("--dataset_path", default="data/processed/comm_5k_v2_with_encodings.pt")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k_geo", type=int, default=10)
    parser.add_argument("--lap_k", type=int, default=10)
    parser.add_argument("--rw_k", type=int, default=16)
    parser.add_argument("--use_rrwp", type=lambda s: s.lower() not in ("0", "false", "no"),
                        default=True, help="include RRWP in structural encodings; "
                        "set 0/false for the LapPE-only ablation (Q14 structural-encoding decision)")
    parser.add_argument("--use_tau", type=lambda s: s.lower() not in ("0", "false", "no"),
                        default=True, help="condition the readout on temperature; "
                        "set 0/false for the no-temperature ablation (Q15)")
    parser.add_argument("--use_geo_mp", type=lambda s: s.lower() not in ("0", "false", "no"),
                        default=True, help="include the geometric kNN message passing; "
                        "set 0/false for the no-geometric-rewiring ablation (Q16)")
    parser.add_argument("--use_equiv_readout", type=lambda s: s.lower() not in ("0", "false", "no"),
                        default=True, help="equivariant readout (scalar * unit vector); "
                        "set 0/false for the non-equivariant-readout ablation (Q17)")

    parser.add_argument("--supervision", type=str, default="steps",
                        choices=["steps", "endpoint"],
                        help="'steps': per-step MSE on displacement (Phi_1). "
                             "'endpoint': the no-hint / endpoint-only control "
                             "(the executor's own ablation B) -- the same "
                             "teacher-forced step loop but loss only on the "
                             "final step's displacement, so intermediate states "
                             "are inputs but never targets, matching the "
                             "Stage-2 ablation-B condition.")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_graphs", type=int, default=None, help="debug: cap train set size")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=1)
    return parser.parse_args()


def graph_loss(encoder, model, data, device, supervision: str = "steps"):
    """Sum of per-step MSE(dX_pred, dX_target) over every recorded FR
    iteration for this graph, in the k-normalized frame.

    `supervision="endpoint"` is the no-hint control (the executor's own
    ablation B, PAPER_PLAN.md §8 row 1): keep the same teacher-forced step
    loop and the same per-step inputs, but supervise only the final step's
    displacement. Intermediate states are still provided as inputs (exactly
    as in the full model) but contribute no loss, so the model receives no
    intermediate target signal -- the Stage-2 ablation-B condition on the
    executor architecture. Expectation (measured for Stage-2): worse fidelity
    and a collapse of the 2-D layout. Any supervision value other than
    "endpoint" keeps the standard per-step teacher-forced Phi_1 loss."""
    edge_index = data.edge_index.to(device)
    node_feat = encoder.encode_precomputed(
        data.lap_pe.to(device), data.rrwp_node.to(device)
    )  # computed once, reused for every step below

    if supervision == "endpoint":
        steps = list(iter_steps(data))
        last = steps[-1]
        dX_pred = model(node_feat, edge_index, last.pos_before.to(device), last.tau.to(device))
        dX_target = last.pos_after.to(device) - last.pos_before.to(device)
        return F.mse_loss(dX_pred, dX_target, reduction="mean")

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


def run_epoch(encoder, model, graphs, device, optimizer=None, grad_clip=1.0, supervision="steps"):
    train = optimizer is not None
    encoder.train(train)
    model.train(train)

    total_loss = 0.0
    for data in graphs:
        loss = graph_loss(encoder, model, data, device, supervision=supervision)

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

    encoder = StructuralEncoder(
        out_dim=args.hidden_dim, lap_k=args.lap_k, rw_k=args.rw_k, use_rrwp=args.use_rrwp
    ).to(device)
    model = EquivariantExecutor(
        node_feat_dim=args.hidden_dim, hidden_dim=args.hidden_dim, k_geo=args.k_geo,
        num_teachers=1, use_tau=args.use_tau, use_geo_mp=args.use_geo_mp,
        use_equiv_readout=args.use_equiv_readout,
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
        train_loss = run_epoch(encoder, model, train_graphs, device, optimizer, args.grad_clip,
                               supervision=args.supervision)
        with torch.no_grad():
            val_loss = run_epoch(encoder, model, val_graphs, device, optimizer=None,
                                 supervision=args.supervision)
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
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, "executor_best.pt"))

    print(f"done. best val_loss {best_val:.6f}")


if __name__ == "__main__":
    main()
