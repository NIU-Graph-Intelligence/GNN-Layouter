"""make_rollout_figure.py - Fig A for the manuscript: teacher vs Phi-layouter
at matched steps.

The fidelity claim of Study 2 is a Procrustes-aligned number; this figure shows
what that number looks like. One representative test graph - deterministically
the graph whose step-50 Procrustes error is closest to the split median - is
rolled out for T=50. Columns are matched step indices (0 is the shared initial
state, supplied to the model as input); rows are the FR teacher trajectory and
the Phi_1 rollout. Model panels are Procrustes-aligned to the teacher at the
same step, exactly as eval.metrics credits the fidelity metric.

Usage:
    cd /home/lei/work/GNN-Layouter
    .venv/bin/python eval/make_rollout_figure.py \\
        --checkpoint checkpoints/phi1_v1/executor_best.pt \\
        --dataset_path data/processed/comm_5k_v2_with_encodings.pt \\
        --out eval/results/fig_rollout_match.png
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from eval.metrics import rollout_error, procrustes_align
from philayouter.executor.evaluate import load_checkpoint, rollout_model
from philayouter.executor.data import graph_to_raw_trajectory, temperature_schedule, load_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
})

STEPS = [0, 1, 10, 25, 50]
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def draw(ax, P, edges, comm, title=None):
    seg = np.stack([P[edges[:, 0]], P[edges[:, 1]]], axis=1)
    ax.add_collection(LineCollection(seg, colors="#999999", linewidths=0.55,
                                     alpha=0.75, zorder=1))
    colors = ([PALETTE[c % len(PALETTE)] for c in comm]
              if comm is not None else PALETTE[0])
    ax.scatter(P[:, 0], P[:, 1], s=24, c=colors, zorder=2,
               edgecolors="white", linewidths=0.45)
    span = max(np.ptp(P[:, 0]), np.ptp(P[:, 1])) or 1.0
    cx, cy = P[:, 0].mean(), P[:, 1].mean()
    ax.set_xlim(cx - span * 0.56, cx + span * 0.56)
    ax.set_ylim(cy - span * 0.56, cy + span * 0.56)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9.2, pad=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_path", default="data/processed/comm_5k_v2_with_encodings.pt")
    ap.add_argument("--out", default="eval/results/fig_rollout_match.png")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    enc, model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"loaded {args.checkpoint} (epoch {ckpt['epoch']})")

    test = load_split(args.dataset_path, seed=args.seed)["test"]
    T = 50
    err = np.empty(len(test))
    all_pred, all_traj, all_pos0 = {}, {}, {}
    for i, data in enumerate(test):
        pos0_raw, traj_raw, k = graph_to_raw_trajectory(data)
        temps = temperature_schedule(pos0_raw, T)
        pred = rollout_model(enc, model, data, device, T, temps)
        err[i] = rollout_error(pred, traj_raw, aligned=True)[T - 1]
        all_pred[i] = pred
        all_traj[i] = traj_raw
        all_pos0[i] = pos0_raw

    med = np.median(err)
    pick = int(np.argmin(np.abs(err - med)))
    print(f"median step-50 error {med:.4f}; picked graph {pick} "
          f"({err[pick]:.4f}, {int(test[pick].num_nodes)} nodes, "
          f"{int(test[pick].community.max()) + 1} communities)")

    d = test[pick]
    ei = d.edge_index.numpy()
    edges = ei[:, ei[0] < ei[1]].T
    comm = d.community.numpy()
    traj = all_traj[pick]
    pred = all_pred[pick]
    pos0 = all_pos0[pick]

    full_t = np.concatenate([pos0[None], traj], axis=0)
    full_p = np.concatenate([pos0[None], pred], axis=0)

    ncol = len(STEPS)
    fig, axes = plt.subplots(2, ncol, figsize=(2.02 * ncol, 3.55), squeeze=False)
    for c, s in enumerate(STEPS):
        tP = full_t[s]
        pP, _ = procrustes_align(full_p[s], tP)
        title = "initial state" if s == 0 else f"step {s}"
        draw(axes[0][c], tP, edges, comm, title=title)
        draw(axes[1][c], pP, edges, comm)
        if s == 50:
            axes[1][c].text(0.5, -0.06, f"Procrustes {err[pick]:.3f}",
                            transform=axes[1][c].transAxes, ha="center", va="top",
                            fontsize=7.5, color="#555555", style="italic")
    axes[0][0].set_ylabel("teacher FR", fontsize=10, rotation=0, ha="right",
                          va="center", labelpad=10)
    axes[1][0].set_ylabel("$\\Phi$-layouter", fontsize=10, rotation=0, ha="right",
                          va="center", labelpad=10)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.92, bottom=0.055,
                        wspace=0.05, hspace=0.08)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    if os.path.splitext(args.out)[1].lower() == ".png":
        fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}  (graph {pick})")
    with open(args.out + ".graph.txt", "w") as f:
        f.write(f"{pick} {err[pick]:.6f} {int(test[pick].num_nodes)} "
                f"{int(test[pick].community.max()) + 1}\n")


if __name__ == "__main__":
    main()
