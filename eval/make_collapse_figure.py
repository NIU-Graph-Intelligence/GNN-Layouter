"""make_collapse_figure.py - Fig B: the coincident-node collapse, rendered.

Same graph (deterministic pick, a multi-community test graph where GND-GAT
collapses nodes): teacher FR final layout | GND-GAT (topology-only input,
collapsed) | Phi-layouter (geometric-state input, separated). Every panel is
Procrustes-aligned to the teacher, exactly as the fidelity metric credits, so
the panels are directly comparable. Coincident pairs (positions within 1e-6 in
the shared frame) are ringed in red in the GND panel and counted in the title.

Usage:
    cd /home/lei/work/GNN-Layouter
    .venv/bin/python eval/make_collapse_figure.py --out eval/results/fig_collapse_match.png
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from eval.metrics import procrustes_align

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
})

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
COLLAPSE_EPS = 1e-6


def load_npz_positions(name):
    z = np.load(os.path.join(ROOT, "eval", "predictions", f"{name}.npz"))
    nn = z["n_nodes"]
    return {int(s): p for s, p in zip(z["source_indices"],
                                      np.split(z["positions"], np.cumsum(nn)[:-1]))}


def coincident_pairs(P):
    from scipy.spatial import cKDTree
    pairs = set(cKDTree(P).query_pairs(COLLAPSE_EPS))
    out = {}
    for (a, b) in pairs:
        out.setdefault((P[a, 0], P[a, 1]), []).append((a, b))
    return out


def draw(ax, P, edges, comm, title, ring=None, n_ring=0):
    seg = np.stack([P[edges[:, 0]], P[edges[:, 1]]], axis=1)
    ax.add_collection(LineCollection(seg, colors="#999999", linewidths=0.4,
                                     alpha=0.75, zorder=1))
    colors = ([PALETTE[c % len(PALETTE)] for c in comm]
              if comm is not None else PALETTE[0])
    ax.scatter(P[:, 0], P[:, 1], s=11, c=colors, zorder=2,
               edgecolors="white", linewidths=0.3)
    if ring:
        for (cx, cy), _ in ring.items():
            ax.add_patch(plt.Circle((cx, cy), 0.13, fill=False, ec="#C0392B",
                                    lw=1.1, zorder=3))
    # Bounding box, not mean +/- fixed-fraction-of-span: the mean-centred
    # window silently clips outlier nodes whenever a layout is skewed
    # relative to its own mean, which is exactly what a collapse does to
    # this panel's geometry. Seen in the wild: the bottom-right community's
    # last node fell outside the old window and was never drawn.
    xmin, xmax = P[:, 0].min(), P[:, 0].max()
    ymin, ymax = P[:, 1].min(), P[:, 1].max()
    span = max(xmax - xmin, ymax - ymin) or 1.0
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = span / 2 * 1.08
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ttl = title if n_ring == 0 else f"{title} ({n_ring} pairs)"
    ax.set_title(ttl, fontsize=6.4, pad=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=int, default=2745, help="dataset index of the test graph")
    ap.add_argument("--dataset_path", default="data/processed/comm_5k_v2_with_encodings.pt")
    ap.add_argument("--out", default="eval/results/fig_collapse_match.png")
    args = ap.parse_args()

    dataset = torch.load(args.dataset_path, weights_only=False)
    if not isinstance(dataset, list):
        dataset = dataset.get("dataset", dataset)
    d = dataset[args.graph]
    n = int(d.num_nodes)
    ei = d.edge_index.numpy()
    edges = ei[:, ei[0] < ei[1]].T
    comm = d.community.numpy() if getattr(d, "community", None) is not None else None
    target = d.y.numpy().astype(np.float64)

    gnd = load_npz_positions("gnd_fr_gat")
    phi = load_npz_positions("phi1_v1")
    if args.graph not in gnd or args.graph not in phi:
        sys.exit(f"graph {args.graph} not in both prediction sets")
    Pgnd, _ = procrustes_align(gnd[args.graph].astype(np.float64), target)
    Pphi, _ = procrustes_align(phi[args.graph].astype(np.float64), target)

    ring = coincident_pairs(Pgnd)
    n_ring = sum(len(v) for v in ring.values())

    # Single-column width (three panels at ~1.1in each) instead of a
    # textwidth-spanning figure: this renders one 39-node graph three times
    # at different conditioning, which is a qualitative illustration of a
    # result Table I already states as a population statistic (500 graphs) --
    # it doesn't need double-column width to make that point. Titles
    # shortened accordingly; "step 50" / "topology-only input" / "geometric
    # state input" now live in the caption instead of every panel's title.
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.35))
    draw(axes[0], target, edges, comm, "teacher FR")
    draw(axes[1], Pgnd, edges, comm, "GND-GAT", ring=ring, n_ring=n_ring)
    draw(axes[2], Pphi, edges, comm, "$\\Phi$-layouter")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.08)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor="white")
    if os.path.splitext(args.out)[1].lower() == ".png":
        fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}  (graph {args.graph}, N={n}, {n_ring} coincident pairs in GND)")


if __name__ == "__main__":
    main()
