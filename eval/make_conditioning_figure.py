"""make_conditioning_figure.py - Fig C: one model, three drawing conventions.

Same test graph (deterministic pick: median-N multi-community test graph) under
the three teachers. Laid out as three rows (FR / ForceAtlas2 / Kamada-Kawai),
each with the recorded teacher layout on the left and the single multi-teacher
model's conditioned rollout on the right; each right panel is Procrustes-aligned
to its conditioning teacher's layout (the metric's convention), so the two
panels in a row are the on-diagonal match the fidelity numbers describe.
Single-column width: two panels per row read better at columnwidth than three
panels per row do at textwidth once you're down to a 36-node graph.

Usage:
    cd /home/lei/work/GNN-Layouter
    .venv/bin/python eval/make_conditioning_figure.py --out eval/results/fig_conditioning_match.png
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

from eval.metrics import procrustes_align, denormalize

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
TEACHERS = ["FR", "ForceAtlas2", "Kamada–Kawai"]
DATASETS = {
    "FR": "data/processed/comm_5k_v2_with_encodings.pt",
    "FA2": "data/processed/comm_5k_fa2_with_encodings.pt",
    "KK": "data/processed/comm_5k_kk_with_encodings.pt",
}
DUMP = {"FR": "phi1_mt_fr", "FA2": "phi1_mt_fa2", "KK": "phi1_mt_kk"}


def draw(ax, P, edges, comm, title=None, fontsize=9.2):
    seg = np.stack([P[edges[:, 0]], P[edges[:, 1]]], axis=1)
    ax.add_collection(LineCollection(seg, colors="#999999", linewidths=0.55,
                                     alpha=0.75, zorder=1))
    colors = ([PALETTE[c % len(PALETTE)] for c in comm]
              if comm is not None else PALETTE[0])
    ax.scatter(P[:, 0], P[:, 1], s=24, c=colors, zorder=2,
               edgecolors="white", linewidths=0.45)
    # Bounding box, not mean +/- fixed-fraction-of-span: the previous version
    # centred the window on the point *mean* and sized it off the larger of
    # the two axis spans, which silently clips outlier nodes whenever the
    # layout is skewed relative to its own mean (this is exactly what
    # happened on the collapsed-layout panel in fig_collapse_match -- a
    # node fell outside the window and was never drawn). Centring on the
    # actual bounding box, not the mean, guarantees every point is inside.
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
    if title:
        ax.set_title(title, fontsize=fontsize, pad=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=int, default=None)
    ap.add_argument("--out", default="eval/results/fig_conditioning_match.png")
    args = ap.parse_args()

    dsets = {}
    for tag, p in DATASETS.items():
        ds = torch.load(p, weights_only=False)
        dsets[tag] = ds if isinstance(ds, list) else ds.get("dataset", ds)

    # pick a deterministic mid-size multi-community TEST graph (same index in
    # all three datasets -- test splits are identical across teachers)
    def test_n(x):
        return int(x.community.max()) + 1 if getattr(x, "community", None) is not None else 1
    import random as _random
    _idx = list(range(len(dsets["FR"])))
    _random.seed(42)
    _random.shuffle(_idx)
    test_idx = _idx[4500:]
    if args.graph is None:
        cand = [(gi, int(dsets["FR"][gi].num_nodes)) for gi in test_idx
                if test_n(dsets["FR"][gi]) >= 4]
        mid = np.median([n for _, n in cand])
        args.graph = min((gi for gi, n in cand if n >= mid),
                         key=lambda gi: abs(int(dsets["FR"][gi].num_nodes) - mid))
    elif args.graph not in test_idx:
        sys.exit(f"graph {args.graph} is not in the test split")

    gi = args.graph
    d = dsets["FR"][gi]
    n = int(d.num_nodes)
    ei = d.edge_index.numpy()
    edges = ei[:, ei[0] < ei[1]].T
    comm = d.community.numpy()

    def raw_layout(tag):
        dd = dsets[tag][gi]
        return denormalize(dd.y.numpy().astype(np.float64),
                           dd.y_mean.numpy(), dd.y_std.numpy())

    def raw_pred(tag):
        z = np.load(os.path.join(ROOT, "eval", "predictions", f"{DUMP[tag]}.npz"))
        nn = z["n_nodes"]
        pos = {int(s): p for s, p in zip(z["source_indices"], np.split(z["positions"], np.cumsum(nn)[:-1]))}
        dd = dsets[tag][gi]
        return denormalize(pos[gi].astype(np.float64), dd.y_mean.numpy(), dd.y_std.numpy())

    # Three rows (one per teacher convention) x two columns (teacher |
    # executor) instead of two rows x three columns: fits single-column
    # width, and each row keeps the on-diagonal pair (teacher, conditioned
    # executor) adjacent, which is the comparison the caption asks the
    # reader to make. "compressed" (not "constrained") layout: both
    # reserve space per-axes including titles, so neither lets a panel's
    # elongated data extent bleed into a neighbouring row's title (the
    # fixed-hspace layout this replaces is what caused that), but
    # "compressed" is the engine matplotlib recommends specifically for a
    # grid of equal-aspect axes -- it collapses the extra row/column
    # padding "constrained" leaves room for in case aspect ratios varied,
    # which they don't here.
    LABEL_FS = 7.4  # shared by row labels and column headers -- one size, one weight
    ABBREV = ["FR", "FA2", "KK"]  # matches Tables III/IV; the in-panel attempt used
    # the long form (TEACHERS) and had nowhere to put it without risking overlap.
    # Back to a left-margin label -- the overlap in the in-panel version wasn't a
    # one-off, the bounding box in draw() is sized to the data's own extent, so a
    # node sits near a given corner about as often as not. Short abbreviations
    # keep the margin this costs close to what three characters need.
    fig, axes = plt.subplots(3, 2, figsize=(3.45, 4.35), layout="compressed")
    for r, tag in enumerate(["FR", "FA2", "KK"]):
        tgt = raw_layout(tag)
        pred = raw_pred(tag)
        P, _ = procrustes_align(pred, tgt)
        top_titles = ("teacher layout", "$\\Phi$-layouter\n(conditioned)") if r == 0 else (None, None)
        draw(axes[r][0], tgt, edges, comm, top_titles[0], fontsize=LABEL_FS)
        draw(axes[r][1], P, edges, comm, top_titles[1], fontsize=LABEL_FS)
        axes[r][0].set_ylabel(ABBREV[r], fontsize=LABEL_FS, rotation=0,
                              ha="right", va="center", labelpad=4)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white")
    if os.path.splitext(args.out)[1].lower() == ".png":
        fig.savefig(os.path.splitext(args.out)[0] + ".pdf", facecolor="white")
    print(f"wrote {args.out}  (graph {gi}, N={n}, {int(comm.max())+1} communities)")


if __name__ == "__main__":
    main()
