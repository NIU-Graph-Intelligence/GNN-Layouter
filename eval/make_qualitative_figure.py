"""
make_qualitative_figure.py — side-by-side layouts: FR ground truth vs models.

Rows are graphs, columns are FR then each chosen model, nodes coloured by
community. Every prediction is Procrustes-aligned to FR (the same O(2)+scale
transform the fidelity metric uses) so the panels are directly comparable — a
mirrored layout is un-mirrored for display, exactly as the metric credits it.

USAGE
-----
Interactive (pick which models and which graphs from a menu):

    python eval/make_qualitative_figure.py

Choose models explicitly (by .npz stem), auto-pick graphs:

    python eval/make_qualitative_figure.py --models deepdrawing gnd_fr smartgd coregd dnn2 vo2

Choose specific graphs too:

    python eval/make_qualitative_figure.py --models vo2 smartgd --indices 76 4503 2988

FR ground truth is always the first column and is read from the dataset; every
other column is read from eval/predictions/<stem>.npz. Any stem present there
can be shown — nothing is hardcoded.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from metrics import procrustes_align

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

DEFAULT_DATASET = os.path.join(ROOT, "data/processed/comm_5k_v2_with_encodings.pt")
DEFAULT_PRED_DIR = os.path.join(HERE, "predictions")
DEFAULT_OUT = os.path.join(ROOT, "visualizations/qualitative.png")

# Pretty labels for known stems; unknown stems fall back to the stem itself.
LABELS = {
    "vo2": "GLIDE (ours)", "ablationB": "Ablation (endpoint-only)",
    "deepdrawing": "DeepDrawing", "gnd_fr": "GND (FR-sup.)",
    "gnd_stress": "GND (stress)", "smartgd": "SmartGD",
    "coregd": "CoRe-GD", "dnn2": "(DNN)²", "motion_a10": "GLIDE +L_motion",
}
# Preferred left-to-right order for any known stems that are shown (others go
# after these, alphabetically); GLIDE is forced last so it reads "... vs ours".
PREFERRED = ["deepdrawing", "gnd_fr", "gnd_stress", "dnn2", "coregd", "smartgd"]

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def discover(pred_dir):
    if not os.path.isdir(pred_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(pred_dir) if f.endswith(".npz"))


def order_models(selected):
    """FR-free ordering: known PREFERRED first, unknowns alphabetical, GLIDE last."""
    sel = list(selected)
    glide = [m for m in sel if m in ("vo2",)]
    rest = [m for m in sel if m not in glide]
    known = [m for m in PREFERRED if m in rest]
    unknown = sorted(m for m in rest if m not in PREFERRED)
    return known + unknown + glide


def menu(stems):
    if not stems:
        sys.exit(f"No prediction files found in {DEFAULT_PRED_DIR}.")
    print("\nModels available to plot (FR ground truth is always shown):\n")
    for i, s in enumerate(stems, 1):
        print(f"  {i:>2}. {s:<14} {LABELS.get(s, '')}")
    print("\nEnter numbers to include (e.g. '1 3 5'), 'a' for all, 'q' to quit.")
    while True:
        raw = input("> ").strip().lower()
        if raw in ("q", "quit", ""):
            sys.exit("Nothing selected.")
        if raw in ("a", "all"):
            return stems
        try:
            idx = [int(t) for t in raw.replace(",", " ").split()]
            picked = [stems[i - 1] for i in idx if 1 <= i <= len(stems)]
            if picked:
                return picked
        except ValueError:
            pass
        print("Didn't understand — try e.g. '1 2 5', 'a', or 'q'.")


def load_predictions(path):
    z = np.load(path, allow_pickle=False)
    n = z["n_nodes"].astype(int)
    pos = np.split(z["positions"].astype(np.float64), np.cumsum(n)[:-1])
    return {int(s): p for s, p in zip(z["source_indices"].astype(int), pos)}


def draw(ax, P, edges, comm, title=None):
    seg = np.stack([P[edges[:, 0]], P[edges[:, 1]]], axis=1)
    ax.add_collection(LineCollection(seg, colors="#999999", linewidths=0.55,
                                     alpha=0.75, zorder=1))
    colors = ([PALETTE[c % len(PALETTE)] for c in comm]
              if comm is not None else PALETTE[0])
    ax.scatter(P[:, 0], P[:, 1], s=26, c=colors, zorder=2,
               edgecolors="white", linewidths=0.45)
    span = max(np.ptp(P[:, 0]), np.ptp(P[:, 1])) or 1.0
    cx, cy = P[:, 0].mean(), P[:, 1].mean()
    ax.set_xlim(cx - span * 0.62, cx + span * 0.62)
    ax.set_ylim(cy - span * 0.62, cy + span * 0.62)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, pad=8)


def main():
    ap = argparse.ArgumentParser(
        description="Render FR vs model layouts side by side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--models", nargs="+", metavar="STEM",
                    help="prediction stems to show as columns (order preserved). "
                         "Omit to pick interactively / use all found.")
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="explicit dataset indices for the rows; else auto-picked")
    ap.add_argument("--n_rows", type=int, default=5)
    ap.add_argument("--dataset_path", default=DEFAULT_DATASET)
    ap.add_argument("--pred_dir", default=DEFAULT_PRED_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not os.path.isfile(args.dataset_path):
        sys.exit(f"ERROR: dataset not found: {args.dataset_path}")
    stems = discover(args.pred_dir)
    if not stems:
        sys.exit(f"ERROR: no .npz predictions in {args.pred_dir}.")

    # resolve which models to show
    if args.models:
        selected = args.models
    elif sys.stdin.isatty():
        selected = order_models(menu(stems))
    else:
        selected = order_models(stems)
        print(f"No --models given; showing all found: {selected}")

    missing = [m for m in selected if m not in stems]
    if missing:
        sys.exit(f"ERROR: no prediction file for: {', '.join(missing)}\n"
                 f"Available: {', '.join(stems)}")

    columns = [("fr", "Ground truth (FR)")] + [(m, LABELS.get(m, m)) for m in selected]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_list = torch.load(args.dataset_path, weights_only=False)
    if isinstance(data_list, dict) and "dataset" in data_list:
        data_list = data_list["dataset"]

    preds = {m: load_predictions(os.path.join(args.pred_dir, f"{m}.npz")) for m in selected}
    common = sorted(set.intersection(*(set(p) for p in preds.values())))
    if not common:
        sys.exit("ERROR: the selected models share no common graphs to plot.")

    def ncomm(d):
        return int(d.community.max()) + 1 if getattr(d, "community", None) is not None else 1

    if args.indices:
        bad = [i for i in args.indices if i not in common]
        if bad:
            sys.exit(f"ERROR: indices not present in all selected models: {bad}")
        chosen = args.indices
    else:
        # spread across graph sizes, preferring multi-community graphs
        cand = sorted(((i, int(data_list[i].num_nodes)) for i in common
                       if ncomm(data_list[i]) >= 3), key=lambda t: t[1])
        cand = cand or sorted((i, int(data_list[i].num_nodes)) for i in common)
        picks = np.linspace(0, len(cand) - 1, min(args.n_rows, len(cand))).round().astype(int)
        chosen = [cand[p][0] for p in picks]

    nrow, ncol = len(chosen), len(columns)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 2.45 * nrow), squeeze=False)

    for r, src in enumerate(chosen):
        d = data_list[src]
        ei = d.edge_index.numpy()
        edges = ei[:, ei[0] < ei[1]].T
        comm = d.community.numpy() if getattr(d, "community", None) is not None else None
        target = d.y.numpy().astype(np.float64)
        for c, (key, label) in enumerate(columns):
            ax = axes[r][c]
            if key == "fr":
                P = target
            else:
                P, _ = procrustes_align(preds[key][src], target)
            draw(ax, P, edges, comm, title=label if r == 0 else None)
        axes[r][0].set_ylabel(f"{int(d.num_nodes)} nodes\n{ncomm(d)} communities",
                              fontsize=11, style="italic", labelpad=12,
                              rotation=0, ha="right", va="center")

    fig.subplots_adjust(left=0.085, right=0.995, top=0.945, bottom=0.01,
                        wspace=0.05, hspace=0.05)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"\nWrote {args.out}")
    print(f"Columns: {' | '.join(lab for _, lab in columns)}")
    print(f"Rows (dataset indices): {chosen}")


if __name__ == "__main__":
    main()
