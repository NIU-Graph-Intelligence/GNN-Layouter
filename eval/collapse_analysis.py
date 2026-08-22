"""
eval/collapse_analysis.py

Coincident-node collapse measurement on prediction .npz files, made
reusable and reproducible.

Definition (same as the paper's GND-collapse finding): two nodes are
"coincident" if their pairwise embedded distance is below 1e-6 (float-level
identical coordinates). A graph is "affected" if it contains at least one such
pair. GND (topology-only + Laplacian PE) collapses in 92.8% of graphs; the
random-feature control (GND + random feature) relieves it partially (86.4%);
the CoRe-GD control (random+beacon features removed, Laplacian-only input)
is expected to RE-introduce it if input symmetry drives the collapse.

This is a measurement of the prediction file, not of the model: it consumes
the shared eval/predictions/*.npz format (source_indices, n_nodes, positions,
meta) that every baseline dumps, so any row can be scored with the same rule.

Usage:
    .venv/bin/python eval/collapse_analysis.py eval/predictions/gnd_fr.npz \
        [eval/predictions/gnd_fr_randfeat.npz ...]
    .venv/bin/python eval/collapse_analysis.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "eval" / "predictions"

THRESHOLD = 1e-6  # float-level identical coordinates


def collapse_counts(positions: np.ndarray, n_nodes: np.ndarray, threshold: float = THRESHOLD):
    """Per-graph coincident-pair counts.

    Returns (affected_graphs, total_pairs, n_graphs): how many graphs contain
    at least one coincident pair, how many coincident pairs total, and how
    many graphs were examined (nodes with n < 2 are skipped)."""
    affected = 0
    total_pairs = 0
    n_graphs = 0
    i = 0
    for n in n_nodes:
        p = positions[i:i + n]
        i += n
        if n < 2:
            continue
        d = pdist(p)
        if d.size == 0:
            continue
        n_graphs += 1
        close = int((d < threshold).sum())
        if close:
            affected += 1
            total_pairs += close
    return affected, total_pairs, n_graphs


def analyze(path: Path):
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"])) if data["meta"].shape else json.loads(data["meta"].item())
    affected, pairs, n_graphs = collapse_counts(data["positions"], data["n_nodes"])
    pct = 100.0 * affected / n_graphs if n_graphs else 0.0
    return {
        "file": path.name,
        "name": meta.get("name", "?"),
        "family": meta.get("family", "?"),
        "affected_graphs": affected,
        "n_graphs": n_graphs,
        "pct_affected": pct,
        "coincident_pairs": pairs,
    }


def main():
    ap = argparse.ArgumentParser(description="coincident-node collapse analysis")
    ap.add_argument("paths", nargs="*", help="prediction .npz files; omit for default set")
    ap.add_argument("--all", action="store_true",
                    help="score every *.npz in eval/predictions/")
    args = ap.parse_args()

    if args.all:
        paths = sorted(PRED_DIR.glob("*.npz"))
    elif args.paths:
        # accept either a bare filename (suffix added), a relative path, or
        # an absolute path
        paths = []
        for p in args.paths:
            cand = Path(p)
            if cand.suffix != ".npz":
                cand = Path(p + ".npz")
            if not cand.is_absolute():
                cand = PRED_DIR / cand
            paths.append(cand if cand.exists() else PRED_DIR / cand.name)
    else:
        paths = sorted(PRED_DIR.glob("*.npz"))

    print(f"{'file':<30} {'model':<16} {'family':<14} {'affected':>9} "
          f"{'n_graphs':>9} {'pct':>7} {'pairs':>7}")
    print("-" * 100)
    results = []
    for p in paths:
        r = analyze(p)
        results.append(r)
        print(f"{r['file']:<30} {r['name']:<16} {r['family']:<14} "
              f"{r['affected_graphs']:>9} {r['n_graphs']:>9} "
              f"{r['pct_affected']:>6.1f}% {r['coincident_pairs']:>7}")


if __name__ == "__main__":
    main()
