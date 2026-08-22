"""
eval/convert_real_graphs.py

Convert real graphs that arrive at
data/corpora/real/ (Matrix Market .mtx or two-column edge lists) into the same
sparse representation generate_corpus.py emits -- a [2,E] int64 torch tensor
per graph saved as <name>.pt -- so evaluate_scale.py and the sampled scorer run
on them unchanged.

Node ids are renumbered to 0..n-1 (real corpora do not use contiguous ids);
self-loops are dropped and duplicate edges removed. A registry JSON records
n, E and the source file for provenance.

Usage:
    .venv/bin/python eval/convert_real_graphs.py --in_dir data/corpora/real \
        --out_dir data/corpora/real
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from eval.score_large_layout import load_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/corpora/real")
    ap.add_argument("--out_dir", default="data/corpora/real")
    ap.add_argument("--ext", nargs="+", default=[".mtx", ".edges", ".edgelist", ".txt", ".graph"])
    args = ap.parse_args()

    if not os.path.isdir(args.in_dir):
        raise SystemExit(f"no such directory: {args.in_dir} (real graphs not yet arrived)")

    files = sorted(
        f for f in os.listdir(args.in_dir)
        if os.path.splitext(f)[1].lower() in args.ext)
    if not files:
        raise SystemExit(f"no {args.ext} files in {args.in_dir}")

    registry = {}
    for fname in files:
        path = os.path.join(args.in_dir, fname)
        name = os.path.splitext(fname)[0]
        ei, n = load_graph(path)
        E = ei.shape[1]
        out = os.path.join(args.out_dir, f"{name}.pt")
        torch.save(torch.from_numpy(ei), out)
        registry[name] = {"source": fname, "n": int(n), "E": int(E),
                          "pt": os.path.relpath(out, ".")}
        print(f"{name:>20}  n={n:>9,}  E={E:>11,}  -> {out}", flush=True)

    with open(os.path.join(args.out_dir, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)
    print(f"registry -> {os.path.join(args.out_dir, 'registry.json')}")


if __name__ == "__main__":
    main()
