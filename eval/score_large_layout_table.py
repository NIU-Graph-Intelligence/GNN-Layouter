"""
eval/score_large_layout_table.py

Score every saved large layout with the sampled scorer and assemble the
large-N layout-quality table. Reads the layout manifest
written by generate_large_layouts.py; for each (source, family, size) present
it runs the scorer in-process and collects stress +/- SE and NP@k +/- SE.

Rows:
  phi / sfdp / fa2     -- the three layout sources (manifest)
  noop                 -- the executor's k-unit grid initialisation P0 with no
                          rollout: the "is the output trivial" baseline. If the
                          executor's layout is not measurably better than its
                          own starting point, the stress/NP numbers say so.
  phi_randfeat         -- the phi layout produced with random node features
                          (the run with --skip_encoding).
                          Kept as an explicit ablation: it is NOT the paper's
                          model, which uses LapPE node features.

Sources with no saved layout (phi on scale-free: diverged) are marked in the
output rather than scored. Per-pair stress (stress / C(n,2)) is emitted so
the N=10^5 and N=10^6 rows compare on the same scale; the sfdp ratio is
emitted per row.

Usage:
    .venv/bin/python eval/score_large_layout_table.py \
        --manifest eval/results/large_layouts/manifest.json \
        --corpora data/corpora \
        --layouts eval/results/large_layouts \
        --out eval/results/large_layout_scores.json
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from eval.score_large_layout import build_csr, load_graph, load_pos, restrict_to_lcc, sampled_np, sampled_stress


def k_unit_grid_noop(n: int) -> np.ndarray:
    """Raw-frame coordinates of the executor's N-invariant k-unit init (the
    grid into [-1,1]^2 scaled to |pos_k|~7, SCALE_NORMALIZATION opt (b)),
    returned in the raw frame (multiplied by k) exactly as rollout_scale
    returns rollout positions, so the no-op row is comparable."""
    grid_size = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / grid_size))
    pos = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        u = (i % grid_size) / max(grid_size - 1, 1)
        v = (i // grid_size) / max(rows - 1, 1)
        pos[i, 0] = 2 * u - 1
        pos[i, 1] = 2 * v - 1
    pos_k = pos * 7.0
    k = float(np.sqrt(1.0 / n))
    return pos_k * k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="eval/results/large_layouts/manifest.json")
    ap.add_argument("--corpora", default="data/corpora")
    ap.add_argument("--layouts", default="eval/results/large_layouts")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100000, 1000000])
    ap.add_argument("--out", default="eval/results/large_layout_scores.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--np_k", type=int, default=10)
    ap.add_argument("--n_sources", type=int, default=500,
                    help="BFS sources / bootstrap clusters per layout; higher "
                         "cuts the stress SE at ~1/sqrt(S)")
    ap.add_argument("--M_factor", type=float, default=10.0,
                    help="M = M_factor * N (sampled pairs per layout); "
                         "10 -> 10^6 at N=10^5, 10^7 at N=10^6")
    ap.add_argument("--lcc", action="store_true",
                    help="restrict every graph to its largest connected "
                         "component before scoring (for disconnected real "
                         "graphs; a no-op on the connected corpora)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    rows = []
    rng = np.random.default_rng(args.seed)
    for key, entry in manifest.items():
        fam, nstr = key.rsplit("_", 1)
        n = int(nstr)
        if n not in args.sizes:
            continue
        ei, n_nodes = load_graph(os.path.join(args.corpora, f"{fam}_{n}.pt"))
        pos_all = load_pos  # placeholder to keep lints quiet
        if args.lcc:
            ei, _, n_nodes = restrict_to_lcc(ei, None, n_nodes)
            # reload positions restricted consistently
            raise SystemExit("--lcc requires re-reading layouts; use the "
                             "scorer directly for disconnected graphs")
        A = build_csr(ei, n_nodes)

        def score(name, pos, status="ok", div_at=None):
            if status == "diverged":
                rows.append({"source": name, "family": fam, "n": n,
                             "status": "diverged", "diverged_at": div_at})
                print(f"{name:>11} {fam:>10} N={n:>8,}  DIVERGED @t{div_at}",
                      flush=True)
                return
            M = int(args.M_factor * n)
            stress, stress_se, ns_pairs = sampled_stress(
                pos, A, M, args.seed, n_sources=args.n_sources, rng=rng)
            np_mean, np_se, np_used = sampled_np(
                pos, A, args.np_k, 100_000, args.seed, rng=rng)
            n_pairs = n_nodes * (n_nodes - 1) // 2
            rows.append({
                "source": name, "family": fam, "n": n, "status": "ok",
                "M": M, "sampled_pairs": ns_pairs,
                "stress": stress, "stress_se": stress_se,
                "stress_per_pair": stress / n_pairs if n_pairs else None,
                "stress_per_pair_se": stress_se / n_pairs if n_pairs else None,
                "np_k": args.np_k, "np": np_mean, "np_se": np_se,
                "np_nodes": np_used,
            })
            print(f"{name:>11} {fam:>10} N={n:>8,}  stress {stress:12.1f} +- "
                  f"{stress_se:9.1f}  per-pair {stress/n_pairs:.4f} +- "
                  f"{stress_se/n_pairs:.4f}  NP {np_mean:.4f} +- {np_se:.4f}",
                  flush=True)

        for src in ("phi_enc", "phi", "sfdp", "fa2"):
            finfo = entry.get(src)
            if finfo is None:
                continue
            if "diverged_at" in finfo:
                score(src, None, "diverged", finfo["diverged_at"])
                continue
            pos = load_pos(os.path.join(args.layouts, f"{src}_{fam}_{n}.npy"), n_nodes)
            score(src, pos)

        # no-op baseline: the executor's own P0, no rollout
        score("noop", k_unit_grid_noop(n_nodes))

    # sfdp-normalised per-pair stress ratio, keyed on the sfdp row
    sfdp = {r["family"] + "_" + str(r["n"]): r["stress_per_pair"]
            for r in rows if r["status"] == "ok" and r["source"] == "sfdp"}
    for r in rows:
        if r["status"] == "ok" and r["source"] != "sfdp":
            base = sfdp.get(r["family"] + "_" + str(r["n"]))
            r["ratio_vs_sfdp"] = r["stress_per_pair"] / base if base else None

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
