"""
eval/equivariance_check.py  --  equivariance sanity check.

Verifies that the BH FR teacher is equivariant under node-label permutation:
  BH(permute(G, pi), permute(X0, pi)) = permute(BH(G, X0), pi)

FR is analytically equivariant to node-label permutation (it only depends on
topology, not on which integer labels the nodes carry).  This check verifies
the implementation (graph loading, permutation mapping, BH step) is correct.

Usage:
    .venv/bin/python eval/equivariance_check.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from philayouter.executor.barneshut import fr_step as bh_fr_step

OUT_DIR = ROOT / "eval/results/r1_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T = 50
EPS = 0.01
THETA = 0.7
PASS_THRESHOLD = 1e-5  # accommodate np.add.at FP non-commutativity over 50 steps


def k_unit_init_raw(n):
    k = np.sqrt(1.0 / n)
    gs = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / gs))
    pos = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        u = (i % gs) / max(gs - 1, 1)
        v = (i // gs) / max(rows - 1, 1)
        pos[i, 0] = 2 * u - 1
        pos[i, 1] = 2 * v - 1
    return pos * 7.0 * k


def executor_temp_schedule(n_steps, k):
    tau = np.maximum(1.2 - np.arange(n_steps) * (1.2 / (n_steps + 1)), 1e-6)
    return tau * k


def bh_rollout(ei, n, x0=None, save_steps=(1, 10, 50)):
    """Run 50-step BH FR from x0 (or k_unit_init if None), return saved positions."""
    k     = np.sqrt(1.0 / n)
    pos   = x0.copy() if x0 is not None else k_unit_init_raw(n)
    temps = executor_temp_schedule(T, k)
    saved = {}
    for step in range(1, T + 1):
        pos = bh_fr_step(pos, ei, k, float(temps[step - 1]),
                         theta=THETA, eps=EPS, backend="jit")
        if step in save_steps:
            saved[step] = pos.copy()
    return saved


def permute_ei(ei, perm):
    """Apply permutation perm[old_id] = new_id to edge_index [2,E]."""
    new_ei = np.stack([perm[ei[0]], perm[ei[1]]], axis=0)
    return new_ei


def check_equivariance(family, n, ei, perm_seed=42_000):
    """Apply a random permutation, check that BH output permutes consistently.

    Equivariance: BH(permute(G, pi), permute(X0, pi)) = permute(BH(G, X0), pi)

    perm[old_id] = new_id, so node i in G maps to node perm[i] in G'.
    X0'[new_id] = X0[old_id] = X0[inv_perm[new_id]], i.e., X0' = X0[inv_perm].
    After rolling out: y_perm[perm[i]] should equal y_orig[i] for all i.
    """
    rng      = np.random.default_rng(perm_seed)
    perm     = rng.permutation(n)       # perm[old_id] = new_id
    inv_perm = np.argsort(perm)         # inv_perm[new_id] = old_id

    x0_orig = k_unit_init_raw(n)
    # X0 for permuted graph: node perm[i] gets position x0_orig[i]
    # => x0_perm[j] = x0_orig[inv_perm[j]]
    x0_perm = x0_orig[inv_perm]

    # Run BH on original (G, X0)
    saved_orig = bh_rollout(ei, n, x0=x0_orig)

    # Run BH on permuted (G', X0') -- must start from identically permuted state
    ei_perm    = permute_ei(ei, perm)
    saved_perm = bh_rollout(ei_perm, n, x0=x0_perm)

    results = {}
    for step in (1, 10, 50):
        y_orig = saved_orig[step]                   # shape [n, 2]
        y_perm = saved_perm[step]                   # shape [n, 2]
        # y_perm[perm[i]] == y_orig[i] for all i
        # => y_orig == y_perm[perm]
        y_perm_inv = y_perm[perm]                   # reordered back to original IDs
        max_abs = float(np.max(np.abs(y_perm_inv - y_orig)))
        rms     = float(np.sqrt(np.mean((y_perm_inv - y_orig) ** 2)))
        results[step] = {"max_abs_err": max_abs, "rms_err": rms}

    return results


def main():
    import os
    print(f"Equivariance sanity check (BH teacher, N=1000 corpus)\n"
          f"Threshold: {PASS_THRESHOLD:.0e} (residual is FP non-commutativity)\n")

    families = ["grid", "rgg", "scale_free", "er"]
    n_target = 1000

    all_ok = True
    records = []
    for family in families:
        path = str(ROOT / f"data/corpora/{family}_{n_target}.pt")
        if not os.path.exists(path):
            print(f"  [skip] {family}: no file")
            continue
        ei = torch.load(path, weights_only=True).numpy().astype(np.int64)
        n  = int(ei.max()) + 1

        print(f"  {family:>10} N={n:,} ... ", end="", flush=True)
        res = check_equivariance(family, n, ei)
        ok  = all(res[s]["max_abs_err"] < PASS_THRESHOLD for s in (1, 10, 50))
        if not ok: all_ok = False
        print(f"{'OK' if ok else 'FAIL'}  max_abs_err: "
              f"s1={res[1]['max_abs_err']:.2e}  "
              f"s10={res[10]['max_abs_err']:.2e}  "
              f"s50={res[50]['max_abs_err']:.2e}")
        records.append({"family": family, "n": n,
                        **{f"step{s}": res[s] for s in (1, 10, 50)},
                        "pass_threshold": PASS_THRESHOLD, "ok": ok})

    note = ("Residual errors ~1e-9..1e-6 at step 50 are floating-point "
            "accumulation-order artefacts from np.add.at when edge ordering "
            "changes under permutation. Not a physics error.")
    out = OUT_DIR / "equivariance_check.json"
    with open(out, "w") as f:
        json.dump({"description": "BH teacher equivariance check (fixed X0 permutation)",
                   "method": "BH_FR", "T": T, "theta": THETA,
                   "pass_threshold": PASS_THRESHOLD,
                   "fp_commutativity_note": note,
                   "verdict": "PASS" if all_ok else "FAIL",
                   "records": records}, f, indent=2)
    print(f"\n{'PASS' if all_ok else 'FAIL'}: BH FR equivariance under node permutation")
    print(f"Saved -> {out}")
    return all_ok


if __name__ == "__main__":
    main()
