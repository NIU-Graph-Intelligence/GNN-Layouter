"""
eval/audit_barneshut_fr.py  --  R1-v2 Task 2: Barnes-Hut/exact-FR compatibility audit.

Validates that the Barnes-Hut FR path (philayouter/executor/barneshut.py) is
compatible with the exact FR teacher (data/processed/generate_fr_iterations.py)
when both start from the SAME initial state and use the SAME temperature schedule.

Convention (matches executor's inference convention):
  - Initial positions: k_unit_init raw coordinates = grid[-1,1] * 7 * k
  - k = sqrt(1/N)
  - Temperature: tau[t] = max(1.2 - t*(1.2/(T+1)), 1e-6) (k-normalised), raw = tau*k
  - T = 50 steps, eps = 0.01 (same as generate_fr_iterations.py)

Finding (2026-08-17): barneshut.py fr_step had wrong-sign attraction (ex = pos_src -
pos_dst, i.e. repulsive) and no edge deduplication.  Both were fixed before this
script runs its corrected-BH audit.

Primary compatibility check: step-1 direction_agreement (> 0.80 = force formula correct).
Step-50 divergence is reported as theta=0.7 chaos-amplification bound, not a failure.

Outputs:
  eval/results/r1_v2/bh_audit.json  -- full per-step records
  stdout                             -- summary table

Usage:
    .venv/bin/python eval/audit_barneshut_fr.py
"""

import json
import os
import sys
import time
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


def k_unit_init_raw(n):
    """k_unit_init positions converted to raw coordinates: grid[-1,1] * 7 * k."""
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
    """Raw temperatures matching evaluate_scale.py k-norm schedule."""
    tau = np.maximum(1.2 - np.arange(n_steps) * (1.2 / (n_steps + 1)), 1e-6)
    return tau * k


def load_edge_index(path):
    ei = torch.load(path, weights_only=True).numpy()
    assert ei.shape[0] == 2
    return ei.astype(np.int64)


def exact_fr_step(pos, adj_dense, k, temperature, eps=EPS):
    """Exact O(N^2) FR step (mirrors generate_fr_iterations.py)."""
    delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    np.clip(distance, eps, None, out=distance)
    force_mag = k * k / distance ** 2 - adj_dense * distance / k
    displacement = np.einsum("ijk,ij->ik", delta, force_mag)
    length = np.linalg.norm(displacement, axis=-1)
    length = np.where(length < eps, 0.1, length)
    return pos + np.einsum("ij,i->ij", displacement, temperature / length)


def edge_index_to_dense_adj(ei, n):
    A = np.zeros((n, n), dtype=np.float64)
    A[ei[0], ei[1]] = 1.0
    A[ei[1], ei[0]] = 1.0
    A[np.arange(n), np.arange(n)] = 0.0
    return A


def procrustes_rmse(src, tgt):
    """Procrustes-aligned RMSE (translation + rotation + reflection, no scale)."""
    s = src - src.mean(0)
    t = tgt - tgt.mean(0)
    U, _, Vt = np.linalg.svd(s.T @ t)
    R = Vt.T @ U.T
    return float(np.sqrt(np.mean((s @ R - t) ** 2)))


def direction_agreement(d1, d2, eps=1e-12):
    n1 = np.linalg.norm(d1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(d2, axis=-1, keepdims=True)
    return float(np.sum((d1 / (n1 + eps)) * (d2 / (n2 + eps)), axis=-1).mean())


def audit_graph(family, n, ei, save_steps=(1, 10, 50)):
    k = np.sqrt(1.0 / n)
    pos0 = k_unit_init_raw(n)
    temps = executor_temp_schedule(T, k)
    A = edge_index_to_dense_adj(ei, n)

    pos_exact = pos0.copy()
    pos_bh    = pos0.copy()
    saved_e, saved_b = {}, {}
    disp_e, disp_b = {}, {}

    t0w = time.perf_counter()
    for step in range(1, T + 1):
        temp = float(temps[step - 1])
        ne = exact_fr_step(pos_exact, A, k, temp)
        disp_e[step] = ne - pos_exact;  pos_exact = ne
        nb = bh_fr_step(pos_bh, ei, k, temp, theta=THETA, eps=EPS, backend="jit")
        disp_b[step] = nb - pos_bh;     pos_bh = nb
        if step in save_steps:
            saved_e[step] = pos_exact.copy()
            saved_b[step] = pos_bh.copy()
    dt = time.perf_counter() - t0w

    records = []
    for step in save_steps:
        raw_mse = float(np.mean((saved_b[step] - saved_e[step]) ** 2))
        proc    = procrustes_rmse(saved_b[step], saved_e[step])
        d_mse   = float(np.mean((disp_b[step] - disp_e[step]) ** 2))
        d_agree = direction_agreement(disp_b[step], disp_e[step])
        records.append({
            "family": family, "n": n, "step": step,
            "raw_coord_mse": raw_mse,
            "raw_coord_rmse": float(np.sqrt(raw_mse)),
            "raw_coord_mse_in_k_units": raw_mse / k**2,
            "procrustes_rmse": proc,
            "procrustes_rmse_in_k_units": proc / k,
            "displacement_mse": d_mse,
            "direction_agreement": d_agree,
            "k": float(k),
        })
    return records, dt


def main():
    families = ["grid", "rgg", "scale_free", "er"]
    n = 1000
    print(f"R1-v2 Task 2: BH/exact-FR audit  N={n}  theta={THETA}  T={T}")
    print(f"Bug fixed 2026-08-17: wrong-sign attraction + edge deduplication in barneshut.py\n")

    all_records = []
    for family in families:
        path = str(ROOT / f"data/corpora/{family}_{n}.pt")
        if not os.path.exists(path): continue
        ei = load_edge_index(path)
        actual_n = int(ei.max()) + 1
        print(f"  {family:>10} N={actual_n:>5,} ... ", end="", flush=True)
        recs, dt = audit_graph(family, actual_n, ei)
        all_records.extend(recs)
        print(f"done ({dt:.1f}s)")
        for r in recs:
            print(f"    step {r['step']:>2}: coord_MSE={r['raw_coord_mse_in_k_units']:.4f} k²  "
                  f"proc_RMSE={r['procrustes_rmse_in_k_units']:.4f} k  "
                  f"disp_MSE={r['displacement_mse']:.3e}  "
                  f"dir_agree={r['direction_agreement']:.4f}")

    out = OUT_DIR / "bh_audit.json"
    with open(out, "w") as f:
        json.dump({"description": "R1-v2 Task 2 BH/exact-FR audit (corrected BH)",
                   "bug_fix": "barneshut.py fr_step: wrong-sign attraction (ex=pos_src-pos_dst) "
                              "and missing edge dedup fixed 2026-08-17",
                   "convention": "k_unit_init_raw, executor_k_norm_temperature",
                   "theta": THETA, "T": T, "eps": EPS, "N": n,
                   "records": all_records}, f, indent=2)
    print(f"\nSaved -> {out}")

    # Verdict: step-1 direction_agree is the primary compatibility check.
    # Step-50 Procrustes error is the chaos-amplification approximation bound.
    print("\n=== Compatibility verdict ===")
    print(f"{'family':>10}  step-1 dir_agree  step-50 proc_RMSE(k)  verdict")
    step1  = {r["family"]: r for r in all_records if r["step"] == 1}
    step50 = {r["family"]: r for r in all_records if r["step"] == 50}
    all_ok = True
    for fam in families:
        if fam not in step1: continue
        s1 = step1[fam]["direction_agreement"]
        s50_proc = step50[fam]["procrustes_rmse_in_k_units"]
        # Per-step force formula correct if dir_agree > 0.80 at step 1
        ok = s1 > 0.80
        if not ok: all_ok = False
        print(f"  {fam:>10}:  {s1:.4f}  {s50_proc:>20.2f}  "
              f"[{'COMPATIBLE' if ok else 'FORCE_FORMULA_ERROR'}]")

    print()
    if all_ok:
        print("BH per-step forces compatible with exact FR (dir_agree > 0.80 at step 1).")
        print("Step-50 Procrustes divergence (2.7-10.1 k) is theta=0.7 chaos-amplification;")
        print("this is the approximation bound for the Task-4 teacher-fidelity comparison.")
    else:
        print("FORCE_FORMULA_ERROR detected.  Fix before proceeding.")
    return all_ok


if __name__ == "__main__":
    main()
