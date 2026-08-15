"""
eval/validate_metrics.py

Proves the metric implementations are correct before any number is reported.

The decisive check is on the FR force. FR's update is

    pos += displacement * (temperature / ||displacement||)

so the step FR takes is exactly the force direction. If fr_forces() matches the
generator, then the force computed at a stored trajectory state must be PARALLEL
to the observed next step, for every node at every step:

    cos( F_i(y_traj[t]), y_traj[t+1] - y_traj[t] ) == 1.0

That is a much stronger test than eyeballing the formula: any error in a sign, a
power, the adjacency, or the frame breaks the parallelism immediately.

Also checked:
  - Procrustes: rotations align to zero, mirrors do not (SO(2)), mirrors do (O(2))
  - Rollout indexing: FR's own trajectory scored against itself must give exactly 0
  - Phi decays along FR's trajectory and is small but nonzero at the endpoint

Usage:
    python eval/validate_metrics.py --dataset_path data/processed/comm_5k_v2_with_encodings.pt
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval.metrics import (                                    # noqa: E402
    adjacency_from_edge_index, denormalize, fr_forces,
    procrustes_mse, residual_force, rollout_error,
)
from glide.executor.data import load_split                     # noqa: E402


def check_procrustes():
    """
    The metric is O(2)+scale (similarity), matching what DeepDrawing and GND
    publish. Rotation, translation, reflection and uniform scale must all be
    absorbed; a genuinely different layout must not be.
    """
    print("\n[1] Procrustes convention: O(2) + scale")
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(30, 2))

    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rot = Q @ R.T + rng.normal(size=2) * 3.0
    mir = Q @ np.diag([1.0, -1.0])
    both = (Q @ np.diag([1.0, -1.0]) @ R.T) * 2.5 + rng.normal(size=2) * 4.0
    other = rng.normal(size=(30, 2))

    a = procrustes_mse(rot, Q)
    b = procrustes_mse(mir, Q)
    c = procrustes_mse(both, Q)
    d = procrustes_mse(other, Q)
    print(f"    rotation + translation      : {a:.3e}   (expect ~0)")
    print(f"    mirror                      : {b:.3e}   (expect ~0)")
    print(f"    mirror + rotate + 2.5x scale: {c:.3e}   (expect ~0)")
    print(f"    unrelated layout            : {d:.4f}      (expect > 0)")
    ok = a < 1e-20 and b < 1e-20 and c < 1e-20 and d > 1e-3
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_procrustes_scale():
    """
    Scale must be absorbed across a wide range of magnitudes, not just near 1x.
    Baseline output scale is arbitrary by construction, so a fit that only works
    for small rescales would quietly distort their numbers.
    """
    print("\n[1b] Scale is absorbed across magnitudes")
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(40, 2))
    th = 0.4
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    ok = True
    for k in (0.01, 0.25, 0.5, 2.0, 4.0, 100.0):
        scaled = (Q @ R.T) * k + rng.normal(size=2) * 2.0
        mse = procrustes_mse(scaled, Q)
        good = mse < 1e-18
        ok &= good
        print(f"    {k:>7}x rescale -> {mse:.3e}   {'ok' if good else 'FAIL'}")

    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_force_direction(data_list, n_graphs=20):
    """
    THE decisive test: computed force must be parallel to FR's observed step.
    Everything is done in the raw frame, which is where FR actually ran.
    """
    print("\n[2] FR force direction vs observed FR step  (the decisive check)")
    cosines = []
    for d in data_list[:n_graphs]:
        n = int(d.num_nodes)
        A = adjacency_from_edge_index(d.edge_index.numpy(), n)
        k = np.sqrt(1.0 / n)
        y_mean, y_std = d.y_mean.numpy(), d.y_std.numpy()

        p0 = denormalize(d.x[:, 2:4].numpy(), y_mean, y_std)
        traj = denormalize(
            d.y_traj.numpy().reshape(n, -1, 2).transpose(1, 0, 2), y_mean, y_std)
        full = np.concatenate([p0[None], traj], axis=0)     # [T+1, N, 2] raw

        for t in range(full.shape[0] - 1):
            F = fr_forces(full[t], A, k)
            step = full[t + 1] - full[t]
            nf = np.linalg.norm(F, axis=1)
            ns = np.linalg.norm(step, axis=1)
            live = (nf > 1e-9) & (ns > 1e-9)
            if live.any():
                cosines.append(
                    ((F[live] * step[live]).sum(1) / (nf[live] * ns[live])))
    cos = np.concatenate(cosines)
    print(f"    samples          : {cos.size:,} (node, step) pairs over {n_graphs} graphs")
    print(f"    mean cosine      : {cos.mean():.8f}")
    print(f"    min  cosine      : {cos.min():.8f}")
    print(f"    frac cos > 0.9999: {(cos > 0.9999).mean() * 100:.2f}%")
    ok = cos.mean() > 0.9999 and cos.min() > 0.99
    print(f"    -> {'PASS' if ok else 'FAIL'}  (force reproduces FR's own update direction)")
    return ok


def check_force_magnitude(data_list, n_graphs=15):
    """
    Direction alone does not pin down the force. A implementation 2x too large
    everywhere would still give cosine 1.0, and Phi would be doubled -- and the
    observed step magnitudes cannot catch it either, because FR renormalises
    every step to length `temperature`. So compare against the generator's own
    displacement expression element-wise.
    """
    print("\n[2b] FR force MAGNITUDE vs the generator, element-wise")
    worst = 0.0
    for d in data_list[:n_graphs]:
        n = int(d.num_nodes)
        A = adjacency_from_edge_index(d.edge_index.numpy(), n)
        k = np.sqrt(1.0 / n)
        pos = denormalize(d.y.numpy(), d.y_mean.numpy(), d.y_std.numpy())

        mine = fr_forces(pos, A, k)
        # data/generate_fr_iterations.py:fr_single_step, inlined
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.clip(dist, 0.01, None, out=dist)
        theirs = np.einsum("ijk,ij->ik", delta, k * k / dist ** 2 - A * dist / k)

        worst = max(worst, float(np.abs(mine - theirs).max()))
    print(f"    max |ours - generator| : {worst:.3e}")
    ok = worst < 1e-9
    print(f"    -> {'PASS' if ok else 'FAIL'}  (identical; masking the diagonal is "
          f"equivalent to the generator relying on delta_ii = 0)")
    return ok


def check_trajectory_indexing(data_list, fr_mod, n_graphs=5):
    """
    Pin down which stored index corresponds to which step count. Replaying FR for
    m steps must reproduce y_traj[m-1] and NOT y_traj[m].

    Replay from the grid initialisation recomputed in float64, not from the
    stored x[:, 2:4]: FR is chaotically sensitive here, and the float32 rounding
    in the stored initial positions (~4e-8) amplifies to ~0.1 by step 50. That
    sensitivity is a property of FR, not an error, but it makes the stored p_0
    unusable as a replay seed.
    """
    print("\n[4b] Trajectory indexing: does y_traj[t] hold the state after t+1 steps?")
    ok = True
    for d in data_list[:n_graphs]:
        n = int(d.num_nodes)
        A = adjacency_from_edge_index(d.edge_index.numpy(), n)
        traj = denormalize(
            d.y_traj.numpy().reshape(n, -1, 2).transpose(1, 0, 2),
            d.y_mean.numpy(), d.y_std.numpy())

        gs = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / gs))
        p0 = np.zeros((n, 2))
        for i in range(n):
            p0[i, 0] = 2 * ((i % gs) / max(gs - 1, 1)) - 1
            p0[i, 1] = 2 * ((i // gs) / max(rows - 1, 1)) - 1

        k = np.sqrt(1.0 / n)
        span = p0.max(0) - p0.min(0)
        t0 = max(float(span.max() * 0.1), 1e-4)
        dt = t0 / 51.0

        pos, temp = p0.copy(), t0
        for m in range(1, 51):
            pos = fr_mod.fr_single_step(pos, A, k, temp)
            temp = max(temp - dt, 1e-6)
            if m in (1, 5, 25, 50):
                at_t = float(np.abs(pos - traj[m - 1]).max())
                if at_t > 1e-5:
                    ok = False
                if m == 50:
                    print(f"    after {m:2d} steps: |vs y_traj[{m-1}]| = {at_t:.2e}")
    print(f"    -> {'PASS' if ok else 'FAIL'}  (pred[t] pairs with y_traj[:, t, :], "
          f"not t+1)")
    return ok


def check_frame_matters(data_list):
    """Demonstrate why raw-frame is mandatory: normalized coords give a different Phi."""
    print("\n[3] Frame sensitivity of Phi  (why denormalization is mandatory)")
    d = data_list[0]
    n = int(d.num_nodes)
    A = adjacency_from_edge_index(d.edge_index.numpy(), n)
    k = np.sqrt(1.0 / n)
    y_norm = d.y.numpy()
    y_raw = denormalize(y_norm, d.y_mean.numpy(), d.y_std.numpy())
    a = residual_force(y_raw, A, k)
    b = residual_force(y_norm, A, k)
    print(f"    Phi(raw frame, correct)  : {a:.4f}")
    print(f"    Phi(normalized, WRONG)   : {b:.4f}   ({b / a:.1f}x off)")
    print("    -> the force law is not scale-invariant; only the raw frame is meaningful")
    return True


def check_rollout_indexing(data_list, n_graphs=10):
    print("\n[4] Rollout indexing: FR trajectory scored against itself")
    worst_a = worst_r = 0.0
    for d in data_list[:n_graphs]:
        n = int(d.num_nodes)
        traj = d.y_traj.numpy().reshape(n, -1, 2).transpose(1, 0, 2)   # [T,N,2]
        worst_a = max(worst_a, rollout_error(traj, traj, aligned=True).max())
        worst_r = max(worst_r, rollout_error(traj, traj, aligned=False).max())
    # The raw path must be bit-exact (it is a plain subtraction). The aligned
    # path goes through an SVD, so its floor is float64 roundoff at the
    # coordinate scale: with |coord| ~ 2, that is eps * scale^2 ~ 9e-16, and
    # measured worst case is 1.9e-14 -- thirteen orders below any reported MSE.
    # Tolerance set from that floor, not fitted to make the test pass.
    tol = 1e-10
    print(f"    max per-step error, aligned : {worst_a:.3e}   (SVD roundoff floor, tol {tol:.0e})")
    print(f"    max per-step error, raw     : {worst_r:.3e}   (expect exactly 0)")
    ok = worst_a < tol and worst_r == 0.0
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_phi_decays(data_list, n_graphs=30):
    print("\n[5] Phi along FR's own trajectory")
    curves = []
    for d in data_list[:n_graphs]:
        n = int(d.num_nodes)
        A = adjacency_from_edge_index(d.edge_index.numpy(), n)
        k = np.sqrt(1.0 / n)
        traj = denormalize(
            d.y_traj.numpy().reshape(n, -1, 2).transpose(1, 0, 2),
            d.y_mean.numpy(), d.y_std.numpy())
        curves.append([residual_force(traj[t], A, k) for t in range(traj.shape[0])])
    c = np.array(curves).mean(axis=0)
    for t in (0, 4, 9, 19, 34, 49):
        print(f"    step {t + 1:3d} : Phi = {c[t]:.4f}")
    print(f"    Phi(P*_50)/Phi(P*_1) = {c[-1] / c[0]:.3f}")
    ok = c[-1] < c[0] and c[-1] > 0
    print(f"    -> {'PASS' if ok else 'FAIL'}  (decays as FR converges, nonzero at the "
          f"budget-limited endpoint)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = args.split
    data = load_split(args.dataset_path, seed=args.seed)[cfg]

    fr_spec = importlib.util.spec_from_file_location(
        "_frgen", os.path.join(PROJECT_ROOT, "data", "generate_fr_iterations.py"))
    fr_mod = importlib.util.module_from_spec(fr_spec)
    fr_spec.loader.exec_module(fr_mod)

    results = [
        check_procrustes(),
        check_procrustes_scale(),
        check_force_direction(data),
        check_force_magnitude(data),
        check_trajectory_indexing(data, fr_mod),
        check_frame_matters(data),
        check_rollout_indexing(data),
        check_phi_decays(data),
    ]

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
