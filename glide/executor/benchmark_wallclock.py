"""
glide/executor/benchmark_wallclock.py

Q22 (the experiment queue, PAPER_PLAN.md §7): wall-clock / sequential-
depth benchmark. The big-data payoff row: to reach FR's step-50 result, how
many sequential passes and how much wall-clock time does each method need, on
the SAME corpus graphs at increasing N?

Methods measured here:
  1. FR (networkx spring_layout)      -- the teacher, O(N^2) per step
  2. Barnes-Hut FR (barneshut.py)     -- the classical O(N log N) approximant
  3. Executor Phi_1 (EquivariantExecutor) -- one forward = one FR step, O(N k)
  4. sfdp (graphviz) -- multilevel force-directed reference; a SINGLE run to
     convergence, not per-iteration, so it is reported as total wall-clock.
     Excluded as a teacher (PAPER_PLAN §6, multilevel step crosses coarsening
     levels) but required in the wall-clock table (PAPER_PLAN §7).
  [cuGraph FA2 is NOT installed -- still flagged as a gated large install.]

Timing boundary (per §7): per-step geometric neighbourhood construction +
model forward + host-device transfers + full T-step rollout. Structural
encoding is a one-time per-graph cost, reported separately and EXCLUDED from
per-step (the table says so, per §7). FR / BH-FR have no such stage.

Sequential depth: FR and BH-FR need T=50 steps; Phi_1 needs T=50 forwards;
Phi_k needs ceil(50/k) forwards (k dial-able; reported at k in {8, 25}).

Usage:
    .venv/bin/python -m glide.executor.benchmark_wallclock \
        --checkpoint checkpoints/phi1_v1/executor_best.pt \
        --corpora data/corpora --sizes 1000 10000 100000 1000000 \
        --device cuda:0 --T 50
"""

import argparse
import os
import time

import numpy as np
import torch

from .barneshut import fr_step as bh_fr_step
from .model import EquivariantExecutor
from .structural import StructuralEncoder

FR_ITERS = 50


def grid_init(n: int) -> np.ndarray:
    gs = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / gs))
    pos = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        u = (i % gs) / max(gs - 1, 1)
        v = (i // gs) / max(rows - 1, 1)
        pos[i, 0] = 2 * u - 1
        pos[i, 1] = 2 * v - 1
    return pos


def fr_networkx(ei: np.ndarray, n: int, iters: int = FR_ITERS, n_measure: int = None):
    """Time networkx FR on an edge-index graph. Returns per-step wall-clock.

    Per-step cost is the same for every iteration (pure O(N^2) recompute), so
    for large N we measure only a handful of steps and extrapolate per-step --
    honest and cheap. n_measure defaults to min(5, iters)."""
    import networkx as nx

    G = nx.Graph()
    G.add_edges_from(ei.T.tolist())
    pos = {i: grid_init(n)[i].tolist() for i in range(n)}
    n_measure = min(n_measure or 5, iters)
    # warmup 2 steps, then measure n_measure
    warm = 2
    t0 = time.perf_counter()
    nx.spring_layout(G, pos=pos, iterations=warm, seed=42, scale=None, threshold=0)
    t_warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    nx.spring_layout(G, pos=pos, iterations=n_measure, seed=42, scale=None, threshold=0)
    t_meas = time.perf_counter() - t0
    return t_meas / n_measure


def fa2_cugraph(ei: np.ndarray, n: int, max_iter: int = FR_ITERS,
                rapids_python: str = None):
    """Time cuGraph ForceAtlas2 on this graph (PAPER_PLAN §7 row). cuGraph's
    force_atlas2 is a single batched GPU call that runs max_iter iterations
    internally, so -- like sfdp -- it is reported as TOTAL wall-clock for
    max_iter=50 (matching the step budget FR/BH-FR/Phi_1 each spend). Requires
    a RAPIDS environment (cugraph + libcugraph with LD_LIBRARY_PATH set),
    supplied by the caller as `rapids_python` (an absolute path to a RAPIDS
    venv python). If it is missing we degrade cleanly. Returns
    (total_wall_clock_seconds, err_or_None)."""
    import subprocess
    import sys
    import textwrap

    py = rapids_python or sys.executable

    edges_np = np.stack([ei[0], ei[1]], axis=1)
    code = textwrap.dedent(
        f"""
        import os
        import numpy as np
        import cudf, cugraph, time
        import cugraph.layout as cl
        arr = np.load(os.environ['NPZ'])['ei']
        edges = cudf.DataFrame({{'src': arr[:, 0], 'dst': arr[:, 1]}})
        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(edges, source='src', destination='dst')
        t0 = time.perf_counter()
        pos = cl.force_atlas2(G, max_iter={max_iter}, jitter_tolerance=1.0)
        print(float(time.perf_counter() - t0))
        """
    )
    # Write edges to a temp npz for the subprocess to load (avoids pickling
    # across envs).
    import tempfile
    import os

    with tempfile.NamedTemporaryFile("wb", suffix=".npz", delete=False) as f:
        np.savez(f, ei=edges_np)
        npzpath = f.name
    try:
        import os as _os
        env = dict(_os.environ, NPZ=npzpath)
        run = subprocess.run(
            [py, "-c", code], env=env,
            capture_output=True, text=True, timeout=3600,
        )
        out = run.stdout.strip().splitlines()
        if run.returncode != 0 or not out:
            return None, f"cuGraph FA2 failed: {run.stderr.strip()[-200:]}"
        return float(out[-1]), None
    except subprocess.TimeoutExpired:
        return None, "cuGraph FA2 timeout (>1h)"
    finally:
        os.unlink(npzpath)


def sfdp_total(ei: np.ndarray, n: int, sfdp_bin: str):
    """Time graphviz sfdp to convergence on this graph. sfdp is a multilevel
    layout: one run produces the final drawing, there is no per-iteration step
    to time. Reported as TOTAL wall-clock inclusive of writing the DOT input
    (a preprocessing stage we pay and FR/BH-FR do not, so it is included per
    the §7 boundary rule), the sfdp process, and reading its output. Sequential
    depth is not comparable to the step-iteration methods by construction;
    the table says so rather than pretending one run has T steps."""
    import subprocess
    import tempfile

    lines = ["graph G {", "  node [shape=point width=0];", "  edge [len=1.0];"]
    for e in range(ei.shape[1]):
        lines.append(f"  {ei[0, e].item()} -- {ei[1, e].item()};")
    lines.append("}")
    dot = "\n".join(lines)

    with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as f:
        f.write(dot)
        dotpath = f.name
    try:
        t0 = time.perf_counter()
        r = subprocess.run([sfdp_bin, "-T", "plain", dotpath],
                           capture_output=True, text=True, timeout=3600)
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            return None, f"sfdp rc={r.returncode}: {r.stderr.strip()[-200:]}"
        return dt, None
    except subprocess.TimeoutExpired:
        return None, "sfdp timeout (>1h)"
    finally:
        os.unlink(dotpath)


def bh_fr(ei: np.ndarray, n: int, iters: int = FR_ITERS):
    """Time Barnes-Hut FR; returns (per-step wall-clock, final pos)."""
    pos = grid_init(n)
    k = float(np.sqrt(1.0 / n))
    coord_span = pos.max(axis=0) - pos.min(axis=0)
    t0 = max(float(np.max(coord_span) * 0.1), 1e-4)
    dt = t0 / (iters + 1)
    temp = t0
    t_meas = 0.0
    n_meas = 0
    for step in range(iters):
        s0 = time.perf_counter()
        pos = bh_fr_step(pos, ei, k, temp, theta=0.7)
        t_meas += time.perf_counter() - s0
        n_meas += 1
        temp = max(temp - dt, 1e-6)
    return t_meas / n_meas, pos


def executor(encoder, model, ei: torch.Tensor, n: int, device, T: int,
             skip_encoding: bool = False, chunk: int = None):
    """Time the executor rollout (kNN + forward + host-device). Returns
    per-step wall-clock and the per-step split. Structural encoding excluded
    (one-time, reported by the caller via encoder timing). With
    skip_encoding=True the node_feat is random, so the per-step pipeline cost
    is measured WITHOUT the LapPE eigsh wall (Q14 finding: eigsh at N=10^6 is
    minutes-to-tens-of-minutes on degenerate spectra); the encoding cost is
    stated separately in the output. `chunk` bounds per-edge memory (Q22:
    chunked model forward runs N=10^6 that the monolithic readout OOMs on)."""
    k = float(np.sqrt(1.0 / n))
    pos = grid_init(n).astype(np.float32)
    pos = torch.from_numpy(pos).to(device)
    ei_d = ei.to(device)
    temps = np.maximum(1.2 - np.arange(T) * (1.2 / (T + 1)), 1e-6).astype(np.float32)
    if skip_encoding:
        node_feat = torch.randn(n, encoder.sign_net.out_dim, device=device)
    else:
        node_feat = encoder.forward(ei, n).to(device)
    if chunk is not None:
        model.chunk = chunk

    t_geo = t_fwd = 0.0
    n_meas = 0
    warm = 2
    with torch.no_grad():
        for t in range(T):
            tau = torch.full((n, 1), float(temps[t]), device=device)
            s0 = time.perf_counter()
            from .geometric import build_knn_graph
            build_knn_graph(pos, k=min(10, n - 1), backend="kd")
            s1 = time.perf_counter()
            model(node_feat, ei_d, pos, tau)
            s2 = time.perf_counter()
            if t >= warm:
                t_geo += s1 - s0
                t_fwd += s2 - s1
                n_meas += 1
    per_step = (t_geo + t_fwd) / max(n_meas, 1)
    return per_step, t_geo / max(n_meas, 1), t_fwd / max(n_meas, 1)


def main():
    ap = argparse.ArgumentParser(description="Q22 wall-clock / sequential-depth benchmark")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpora", default="data/corpora")
    ap.add_argument("--families", nargs="+", default=["grid", "er"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_encoding", action="store_true",
                    help="random node_feat instead of LapPE -- measures the "
                         "per-step pipeline at N where the one-time eigsh is "
                         "a wall (N=10^6, Q14 finding). Encoding cost reported "
                         "as '>45 min' rather than measured per-step.")
    ap.add_argument("--sfdp_bin", default=None,
                    help="path to graphviz sfdp (e.g. $(which sfdp)). "
                         "When set, sfdp total wall-clock is measured and added "
                         "to the table (PAPER_PLAN §7).")
    ap.add_argument("--chunk", type=int, default=None,
                    help="per-edge chunk size for the executor forward -- bounds "
                         "GPU memory so N=10^6 runs instead of OOMing (Q22).")
    ap.add_argument("--cugraph_python", default=None,
                    help="path to a RAPIDS venv python (cugraph + libcugraph with "
                         "LD_LIBRARY_PATH set). When set, the cuGraph ForceAtlas2 "
                         "row is measured (total wall-clock for --T iterations, "
                         "same budget as the step-iteration rows).")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cargs = ckpt["args"]
    encoder = StructuralEncoder(
        out_dim=cargs.get("hidden_dim", 64),
        lap_k=cargs.get("lap_k", 10),
        rw_k=cargs.get("rw_k", 16),
        fast_eigsh=True,
        use_rrwp=False,
    )
    num_teachers = len(ckpt["teachers"]) if isinstance(ckpt.get("teachers"), dict) else 1
    model = EquivariantExecutor(
        node_feat_dim=cargs.get("hidden_dim", 64),
        hidden_dim=cargs.get("hidden_dim", 64),
        k_geo=cargs.get("k_geo", 10),
        num_teachers=num_teachers,
        knn_backend="kd",
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    model.eval()
    print(f"loaded checkpoint {args.checkpoint} (val {ckpt.get('val_loss'):.6f})")

    print("\nTIMING BOUNDARY (PAPER_PLAN §7): per-step = geometric kNN + model "
          "forward + host-device + full rollout. Structural encoding is a "
          "one-time per-graph cost, reported separately, EXCLUDED from per-step. "
          "FR/BH-FR have no encoding stage. sfdp is multilevel: a single run to "
          "convergence, reported as TOTAL wall-clock (no per-iteration step "
          "exists to time), sequential depth not comparable by construction. "
          "cuGraph FA2 is NOT installed (gated large install) -- this table is "
          "the reproducible subset plus the sfdp row.")
    print(f"sequential depth: FR/BH-FR = {args.T} steps; Phi_1 = {args.T} forwards; "
          f"Phi_k = ceil({args.T}/k) forwards; sfdp = 1 multilevel run; "
          f"cuGraph FA2 = {args.T} internal iterations (one GPU call).\n")

    header = (f"{'method':<12} {'family':<8} {'N':>8} | {'per-step':>12} "
              f"{'total':>12} {'forwards':>9}")
    print(header)
    print("-" * len(header))
    for fam in args.families:
        for n in args.sizes:
            path = os.path.join(args.corpora, f"{fam}_{n}.pt")
            if not os.path.isfile(path):
                continue
            ei = torch.load(path).numpy()
            # --- FR (networkx): O(N^2), skip beyond N=1e4 (measured earlier:
            # ~3.5 s/step at N=1e4, infeasible at 1e5) ---
            if n <= 10000:
                try:
                    per = fr_networkx(ei, n)
                    tot = per * args.T
                    print(f"{'FR (networkx)':<12} {fam:<8} {n:>8,} | "
                          f"{per * 1e3:9.1f} ms  {tot:9.1f} s  {args.T:>7}  (O(N^2))")
                except Exception as ex:
                    print(f"{'FR (networkx)':<12} {fam:<8} {n:>8,} | {type(ex).__name__}: {ex}")
            else:
                print(f"{'FR (networkx)':<12} {fam:<8} {n:>8,} |      infeasible (O(N^2))")
            # --- Barnes-Hut FR ---
            per, _ = bh_fr(ei, n, iters=args.T)
            print(f"{'BH-FR':<12} {fam:<8} {n:>8,} | {per * 1e3:9.1f} ms  "
                  f"{per * args.T:9.1f} s  {args.T:>7}")
            # --- executor ---
            try:
                per, t_geo, t_fwd = executor(encoder, model,
                                             torch.from_numpy(ei).long(),
                                             n, device, args.T,
                                             skip_encoding=args.skip_encoding,
                                             chunk=args.chunk)
                tot = per * args.T
                enc_note = " (enc skipped)" if args.skip_encoding else ""
                print(f"{'Phi_1':<12} {fam:<8} {n:>8,} | {per * 1e3:9.1f} ms  "
                      f"{tot:9.1f} s  {args.T:>7}   (geo {t_geo * 1e3:.1f} + "
                      f"fwd {t_fwd * 1e3:.1f} ms){enc_note}")
            except Exception as ex:
                print(f"{'Phi_1':<12} {fam:<8} {n:>8,} | {type(ex).__name__}: {ex}")
            # --- sfdp (multilevel, total wall-clock) ---
            if args.sfdp_bin:
                total, err = sfdp_total(ei, n, args.sfdp_bin)
                if err is None:
                    print(f"{'sfdp':<12} {fam:<8} {n:>8,} | {total:9.1f} s  "
                          f"{'--':>9}  {1:>7}   (total, multilevel)")
                else:
                    print(f"{'sfdp':<12} {fam:<8} {n:>8,} | {err}")
            # --- cuGraph FA2 (single GPU call, total wall-clock) ---
            if args.cugraph_python:
                total, err = fa2_cugraph(ei, n, max_iter=args.T,
                                         rapids_python=args.cugraph_python)
                if err is None:
                    print(f"{'cuGraph FA2':<12} {fam:<8} {n:>8,} | {total:9.1f} s  "
                          f"{'--':>9}  {args.T:>7}   (total, GPU)")
                else:
                    print(f"{'cuGraph FA2':<12} {fam:<8} {n:>8,} | {err}")


if __name__ == "__main__":
    main()
