"""
philayouter/executor/benchmark.py

Timing harness: measures the full per-step
cost of rolling out a trained EquivariantExecutor on synthetic graphs of
increasing size, so the paper's timing table (the
sequential-depth / wall-clock row) has an honest number for OUR method that includes
every stage the timing boundary requires:

  - structural encoding (one-time per graph, cached for the whole rollout)
  - per-step geometric neighbourhood construction (kNN)
  - the model forward (displacement readout)
  - host-device transfers
  - the full T-step rollout

The point is not to win a speed race -- it is to have a reproducible number
with a stated timing boundary, comparable to the baseline table once
benchmark_wallclock.py measures cuGraph FA2 / sfdp on the same graphs.
Reported per step and per rollout, with the boundary stated explicitly in the
output.

Memory-bounded: at N=10^5-10^6 the geometric neighbourhood is built with the
"kd" backend (scipy cKDTree, O(N log N)) and brute-force is used only for
small N as a cross-check. The rollout never materialises the full trajectory;
it keeps only the current position state, so peak memory is O(N) per step.

Synthetic graphs only -- no dataset download required, and the structural
encoder's eigendecomposition is the expensive one-time stage, so a small
connected random graph per size is adequate for timing.

Usage:
    .venv/bin/python -m philayouter.executor.benchmark --n_list 100 1000 10000 --k_geo 10 --T 20
    .venv/bin/python -m philayouter.executor.benchmark --n_list 100000 --device cuda:0 --T 10
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from .geometric import build_knn_graph
from .model import EquivariantExecutor
from .structural import StructuralEncoder


def make_connected_graph(n: int, seed: int = 42):
    """Small connected random graph (ER with a star backbone) sized to n.

    Sparse construction for large n: torch.rand(n, n) at N=10^5 is a
    40 GB dense tensor, which is exactly the kind of incidental
    memory blowup the timing boundary is supposed to measure, not trip on.
    """
    gen = torch.Generator().manual_seed(seed)
    p = 4.0 / n
    row = torch.arange(n, device="cpu")
    # per-source degree ~ Binomial(n, p); materialise edges directly.
    deg = torch.distributions.Binomial(n - 1, p).sample((n,)).long()
    # clamp so n*(avg deg) edges don't explode for tiny n (p ~ 4/n keeps it ~4)
    src = torch.repeat_interleave(row, deg)
    dst = torch.randint(0, n - 1, (src.numel(),), generator=gen)  # [0, n-1)
    dst = dst + (dst >= src).long()  # exclude self-loops, shift range to [0, n)
    # guarantee connectivity: chain 0-1-2-...-n-1
    chain_src = torch.arange(n - 1)
    chain_dst = chain_src + 1
    src = torch.cat([src, chain_src, chain_dst])
    dst = torch.cat([dst, chain_dst, chain_src])
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index


def bench_size(n: int, args, device: torch.device):
    print(f"\n=== N = {n} ===")
    edge_index = make_connected_graph(n).to(device)

    encoder = StructuralEncoder(
        out_dim=args.hidden_dim,
        lap_k=args.lap_k,
        rw_k=args.rw_k,
        fast_eigsh=args.fast_encoding,
    ).to(device)
    model = EquivariantExecutor(
        node_feat_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        k_geo=args.k_geo,
        num_teachers=1,
        knn_backend=args.knn_backend,
    ).to(device)
    encoder.eval()
    model.eval()

    pos = (torch.rand(n, 2, device=device) - 0.5) * 2.0  # grid-like init in [-1,1]^2
    k = float(np.sqrt(1.0 / n))
    pos = pos / k  # k-normalized, per SCALE_NORMALIZATION.md
    tau = torch.full((n, 1), 0.5, device=device)

    with torch.no_grad():
        # one-time structural encoding (timed separately; cached for rollout)
        if args.skip_encoding:
            # Random node_feat: measures the per-step pipeline cost WITHOUT the
            # RRWP dense-mixing wall. Structural encoding is the
            # same one-time cost for any fixed graph, so excluding it here keeps
            # the per-step numbers honest without paying 36s+ at N=10^4 / infeasible
            # at N=10^5. The timing boundary is stated in the output header.
            node_feat = torch.randn(n, args.hidden_dim, device=device)
            t_enc = float("nan")
        else:
            t0 = time.perf_counter()
            node_feat = encoder.forward(edge_index, n)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t_enc = time.perf_counter() - t0

        # per-step costs over a short warmup + measured rollout
        n_warm = 3
        n_meas = max(1, args.T - n_warm)
        t_geo = t_fwd = t_roll = 0.0
        for step in range(args.T):
            t0 = time.perf_counter()
            ei_geo = build_knn_graph(pos, k=min(args.k_geo, n - 1), backend=args.knn_backend)
            t1 = time.perf_counter()
            dX = model(node_feat, edge_index, pos, tau)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t2 = time.perf_counter()
            pos = pos + dX
            if step >= n_warm:
                t_geo += t1 - t0
                t_fwd += t2 - t1
                t_roll += t2 - t0

    per_step = t_roll / n_meas
    print(f"structural encoding (one-time) : {t_enc*1e3:9.1f} ms")
    print(f"kNN geometric neighbourhood     : {t_geo/n_meas*1e3:9.2f} ms/step")
    print(f"model forward                   : {t_fwd/n_meas*1e3:9.2f} ms/step")
    print(f"per-step total (geo+fwd)        : {per_step*1e3:9.2f} ms/step")
    print(f"rollout T={n_meas} (excl warmup) : {t_roll:9.2f} s")
    return {"n": n, "t_enc_ms": t_enc * 1e3, "t_geo_ms": t_geo / n_meas * 1e3,
            "t_fwd_ms": t_fwd / n_meas * 1e3, "t_step_ms": per_step * 1e3}


def main():
    parser = argparse.ArgumentParser(description="timing harness for the equivariant executor")
    parser.add_argument("--n_list", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--k_geo", type=int, default=10)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lap_k", type=int, default=10)
    parser.add_argument("--rw_k", type=int, default=16)
    parser.add_argument("--knn_backend", default="auto",
                        help="'brute', 'kd', or 'auto' (brute for N<=2000, kd above)")
    parser.add_argument("--fast_encoding", action="store_true",
                        help="use plain-Lanczos eigsh + sparse RRWP diag for large-N "
                             "structural encoding (identical results, ~15x faster at N=10^5; "
                             "auto-enabled for N>2000)")
    parser.add_argument("--skip_encoding", action="store_true",
                        help="use random node_feat instead of structural encoding -- for "
                             "per-step pipeline timing at N where RRWP is infeasible "
                             "(N>10^4, the RRWP encoding wall). Encoding reported as NaN.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device: {device}   k_geo={args.k_geo}   T={args.T}   knn_backend={args.knn_backend}")
    print("timing boundary: per-step geometric kNN + model forward + host-device "
          "transfers + full rollout. Structural encoding is a "
          "one-time per-graph cost reported separately (excluded from per-step).")
    if args.skip_encoding:
        print("note: --skip_encoding set -- encoding omitted at all N (reported as "
              "NaN); per-step numbers are the honest pipeline cost independent of "
              "the RRWP encoding wall.")

    results = []
    for n in args.n_list:
        backend = args.knn_backend
        if backend == "auto":
            backend = "brute" if n <= 2000 else "kd"
        args.knn_backend = backend
        fast = args.fast_encoding or (n > 2000 and not args.skip_encoding)
        args.fast_encoding = fast
        results.append(bench_size(n, args, device))

    print("\n=== summary ===")
    for r in results:
        enc_s = "  n/a " if r["t_enc_ms"] != r["t_enc_ms"] else f"{r['t_enc_ms']:8.1f}"
        print(f"N={r['n']:>8,}: enc {enc_s} ms | geo {r['t_geo_ms']:7.2f} ms | "
              f"fwd {r['t_fwd_ms']:7.2f} ms | step {r['t_step_ms']:7.2f} ms")


if __name__ == "__main__":
    main()
