"""
philayouter/executor/check_knn_backends.py

Verifies that the scalable "kd" kNN backend
(scipy.cKDTree, O(N log N)) produces the SAME neighbourhood as the brute-force
"brute" backend (cdist+topk, O(N^2)) before trusting it at large N. The two
must agree edge-for-edge for every node, because training/eval results are
meaningless if the repulsion neighbourhood silently differs between backends.

Synthetic random graphs only -- no dataset needed, and small N is fine
(agreement is what is checked; performance is benchmark.py's job).

Run: .venv/bin/python -m philayouter.executor.check_knn_backends
"""

import numpy as np
import torch

from .geometric import build_knn_graph


def check(seed: int, n: int, k: int, device: torch.device) -> float:
    torch.manual_seed(seed)
    pos = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed)).to(device)

    ei_brute = build_knn_graph(pos, k, backend="brute")
    ei_kd = build_knn_graph(pos, k, backend="kd")

    assert ei_brute.shape == ei_kd.shape, (ei_brute.shape, ei_kd.shape)

    # Compare as edge sets (row/col) -- ordering may differ between backends
    # only if ties break differently; both return col = target in the same
    # (node, neighbour-slot) layout, so compare directly.
    mismatch = int((ei_brute != ei_kd).any().item())
    return mismatch


def main():
    device = torch.device("cpu")
    total_mismatch = 0
    for n in (20, 50, 120):
        for k in (5, 10, 20):
            if k >= n:
                continue
            for seed in range(5):
                mm = check(seed, n, k, device)
                total_mismatch += mm
                tag = "PASS" if mm == 0 else "FAIL"
                print(f"n={n:3d} k={k:2d} seed={seed}: edge sets match ({tag})")

    print(f"\ntotal mismatching entries: {total_mismatch}  {'PASS' if total_mismatch == 0 else 'FAIL'}")
    if total_mismatch:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
