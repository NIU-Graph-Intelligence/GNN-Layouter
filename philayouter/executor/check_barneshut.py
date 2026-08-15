"""
philayouter/executor/check_barneshut.py

Validates philayouter.executor.barneshut against the reference O(N^2) FR step:

1. Repulsion term vs brute force `k^2 * sum_j (p_i-p_j)/|p_i-p_j|^2` over
   random positions at several N and theta values -- the approximation error
   must stay small and shrink as theta -> 0.
2. Full-step equivalence on a small grid graph: Barnes-Hut FR (theta=0.7)
   must track NetworkX FR closely over a 50-step rollout.

Usage:
    .venv/bin/python -m philayouter.executor.check_barneshut
"""

import time

import networkx as nx
import numpy as np

from .barneshut import fr_repulsion_barnes_hut, fr_step

RNG = np.random.default_rng(0)


def brute_repulsion(pos: np.ndarray, k: float, eps: float = 0.01) -> np.ndarray:
    """Reference O(N^2) repulsion: k^2 * sum_j (p_i-p_j)/|p_i-p_j|^2."""
    delta = pos[:, None, :] - pos[None, :, :]          # [N,N,2]
    d = np.linalg.norm(delta, axis=-1)
    np.clip(d, eps, None, out=d)
    c = (float(k) ** 2) / d ** 2                        # [N,N]
    np.fill_diagonal(c, 0.0)
    return np.einsum("ijk,ij->ik", delta, c)


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


def test_repulsion_tolerance():
    print("== repulsion term vs brute force ==")
    for n in (200, 1000):
        pos = RNG.normal(size=(n, 2))
        k = np.sqrt(1.0 / n)
        ref = brute_repulsion(pos, k)
        for theta in (0.0, 0.3, 0.7, 1.0):
            got = fr_repulsion_barnes_hut(pos, k, theta=theta)
            # relative error on per-node displacement magnitude
            rel = np.linalg.norm(got - ref, axis=1) / (
                np.linalg.norm(ref, axis=1) + 1e-12)
            print(f"  N={n:>4} theta={theta:.1f}: "
                  f"median rel err {np.median(rel):.4f}  "
                  f"max rel err {rel.max():.4f}")


def test_jit_vs_ref():
    print("\n== JIT backend vs pure-Python reference tree ==")
    for n in (200, 1000):
        pos = RNG.normal(size=(n, 2))
        k = np.sqrt(1.0 / n)
        a = fr_repulsion_barnes_hut(pos, k, theta=0.7, backend="jit")
        b = fr_repulsion_barnes_hut(pos, k, theta=0.7, backend="ref")
        diff = np.linalg.norm(a - b, axis=1) / (
            np.linalg.norm(b, axis=1) + 1e-12)
        print(f"  N={n:>4}: max rel diff JIT vs ref {diff.max():.6f}")


def test_full_step_rollout():
    print("\n== full-step rollout vs NetworkX FR (theta=0.7) ==")
    n = 60
    G = nx.connected_watts_strogatz_graph(n, 4, 0.3, seed=42)
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    edges = np.array(list(G.edges())).T if list(G.edges()) else np.zeros((2, 0), int)
    # NetworkX keeps edges sorted by node; make edge array independent of order
    edge_list = np.array(sorted(tuple(sorted(e)) for e in G.edges())).T

    pos = grid_init(n)
    k = np.sqrt(1.0 / n)
    coord_span = pos.max(axis=0) - pos.min(axis=0)
    t0 = max(float(np.max(coord_span) * 0.1), 1e-4)
    dt = t0 / (50 + 1)

    pos_bh = pos.copy()
    temp = t0
    temps = []
    for step in range(50):
        temps.append(temp)
        pos_bh = fr_step(pos_bh, edge_list, k, temp, theta=0.7)
        temp = max(temp - dt, 1e-6)

    init_pos = {node: pos[i].tolist() for i, node in enumerate(nodes)}
    pos_nx = nx.spring_layout(G, pos=init_pos, iterations=50, seed=42,
                              scale=None, threshold=0)
    nx_final = np.array([pos_nx[node] for node in nodes], dtype=np.float64)

    # compare final layouts after similarity alignment (translation + scale)
    def align(src, tgt):
        s = (src - src.mean(0)) / (np.linalg.norm(src - src.mean(0)) + 1e-12)
        t = (tgt - tgt.mean(0)) / (np.linalg.norm(tgt - tgt.mean(0)) + 1e-12)
        return s, t

    s_bh, s_nx = align(pos_bh, nx_final)
    rmse = np.sqrt(np.mean((s_bh - s_nx) ** 2))
    print(f"  aligned RMSE of final layout vs NetworkX FR: {rmse:.4f}")
    print(f"  (two independent implementations of the same algorithm; "
          f"expect a small number, not zero)")


def test_scaling():
    print("\n== rough wall-clock scaling (CPU, 1 step each) ==")
    for n in (1000, 10000, 100000):
        pos = RNG.normal(size=(n, 2))
        k = np.sqrt(1.0 / n)
        t0 = time.perf_counter()
        fr_repulsion_barnes_hut(pos, k, theta=0.7)
        dt = time.perf_counter() - t0
        print(f"  BH repulsion N={n:>7}: {dt*1e3:8.1f} ms/step")


if __name__ == "__main__":
    test_repulsion_tolerance()
    test_jit_vs_ref()
    test_full_step_rollout()
    test_scaling()
