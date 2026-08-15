"""
philayouter/executor/generate_corpus.py

Q12/Q14 (the experiment queue, Lei-decided 2026-08-14): size-extrapolation
corpus for the central claim. Lei accepted the recommendation that a diverse
multi-source set is required, with a synthetic fallback if remote downloads
are blocked (they are on this machine -- both SuiteSparse and Network
Repository direct links return 404/empty here). This script is that fallback.

Synthetic families, all sparse-constructed (no dense rand(n,n) -- the machine
OOM'd on such tests 2026-08-13, recorded in EXPERIMENT_QUEUE.md):

  grid          -- 2D grid (sqrt(N) x sqrt(N)): regular lattice, local
  rgg           -- random geometric graph (connect nodes within radius r):
                    proximity structure, the natural large-N analogue of the
                    community family's geometry
  scale_free    -- Barabasi-Albert (m=3): heavy-tailed degree
  er            -- Erdos-Renyi p=4/N with a Hamiltonian chain backbone for
                    connectivity (same construction benchmark.py uses)

Each family is generated at N = 10^3, 10^4, 10^5, 10^6 as one edge_index
tensor per (family, size), saved as sparse torch files under data/corpora/.
Isolating SIZE per family is the clean way to measure size generalization
(the same topology type at increasing N) -- the actual eval (Q14) then also
crosses FAMILY (train on community 20-50, evaluate at 10^3..10^6 of all four).

Memory-bounded by construction: edge lists only; nothing O(N^2) is materialised.

Usage:
    .venv/bin/python -m philayouter.executor.generate_corpus --sizes 1000 10000 100000 1000000
"""

import argparse
import os
import time

import numpy as np
import torch


def grid_graph(n: int) -> torch.Tensor:
    """2D grid: rows x cols covering n nodes, edges right/down only then mirrored.
    Vectorised: idx - 1 and idx - cols are right/down predecessors; an undirected
    edge exists between (i, i+1) iff same row, and (i, i+cols) iff i+cols < n."""
    cols = int(np.ceil(np.sqrt(n)))
    idx = np.arange(n, dtype=np.int64)
    # right edges: i -> i+1, valid iff (i+1) not at a row boundary
    not_row_end = (idx % cols) != cols - 1
    right = idx[not_row_end & (idx + 1 < n)]
    e_right = np.stack([right, right + 1], 0)
    # down edges: i -> i+cols, valid iff in bounds
    down = idx[idx + cols < n]
    e_down = np.stack([down, down + cols], 0)
    fwd = np.concatenate([e_right, e_down], axis=1)
    rev = fwd[[1, 0]]
    return torch.tensor(np.concatenate([fwd, rev], axis=1), dtype=torch.long)


def rgg_graph(n: int, seed: int = 42, avg_deg: float = 4.0) -> torch.Tensor:
    """Random geometric graph: nodes uniform in [0,1]^2, edges within radius r
    chosen so expected degree ~ avg_deg. Sparse: sample pairwise distances only
    for a cell-hash of nearby points."""
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 2))
    r = np.sqrt(avg_deg / (np.pi * n))  # expected degree = pi n r^2
    # cell hashing to avoid O(N^2)
    cell = 1.0 / max(r, 1e-9)
    cells = (pos * cell).astype(np.int64)
    buckets = {}
    for i in range(n):
        key = (cells[i, 0], cells[i, 1])
        buckets.setdefault(key, []).append(i)
    src, dst = [], []
    for i in range(n):
        cx, cy = cells[i]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((cx + dx, cy + dy), []):
                    if j <= i:
                        continue
                    if np.linalg.norm(pos[i] - pos[j]) < r:
                        src.append(i)
                        dst.append(j)
    # ensure connectivity via Hamiltonian chain
    chain = np.arange(n)
    src = np.concatenate([src, chain[:-1], chain[1:]])
    dst = np.concatenate([dst, chain[1:], chain[:-1]])
    return torch.tensor(np.stack([src, dst], 0), dtype=torch.long)


def scale_free_graph(n: int, seed: int = 42, m: int = 3) -> torch.Tensor:
    """Barabasi-Albert, sparse construction."""
    import networkx as nx

    G = nx.barabasi_albert_graph(n, m, seed=seed)
    edges = np.array(G.edges(), dtype=np.int64)  # [E, 2]
    fwd = edges.T  # [2, E]
    rev = fwd[[1, 0]]
    return torch.tensor(np.concatenate([fwd, rev], axis=1), dtype=torch.long)


def er_graph(n: int, seed: int = 42, p: float = 4.0) -> torch.Tensor:
    """Erdos-Renyi with Hamiltonian backbone (benchmark.py's construction)."""
    gen = torch.Generator().manual_seed(seed)
    deg = torch.distributions.Binomial(n - 1, p / n).sample((n,)).long()
    row = torch.arange(n)
    src = torch.repeat_interleave(row, deg)
    dst = torch.randint(0, n - 1, (src.numel(),), generator=gen)
    dst = dst + (dst >= src).long()
    chain = torch.arange(n - 1)
    src = torch.cat([src, chain, chain + 1])
    dst = torch.cat([dst, chain + 1, chain])
    return torch.stack([src, dst], dim=0)


FAMILIES = {
    "grid": grid_graph,
    "rgg": lambda n: rgg_graph(n),
    "scale_free": lambda n: scale_free_graph(n),
    "er": lambda n: er_graph(n),
}


def main():
    parser = argparse.ArgumentParser(description="Q12/Q14: size-extrapolation corpus")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    parser.add_argument("--out_dir", default="data/corpora")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for fam_name, fn in FAMILIES.items():
        for n in args.sizes:
            t0 = time.time()
            ei = fn(n)
            dt = time.time() - t0
            path = os.path.join(args.out_dir, f"{fam_name}_{n}.pt")
            torch.save(ei, path)
            print(f"{fam_name:10s} N={n:>8,}: E={ei.shape[1]:>12,}  {dt:6.1f}s  -> {path}", flush=True)


if __name__ == "__main__":
    main()
