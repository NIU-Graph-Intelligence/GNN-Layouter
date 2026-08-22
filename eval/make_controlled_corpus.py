"""
eval/make_controlled_corpus.py  --  controlled large-N corpus.

Generates N=10^5 graphs with 4 families (grid, rgg, scale_free/BA, er) and
3 independent graph seeds (101, 202, 303).  After edge construction and any
connectivity repair, applies an independent seeded random permutation to node
labels so that node order is decoupled from graph topology and initial geometry.

Key differences from data/corpora/ (the legacy §11 corpus):
  - Seeds: 101, 202, 303
  - Permutation: applied AFTER all topology decisions
  - RGG avg_deg=6: ensures supercritical regime (percolation threshold ~4.51 for 2D RGG)
  - RGG and ER: use LCC, NOT a Hamiltonian chain
  - Grid and BA: inherently connected, no repair needed

LCC policy (for RGG and ER):
  - If LCC >= 80% of requested N: use LCC (relabel to 0..k-1)
  - If LCC < 80%: add minimum-spanning edges with seeded endpoints (independent of
    final node labels), then re-extract LCC.

Output: data/corpora/r1_v2/  (does not touch data/corpora/*.pt)
Provenance: data/corpora/r1_v2/provenance.json

Usage:
    .venv/bin/python eval/make_controlled_corpus.py [--n 100000]
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

OUT_DIR = ROOT / "data/corpora/r1_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_DEFAULT  = 100_000
GRAPH_SEEDS = [101, 202, 303]
FAMILIES    = ["grid", "rgg", "scale_free", "er"]
PERM_SEED_BASE = 7000    # permutation seed = PERM_SEED_BASE + graph_seed


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def build_grid(n):
    """2D grid (always connected). Bidirectional edge list."""
    cols = int(np.ceil(np.sqrt(n)))
    idx  = np.arange(n, dtype=np.int64)
    not_row_end = (idx % cols) != cols - 1
    right = idx[not_row_end & (idx + 1 < n)]
    down  = idx[idx + cols < n]
    fwd   = np.concatenate([np.stack([right, right+1], 0),
                             np.stack([down,  down+cols], 0)], axis=1)
    rev   = fwd[[1, 0]]
    ei    = np.concatenate([fwd, rev], axis=1)
    return ei[0], ei[1]


def build_rgg(n, seed, avg_deg=6.0):
    """RGG in [0,1]^2, avg_deg=6 (supercritical: threshold ~4.51).
    Returns unique directed edges (bidirectional).  No Hamiltonian chain."""
    rng  = np.random.default_rng(seed)
    pos  = rng.random((n, 2))
    r    = np.sqrt(avg_deg / (np.pi * n))
    cell = 1.0 / max(r, 1e-9)
    cells = (pos * cell).astype(np.int64)
    buckets = {}
    for i in range(n):
        key = (int(cells[i, 0]), int(cells[i, 1]))
        buckets.setdefault(key, []).append(i)
    s_list, d_list = [], []
    for i in range(n):
        cx, cy = int(cells[i, 0]), int(cells[i, 1])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((cx+dx, cy+dy), []):
                    if j <= i: continue
                    if np.linalg.norm(pos[i] - pos[j]) < r:
                        s_list.append(i); d_list.append(j)
    s = np.array(s_list, np.int64)
    d = np.array(d_list, np.int64)
    rows = np.concatenate([s, d])
    cols = np.concatenate([d, s])
    return rows, cols


def build_ba(n, seed, m=3):
    """Barabasi-Albert (always connected). Bidirectional."""
    import networkx as nx
    G   = nx.barabasi_albert_graph(n, m, seed=seed)
    arr = np.array(G.edges(), dtype=np.int64)
    fwd = arr.T
    rev = fwd[[1, 0]]
    ei  = np.concatenate([fwd, rev], axis=1)
    return ei[0], ei[1]


def build_er(n, seed, avg_deg=4.0):
    """ER G(n,p) p=avg_deg/(n-1) via geometric sampling.
    Returns unique bidirectional edges.  No Hamiltonian chain."""
    rng = np.random.default_rng(seed)
    p   = avg_deg / (n - 1)
    s_list, d_list = [], []
    u, v = 0, -1
    while True:
        # Geometric skip: next edge (u,v) with u < v
        skip = rng.geometric(p)
        v   += skip
        while v >= n:
            u  += 1
            v   = v - (n - u)
            if u >= n - 1:
                break
        if u >= n - 1:
            break
        s_list.append(u); d_list.append(v)
    s    = np.array(s_list, np.int64)
    d    = np.array(d_list, np.int64)
    rows = np.concatenate([s, d])
    cols = np.concatenate([d, s])
    return rows, cols


# ---------------------------------------------------------------------------
# LCC extraction
# ---------------------------------------------------------------------------

def lcc_edge_index(rows, cols, n):
    """Return (new_rows, new_cols, new_n, lcc_frac)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    nc, labels = connected_components(A, directed=False)
    if nc == 1:
        return rows, cols, n, 1.0
    sizes = np.bincount(labels)
    big   = int(np.argmax(sizes))
    keep  = labels == big
    new_n = int(keep.sum())
    lcc_f = new_n / n
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(new_n, dtype=np.int64)
    mask  = keep[rows] & keep[cols]
    return remap[rows[mask]], remap[cols[mask]], new_n, lcc_f


def connect_then_lcc(rows, cols, n, repair_seed):
    """Add n_comp-1 edges to connect all components, then re-extract LCC."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    rng = np.random.default_rng(repair_seed)
    A   = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    nc, labels = connected_components(A, directed=False)
    if nc == 1:
        return lcc_edge_index(rows, cols, n)
    members = [np.flatnonzero(labels == c) for c in range(nc)]
    extra_s, extra_d = [], []
    for c in range(1, nc):
        u = int(members[0][rng.integers(0, len(members[0]))])
        v = int(members[c][rng.integers(0, len(members[c]))])
        extra_s += [u, v];  extra_d += [v, u]
    new_rows = np.concatenate([rows, extra_s])
    new_cols = np.concatenate([cols, extra_d])
    return lcc_edge_index(new_rows, new_cols, n)


# ---------------------------------------------------------------------------
# Node permutation
# ---------------------------------------------------------------------------

def permute_nodes(rows, cols, n, perm_seed):
    rng  = np.random.default_rng(perm_seed)
    perm = rng.permutation(n)
    return perm[rows], perm[cols], perm


# ---------------------------------------------------------------------------
# One graph
# ---------------------------------------------------------------------------

def generate_one(family, graph_seed, n_requested, perm_seed,
                 lcc_threshold=0.80):
    t0 = time.perf_counter()
    repair_applied = False

    if family == "grid":
        rows, cols = build_grid(n_requested)
        n, lcc_frac = n_requested, 1.0

    elif family == "rgg":
        rows, cols = build_rgg(n_requested, graph_seed, avg_deg=6.0)
        rows, cols, n, lcc_frac = lcc_edge_index(rows, cols, n_requested)
        if lcc_frac < lcc_threshold:
            rseed = graph_seed * 1000 + 9999
            rows, cols, n, lcc_frac = connect_then_lcc(rows, cols, n, rseed)
            repair_applied = True

    elif family == "scale_free":
        rows, cols = build_ba(n_requested, graph_seed)
        n, lcc_frac = n_requested, 1.0

    elif family == "er":
        rows, cols = build_er(n_requested, graph_seed, avg_deg=4.0)
        rows, cols, n, lcc_frac = lcc_edge_index(rows, cols, n_requested)
        if lcc_frac < lcc_threshold:
            rseed = graph_seed * 1000 + 9999
            rows, cols, n, lcc_frac = connect_then_lcc(rows, cols, n, rseed)
            repair_applied = True
    else:
        raise ValueError(f"Unknown family: {family}")

    # Permute node labels AFTER all topology decisions
    rows, cols, perm = permute_nodes(rows, cols, n, perm_seed)
    dt = time.perf_counter() - t0

    ei   = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    prov = {
        "family": family, "graph_seed": graph_seed,
        "permutation_seed": perm_seed,
        "n_requested": n_requested, "n_actual": int(n),
        "lcc_fraction": float(lcc_frac),
        "edge_count": int(ei.shape[1]),
        "undirected_edge_count": int(ei.shape[1]) // 2,
        "repair_applied": repair_applied,
        "rgg_avg_deg": 6.0 if family == "rgg" else None,
        "er_avg_deg": 4.0 if family == "er" else None,
        "gen_time_s": float(dt),
    }
    return ei, prov


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    args = ap.parse_args()
    n = args.n
    print(f"controlled corpus  N={n:,}  seeds={GRAPH_SEEDS}")
    print(f"RGG avg_deg=6 (supercritical), ER avg_deg=4 (geometric sampling)")
    print(f"Output: {OUT_DIR}\n")

    all_prov = []
    for family in FAMILIES:
        for gseed in GRAPH_SEEDS:
            perm_seed = PERM_SEED_BASE + gseed
            fname     = f"{family}_{n}_seed{gseed}.pt"
            out_path  = OUT_DIR / fname
            if out_path.exists():
                print(f"  [skip] {fname}")
                continue
            print(f"  {family:>10} seed={gseed} ... ", end="", flush=True)
            ei, prov = generate_one(family, gseed, n, perm_seed)
            torch.save(ei, out_path)
            all_prov.append({"file": fname, **prov})
            print(f"n={prov['n_actual']:,}  LCC={prov['lcc_fraction']:.4f}  "
                  f"E_undir={prov['undirected_edge_count']:,}  "
                  f"{prov['gen_time_s']:.1f}s")

    prov_path = OUT_DIR / "provenance.json"
    existing  = {}
    if prov_path.exists():
        with open(prov_path) as f:
            existing = {p["file"]: p for p in json.load(f).get("graphs", [])}
    for p in all_prov:
        existing[p["file"]] = p
    with open(prov_path, "w") as f:
        json.dump({"description": "controlled corpus provenance",
                   "n": n, "graph_seeds": GRAPH_SEEDS, "families": FAMILIES,
                   "perm_seed_base": PERM_SEED_BASE,
                   "lcc_policy": "LCC for rgg/er; connect if < 80%; permute after",
                   "rgg_avg_deg": 6.0, "er_avg_deg": 4.0,
                   "graphs": list(existing.values())}, f, indent=2)
    print(f"\nSaved {len(all_prov)} graphs. Provenance -> {prov_path}")


if __name__ == "__main__":
    main()
