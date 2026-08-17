"""
eval/score_large_layout.py

R1 (rider brief 2026-08-16): sampled stress + neighbourhood-preservation scorer
for layouts on graphs too large for exact all-pairs stress. Exact stress is
O(N^2) and infeasible at N = 10^5-10^6; this scores a layout by sampling node
pairs and reporting a sampling standard error alongside every number, because
a stress figure without one cannot carry a large-graph claim.

Stress convention. Matches eval/metrics.py scale_normalized_stress exactly
(the same scale-normalizing alpha* is used), but the sums are estimated from a
uniform sample of ordered pairs rather than all i<j pairs. d_ij is the graph
distance (BFS from sampled sources, scipy unweighted dijkstra on the CSR
adjacency); e_ij is the Euclidean distance in the layout. Because

    SNS = sum_{i<j} d^-2 (alpha* e - d)^2 ,  alpha* = A / B
    A = sum_{i<j} d^-1 e ,  B = sum_{i<j} d^-2 e^2
    and  d^-2 (alpha e - d)^2 = alpha^2 d^-2 e^2 - 2 alpha d^-1 e + 1,

the quantity collapses to SNS = P' (1 - ma^2 / mb) with P' = n(n-1)/2 and
ma, mb the sample means of d^-1 e and d^-2 e^2. The sampling standard error
is a cluster bootstrap over the BFS sources: pairs sharing a source are
correlated (they share the source's geometry), so the source blocks are the
independent units and a plain sample-variance SE would understate the
uncertainty. The validation check (--compare_exact, run on the N=10^3 corpora
where exact stress is affordable) must show the exact value falling inside
the sampled value plus/minus the reported error before the numbers are
usable.

Neighbourhood preservation at fixed k (default 10): mean over a sample of
nodes of the Jaccard index between the node's graph 1-hop neighbourhood and
its k nearest neighbours in the layout (KD-tree). Reported as mean +/- SE.

Inputs:
    --graph   sparse graph: a torch .pt edge_index [2,E], a Matrix Market
              .mtx, or a two-column edge list (whitespace separated). Node ids
              are renumbered to 0..n-1; self-loops dropped, edges deduped.
    --pos     coordinates: .npy [N,2], .npz (keys pos/layout/coords), or a
              two-column .txt.

Usage:
    .venv/bin/python eval/score_large_layout.py \
        --graph data/corpora/grid_100000.pt \
        --pos eval/results/large_layouts/phi_grid_100000.npy \
        --M 10000000
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_graph(path: str):
    """Load a graph as (edge_index [2,E] int64, n). Node ids renumbered to
    0..n-1; self-loops dropped; duplicate edges removed."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        import torch
        ei = torch.load(path, weights_only=True).numpy()
        if ei.shape[0] == 2:
            rows, cols = ei[0].astype(np.int64), ei[1].astype(np.int64)
        else:
            raise ValueError(f"expected [2,E] edge_index tensor, got {ei.shape}")
    elif ext == ".mtx":
        m = sp.io.mmread(path)
        if sp.issparse(m):
            m = m.tocoo()
            rows, cols = m.row.astype(np.int64), m.col.astype(np.int64)
        else:
            raise ValueError(f".mtx did not parse as sparse: {path}")
    else:  # edge list, two columns
        a = np.loadtxt(path)
        if a.ndim != 2 or a.shape[1] < 2:
            raise ValueError(f"edge list must have >=2 columns: {path}")
        rows, cols = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64)

    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    if rows.size == 0:
        raise ValueError(f"graph has no edges: {path}")
    uniq = np.unique(np.concatenate([rows, cols]))
    remap = {int(u): i for i, u in enumerate(uniq)}
    r = np.array([remap[int(x)] for x in rows], dtype=np.int64)
    c = np.array([remap[int(x)] for x in cols], dtype=np.int64)
    n = uniq.size
    # dedupe (including mirror pairs)
    keys = r.astype(np.int64) * n + c
    keys = np.unique(keys)
    r = keys // n
    c = keys % n
    return np.stack([r, c], axis=0), n


def load_pos(path: str, n: int) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        z = np.load(path)
        for k in ("pos", "layout", "coords", "positions"):
            if k in z:
                pos = z[k]
                break
        else:
            raise ValueError(f"no pos/layout/coords key in {path}")
    elif ext == ".npy":
        pos = np.load(path)
    else:
        pos = np.loadtxt(path)
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"pos must be [N,2], got {pos.shape} from {path}")
    if pos.shape[0] != n:
        raise ValueError(f"pos has {pos.shape[0]} rows but graph has {n} nodes")
    return pos


def build_csr(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    """Unweighted symmetric CSR adjacency, zero diagonal (both directions
    present in edge_index already, so the max-with-transpose is a no-op but
    cheap insurance)."""
    A = sp.coo_matrix(
        (np.ones(edge_index.shape[1], dtype=np.float64),
         (edge_index[0], edge_index[1])),
        shape=(n, n),
    )
    A = A.tocsr()
    A = A.maximum(A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def restrict_to_lcc(edge_index: np.ndarray, pos: np.ndarray, n: int):
    """Restrict graph + layout to the largest connected component. Returns
    (edge_index, pos, n) renumbered to 0..k-1. Needed for disconnected real
    graphs: the sampled stress scales by C(n,2), which assumes every pair is
    connected -- silently scoring a disconnected graph as if its pairs all
    counted would bias the estimate upward."""
    from scipy.sparse.csgraph import connected_components

    A = build_csr(edge_index, n)
    n_comp, labels = connected_components(A, directed=False)
    if n_comp == 1:
        return edge_index, pos, n
    size = np.bincount(labels)
    big = int(np.argmax(size))
    keep = labels == big
    remap = {int(u): i for i, u in enumerate(np.flatnonzero(keep))}
    r = np.array([remap[int(x)] for x in edge_index[0] if keep[int(x)]],
                 dtype=np.int64)
    c = np.array([remap[int(x)] for x in edge_index[1] if keep[int(x)]],
                 dtype=np.int64)
    ei = np.stack([r, c], 0)
    return ei, pos[keep], int(keep.sum())


# ---------------------------------------------------------------------------
# Sampled stress + cluster-bootstrap SE
# ---------------------------------------------------------------------------

def sampled_stress(pos: np.ndarray, A: sp.csr_matrix, M: int, seed: int,
                   n_boot: int = 300, n_sources: int = 50,
                   rng: np.random.Generator = None):
    """Estimate scale-normalized stress by sampling M ordered node pairs
    uniformly, computing d_ij by BFS from sampled sources. Returns
    (stress, se, n_sampled).

    Standard error: pairs drawn from the same BFS source are correlated (they
    share the source's geometry), so a plain sample-variance SE understates the
    uncertainty. The SE is therefore a cluster bootstrap over the S sources:
    resample the S source-blocks with replacement and take the std of the
    resulting stress estimates. Validated against the exact metrics.py stress
    on the N=10^3 corpora (--compare_exact), where the exact value falls inside
    the reported error."""
    n = pos.shape[0]
    if n < 2:
        return float("nan"), float("nan"), 0
    if rng is None:
        rng = np.random.default_rng(seed)

    S = int(n_sources)          # number of BFS sources (one bootstrap cluster each)
    K = max(1, int(np.ceil(M / S)))
    K = min(K, n - 1)           # can't have more targets than other nodes

    clusters = []               # each cluster: (sum_a, sum_b, count)
    sources = rng.integers(0, n, size=S)
    for s in sources:
        # unweighted single-source shortest paths == BFS, in C
        d = dijkstra(A, directed=False, indices=int(s), unweighted=True)
        finite = np.isfinite(d)
        finite[int(s)] = False
        if not finite.any():
            continue
        idx = np.flatnonzero(finite)
        # sample K targets uniformly among nodes reachable from s
        targets = idx[rng.integers(0, idx.size, size=K)]
        dt = d[targets]
        delta = pos[targets] - pos[int(s)]
        e = np.linalg.norm(delta, axis=1)
        a = e / dt
        b = (e * e) / (dt * dt)
        clusters.append((float(a.sum()), float(b.sum()), int(K)))

    if not clusters:
        return float("nan"), float("nan"), 0
    S_used = len(clusters)
    ca = np.array([c[0] for c in clusters])
    cb = np.array([c[1] for c in clusters])
    cc = np.array([c[2] for c in clusters])
    n_sampled = int(cc.sum())

    def stress_from(ca_i, cb_i, cc_i):
        ma, mb = ca_i.sum() / cc_i.sum(), cb_i.sum() / cc_i.sum()
        return (n * (n - 1) / 2.0) * (1.0 - (ma * ma) / mb)

    stress = float(stress_from(ca, cb, cc))
    boot = np.empty(n_boot)
    for r in range(n_boot):
        pick = rng.integers(0, S_used, size=S_used)
        boot[r] = stress_from(ca[pick], cb[pick], cc[pick])
    se = float(boot.std(ddof=1))
    return stress, se, n_sampled


# ---------------------------------------------------------------------------
# Neighbourhood preservation at fixed k
# ---------------------------------------------------------------------------

def sampled_np(pos: np.ndarray, A: sp.csr_matrix, k: int = 10,
               max_nodes: int = 100_000, seed: int = 0,
               rng: np.random.Generator = None):
    """Mean Jaccard between each node's graph 1-hop neighbourhood and its k
    nearest neighbours in the layout, over a sample of nodes. Returns
    (mean, se, n_nodes)."""
    n = pos.shape[0]
    if n < 2 or k < 1:
        return float("nan"), float("nan"), 0
    if rng is None:
        rng = np.random.default_rng(seed)

    n_samp = min(n, max_nodes)
    sample = rng.choice(n, size=n_samp, replace=False)
    tree = cKDTree(pos)
    nbrs = tree.query(pos[sample], k=min(k + 1, n), workers=-1)[1]
    nbrs = np.asarray(nbrs)
    if nbrs.ndim == 1:           # k+1 == n -> shape quirk
        nbrs = nbrs[:, None]
    layout = nbrs[:, 1:k + 1]    # drop the node itself (distance 0)

    indptr, indices = A.indptr, A.indices
    scores = np.empty(n_samp)
    for j, i in enumerate(sample):
        g = indices[indptr[i]:indptr[i + 1]]
        if g.size == 0:
            scores[j] = float("nan")
            continue
        gset = set(g.tolist())
        lset = set(layout[j].tolist())
        inter = len(gset & lset)
        union = len(gset | lset)
        scores[j] = inter / union if union else 0.0
    valid = scores[np.isfinite(scores)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(valid.mean())
    se = float(valid.std(ddof=1) / np.sqrt(valid.size)) if valid.size > 1 else 0.0
    return mean, se, int(valid.size)


# ---------------------------------------------------------------------------
# Exact reference (validation only, N ~ 10^3)
# ---------------------------------------------------------------------------

def exact_metrics(pos: np.ndarray, edge_index: np.ndarray, n: int, k: int = 10):
    """Exact scale-normalized stress (eval/metrics.py) + exact NP@k over all
    nodes, for validating the sampler on graphs where O(N^2) is affordable."""
    from eval.metrics import scale_normalized_stress

    A = build_csr(edge_index, n)
    D = sp.csgraph.shortest_path(A, directed=False, unweighted=True)
    stress, alpha = scale_normalized_stress(pos, D)

    tree = cKDTree(pos)
    nbrs = tree.query(pos, k=min(k + 1, n), workers=-1)[1]
    nbrs = np.asarray(nbrs)
    if nbrs.ndim == 1:
        nbrs = nbrs[:, None]
    layout = nbrs[:, 1:k + 1]
    indptr, indices = A.indptr, A.indices
    scores = []
    for i in range(n):
        g = indices[indptr[i]:indptr[i + 1]]
        if g.size == 0:
            continue
        inter = len(set(g.tolist()) & set(layout[i].tolist()))
        union = len(set(g.tolist()) | set(layout[i].tolist()))
        scores.append(inter / union if union else 0.0)
    return float(stress), alpha, float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--M", type=int, default=None,
                    help="sampled pairs. Default scales with N: "
                         "10^7 at N=10^6, 10^6 at 10^5, etc.")
    ap.add_argument("--np_k", type=int, default=10)
    ap.add_argument("--np_nodes", type=int, default=100_000,
                    help="cap on nodes sampled for neighbourhood preservation")
    ap.add_argument("--sources", type=int, default=50,
                    help="number of BFS sources / bootstrap clusters. Larger "
                         "cuts the standard error at ~1/sqrt(S) cost in BFS "
                         "time; the default is the validation value.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare_exact", action="store_true",
                    help="also compute exact metrics.py stress and exact NP@k "
                         "(affordable only at N ~ 10^3); prints both. This is "
                         "the R1.1 validation gate.")
    ap.add_argument("--lcc", action="store_true",
                    help="restrict scoring to the largest connected component "
                         "(drop other components and renumber). Required for "
                         "disconnected real graphs: the sampled estimator "
                         "scales by the pair count of the graph it is given, "
                         "so scoring a disconnected graph as-is would bias "
                         "stress upward.")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    edge_index, n = load_graph(args.graph)
    pos = load_pos(args.pos, n)
    if args.lcc:
        edge_index, pos, n = restrict_to_lcc(edge_index, pos, n)
    rng = np.random.default_rng(args.seed)

    if args.M is None:
        M = int(min(10_000_000, max(10_000, 10 * n)))
    else:
        M = int(args.M)

    stress, stress_se, ns_pairs = sampled_stress(
        pos, build_csr(edge_index, n), M, args.seed,
        n_sources=args.sources, rng=rng)
    np_mean, np_se, np_nodes = sampled_np(
        pos, build_csr(edge_index, n), args.np_k, args.np_nodes, args.seed, rng)
    n_pairs = n * (n - 1) // 2

    row = {
        "graph": args.graph,
        "pos": args.pos,
        "n": n,
        "M": M,
        "sampled_pairs": ns_pairs,
        "stress": stress,
        "stress_se": stress_se,
        "stress_per_pair": stress / n_pairs if n_pairs else None,
        "stress_per_pair_se": stress_se / n_pairs if n_pairs else None,
        "np_k": args.np_k,
        "np_nodes_used": np_nodes,
        "np_mean": np_mean,
        "np_se": np_se,
    }

    if args.compare_exact:
        ex_stress, ex_alpha, ex_np = exact_metrics(pos, edge_index, n, args.np_k)
        row["exact_stress"] = ex_stress
        row["exact_alpha"] = ex_alpha
        row["exact_np"] = ex_np
        row["stress_diff_sigma"] = (
            (stress - ex_stress) / stress_se if stress_se > 0 else None)
        row["np_diff_sigma"] = (
            (np_mean - ex_np) / np_se if np_se > 0 else None)

    print(json.dumps(row, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(row, f, indent=2)


if __name__ == "__main__":
    main()
