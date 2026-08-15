"""
glide/executor/geometric.py

Geometric neighbourhood construction for the equivariant executor
(PAPER_PLAN.md §4): repulsion is geometrically local, not topologically
local, so the neighbourhood used for it must be rebuilt from the current
positions every step, not read off the graph's static topology.

Backends (Q13, size-extrapolation infra):

  brute  -- O(N^2) cdist + topk. Correct and dependency-free, fine at the
            training scale (N=20-50) and the small eval set. This is the
            default so existing behaviour and the Q4/Q5/Q7 checkpoints are
            untouched.
  kd     -- exact kNN via scipy.spatial.cKDTree, O(N log N). The scalable
            path for N=10^2-10^6. Same neighbours as brute force (verified
            in check_knn_backends), just without the N^2 memory/time blowup.
            CPU-only; positions are moved to numpy for the query and the
            resulting edge_index is returned on the original device. Use
            only when the graph is too big for brute force; the host-device
            transfer is itself part of the timing boundary Q13's harness
            must measure (benchmark.py), so it is not hidden here.
"""

import torch
import numpy as np


def build_knn_graph(pos: torch.Tensor, k: int, backend: str = "brute") -> torch.Tensor:
    """
    Connect every node to its k nearest neighbours by current Euclidean
    distance. Directed: node i's neighbour set need not be symmetric (i may
    be one of j's k nearest without the reverse holding).

    Args:
        pos:     [N, 2] current positions.
        k:       neighbours per node. Must satisfy 0 < k < N.
        backend: "brute" (cdist+topk) or "kd" (scipy cKDTree, scalable).

    Returns:
        edge_index: [2, N*k] in PyG source_to_target convention --
            edge_index[0] = source (the neighbour, u)
            edge_index[1] = target (the node whose neighbourhood this is, v)
    """
    n = pos.shape[0]
    assert 0 < k < n, f"k={k} must be in (0, N={n})"

    if backend == "brute":
        return _knn_brute(pos, k)
    if backend == "kd":
        return _knn_kdtree(pos, k)
    raise ValueError(f"unknown knn backend: {backend!r}")


def _knn_brute(pos: torch.Tensor, k: int) -> torch.Tensor:
    n = pos.shape[0]
    dist = torch.cdist(pos, pos)  # [N, N]
    dist.fill_diagonal_(float("inf"))
    _, nn_idx = torch.topk(dist, k, dim=-1, largest=False)  # [N, k]

    target = torch.arange(n, device=pos.device).unsqueeze(-1).expand(-1, k)  # [N, k]
    edge_index = torch.stack([nn_idx.reshape(-1), target.reshape(-1)], dim=0)
    return edge_index


def _knn_kdtree(pos: torch.Tensor, k: int) -> torch.Tensor:
    """Exact kNN via scipy.spatial.cKDTree. CPU-only; O(N log N)."""
    import scipy.spatial

    n = pos.shape[0]
    pos_np = pos.detach().cpu().numpy().astype(np.float64)
    tree = scipy.spatial.cKDTree(pos_np)
    dist, idx = tree.query(pos_np, k=k + 1)  # +1 includes self
    # query returns shape (N,) if k==1 else (N, k+1); normalise to 2D.
    idx = np.atleast_2d(idx)
    dist = np.atleast_2d(dist)
    nn_idx = idx[:, 1:]  # drop self (always distance 0, first column)

    target = torch.arange(n, device=pos.device).unsqueeze(-1).expand(-1, k)
    nn_idx_t = torch.from_numpy(nn_idx.reshape(-1).astype(np.int64)).to(pos.device)
    edge_index = torch.stack([nn_idx_t, target.reshape(-1)], dim=0)
    return edge_index


def merge_edge_sets(
    edge_index_a: torch.Tensor,
    edge_index_b: torch.Tensor,
    num_nodes: int,
):
    """
    Union two directed edge sets (e.g. topological E and geometric E_geo),
    deduplicated by (source, target), with a boolean membership flag per
    input set. An edge present in both keeps both flags True. This is what
    lets a single readout sum over "everything relevant to v this step"
    (PAPER_PLAN.md §4's dX[v] formula) while still telling phi whether a
    given neighbour is a topological one, a geometric one, or both.

    Returns:
        row, col:   [E] source and target indices of the union
        is_a, is_b: [E] bool membership in edge_index_a / edge_index_b
    """

    def encode(ei):
        return ei[0].to(torch.long) * num_nodes + ei[1].to(torch.long)

    code_a, code_b = encode(edge_index_a), encode(edge_index_b)
    all_codes = torch.cat([code_a, code_b])
    uniq_codes, inverse = torch.unique(all_codes, return_inverse=True)

    is_a = torch.zeros(uniq_codes.numel(), dtype=torch.bool, device=uniq_codes.device)
    is_b = torch.zeros_like(is_a)
    is_a[inverse[: code_a.numel()]] = True
    is_b[inverse[code_a.numel() :]] = True

    row = uniq_codes // num_nodes
    col = uniq_codes % num_nodes
    return row, col, is_a, is_b
