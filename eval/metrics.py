"""
eval/metrics.py

Shared metric implementations for the evaluation section. Every model and every
baseline is scored through this module.

Implemented here:

  procrustes_mse        fidelity to a reference layout, O(2)+scale convention
                        (the similarity transform DeepDrawing and GND publish)
  rollout_error         per-step fidelity across the T-step rollout
  residual_fr_force     how far a layout is from FR equilibrium, no
                        correspondence or alignment required

Procrustes MSE is reported in the normalized frame, matching how the models are
trained and scored. Residual force MUST be computed in the raw frame: the FR
force law is not scale-invariant (rescaling by s sends repulsion k^2/d to
s*k^2/d but attraction d^2/k to d^2/(s^2*k)), and k = sqrt(1/n) is a raw-frame
quantity.
"""

from typing import Optional, Tuple

import numpy as np


# Procrustes

def procrustes_align(
    pred: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, float]:

    pm, tm = pred.mean(axis=0), target.mean(axis=0)
    P, T = pred - pm, target - tm

    U, S, Vt = np.linalg.svd(T.T @ P)
    det = float(np.linalg.det(U @ Vt))
    R = U @ Vt

    # minimise ||T - s P R^T||^2  ->  s = tr(T^T P R^T) / ||P||_F^2, and
    # tr(T^T P R^T) collapses to the sum of the singular values.
    den = float(np.sum(P ** 2))
    s = float(S.sum() / den) if den > 1e-12 else 1.0

    return s * (P @ R.T) + tm, det


def procrustes_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Mean squared error over all N*2 entries after similarity alignment.
    The reduction is the mean over nodes AND coordinates.
    """
    aligned, _ = procrustes_align(pred, target)
    return float(np.mean((aligned - target) ** 2))


# Rollout

def rollout_error(
    pred_traj: np.ndarray,          # [T, N, 2] model positions, step 1..T
    true_traj: np.ndarray,          # [T, N, 2] FR positions,    step 1..T
    aligned: bool = True,
) -> np.ndarray:
    """
    Per-step error across the rollout. Returns [T].

    INDEXING. pred_traj[t] must correspond to true_traj[t], both being the state
    AFTER t+1 updates. The trajectory stored as y_traj[:, t, :] is `pos_after` of
    FR step t, and predicted_positions[t] is the state after t+1 learned updates,
    so they line up at the same index. Using t+1 shifts every point on the curve
    by one step; that mistake has been made in this codebase before.

    aligned=True   Procrustes-align each step independently. Comparable to the
                   endpoint Procrustes MSE.
    aligned=False  raw MSE, no alignment. Both trajectories start from the same
                   P_0 and therefore share a frame, so no rotational ambiguity
                   exists to remove -- this is the honest measure of drift, and
                   it is exactly what L_traj optimises.

    Report both: aligned hides orientation drift, raw includes it.
    """
    if pred_traj.shape != true_traj.shape:
        raise ValueError(
            f"shape mismatch {pred_traj.shape} vs {true_traj.shape} -- pred and "
            f"true must cover the same steps with the same node ordering"
        )

    T = pred_traj.shape[0]
    out = np.empty(T)
    for t in range(T):
        if aligned:
            out[t] = procrustes_mse(pred_traj[t], true_traj[t])
        else:
            out[t] = float(np.mean((pred_traj[t] - true_traj[t]) ** 2))
    return out


# ---------------------------------------------------------------------------
# Residual FR force
# ---------------------------------------------------------------------------

def fr_forces(
    pos: np.ndarray,        # [N, 2] RAW-frame coordinates
    A: np.ndarray,          # [N, N] unweighted symmetric adjacency, zero diagonal
    k: float,               # FR optimal distance, sqrt(1/N) in the raw frame
    eps: float = 0.01,      # minimum distance clip, matching the generator
) -> np.ndarray:
    """
    Net FR force on every node. Returns [N, 2].

        F_i = sum_{j in N(i)} (d_ij^2 / k) * u_ji        attraction along edges
            + sum_{j != i}    (k^2 / d_ij) * u_ij        repulsion, all pairs

    with u_ij the unit vector from j toward i.

    Mirrors data/generate_fr_iterations.py:fr_single_step exactly, which computes
    the same thing as

        force_mag = k*k/d**2 - A*d/k
        displacement = sum_j delta_ij * force_mag_ij

    Because delta_ij = p_i - p_j carries magnitude d, multiplying through turns
    k^2/d^2 into k^2/d and A*d/k into A*d^2/k, giving the expression above.

    The diagonal is masked explicitly rather than left to cancel. The generator
    gets away with delta_ii = 0 zeroing a k^2/eps^2 term that would otherwise be
    enormous; anything computing magnitudes directly would not.
    """
    n = pos.shape[0]
    delta = pos[:, None, :] - pos[None, :, :]            # [N, N, 2], p_i - p_j
    dist = np.linalg.norm(delta, axis=-1)
    np.clip(dist, eps, None, out=dist)

    force_mag = (k * k) / dist ** 2 - A * dist / k       # [N, N]
    np.fill_diagonal(force_mag, 0.0)                     # explicit, see docstring

    return np.einsum("ijc,ij->ic", delta, force_mag)     # [N, 2]


def residual_force(
    pos: np.ndarray,
    A: np.ndarray,
    k: float,
    eps: float = 0.01,
) -> float:
    """
    Mean per-node force magnitude, Phi(P). Normalizes across the 20-50 node range.

    Invariant to translation and rotation (the force field rotates with the
    layout), so no alignment is needed and no node correspondence is assumed --
    which is what makes it a genuine complement to Procrustes MSE.

    NOT invariant to scale, so `pos` must be in the raw frame. See module
    docstring.
    """
    return float(np.linalg.norm(fr_forces(pos, A, k, eps), axis=1).mean())


def fr_force_components(
    pos: np.ndarray,
    A: np.ndarray,
    k: float,
    eps: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Split Phi into its attractive and repulsive halves, plus the minimum pairwise
    distance. Returns (Phi_attraction, Phi_repulsion, min_pair_distance).

    Useful because the two halves fail in opposite directions and a single Phi
    cannot tell them apart:

        repulsion  k^2 / d      blows up when nodes sit TOO CLOSE
        attraction d^2 / k      blows up when linked nodes sit TOO FAR APART

    A layout that is under-spread or clumped therefore shows a large repulsive
    residual, while one that is over-stretched shows a large attractive one.

    Note the repulsive term is very sensitive to the closest pair: at the eps
    floor of 0.01 a single overlapping pair contributes k^2/0.01, which for
    N = 30 is ~3.3. A handful of clumped nodes can dominate the mean, so
    min_pair_distance is returned alongside to make that visible.
    """
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)

    off = ~np.eye(pos.shape[0], dtype=bool)
    min_d = float(dist[off].min()) if off.any() else 0.0

    np.clip(dist, eps, None, out=dist)

    rep_mag = (k * k) / dist ** 2
    att_mag = -A * dist / k
    np.fill_diagonal(rep_mag, 0.0)
    np.fill_diagonal(att_mag, 0.0)

    F_rep = np.einsum("ijc,ij->ic", delta, rep_mag)
    F_att = np.einsum("ijc,ij->ic", delta, att_mag)

    return (float(np.linalg.norm(F_att, axis=1).mean()),
            float(np.linalg.norm(F_rep, axis=1).mean()),
            min_d)


def residual_force_gap(
    pred: np.ndarray,
    target: np.ndarray,
    A: np.ndarray,
    k: float,
    eps: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Returns (|Phi(pred) - Phi(target)|, Phi(pred), Phi(target)), all raw-frame.

    FR terminates on a fixed iteration budget rather than a force criterion, so
    Phi(target) is nonzero and a layout with LOWER residual force than the target
    has departed from it rather than improved on it. The gap is therefore the
    metric and both endpoints are returned so the direction of any deviation
    stays visible.
    """
    a = residual_force(pred, A, k, eps)
    b = residual_force(target, A, k, eps)
    return abs(a - b), a, b


# ---------------------------------------------------------------------------
# Quality metrics
#
# These measure the drawing itself rather than agreement with a reference, so
# they apply to every method including ones that do not target FR at all. FR is
# reported as a reference row, not as a competitor.
#
# They answer a different question from the fidelity group. Residual force asks
# "is this an FR fixed point"; these ask "is this a readable drawing". A layout
# can be far from FR's fixed point and still read well, which is exactly the
# distinction needed to judge whether an under-converged layout is a real defect.
# ---------------------------------------------------------------------------

def scale_normalized_stress(
    pos: np.ndarray,
    D: np.ndarray,
) -> Tuple[float, float]:
    """
    Stress after solving for the stress-minimising uniform scale.

        alpha* = sum_{i<j} d_ij^-1 |p_i-p_j|  /  sum_{i<j} d_ij^-2 |p_i-p_j|^2
        SNS    = sum_{i<j} d_ij^-2 (alpha* |p_i-p_j| - d_ij)^2

    Raw stress depends on the absolute scale of a layout, so methods that emit
    different coordinate ranges cannot be compared on it. Solving for alpha*
    first removes that confound.

    Returns (SNS, alpha*). alpha* is reported too, and is not a nuisance
    parameter here: it is a direct measure of global under- or over-spread.
    alpha* > 1 means the layout must be EXPANDED to best match the graph
    distances, i.e. it is too compressed. Since scale-normalized stress
    deliberately removes exactly the effect the known under-spread is about,
    reporting SNS without alpha* would hide it.

    D is the matrix of graph-theoretic (shortest-path) distances.
    """
    n = pos.shape[0]
    iu = np.triu_indices(n, k=1)
    d = D[iu].astype(np.float64)
    e = np.linalg.norm(pos[iu[0]] - pos[iu[1]], axis=1)

    finite = np.isfinite(d) & (d > 0)      # drop disconnected pairs
    d, e = d[finite], e[finite]
    if d.size == 0:
        return float("nan"), float("nan")

    w = d ** -2.0
    denom = float((w * e ** 2).sum())
    alpha = float((d ** -1.0 * e).sum() / denom) if denom > 0 else 1.0
    return float((w * (alpha * e - d) ** 2).sum()), alpha


def neighborhood_preservation(pos: np.ndarray, A: np.ndarray) -> float:
    """
    Mean Jaccard index between each node's graph neighbourhood and its k nearest
    neighbours in the drawing, with k set to that node's degree.

    Matching the cardinalities (k_i = deg(i)) matters: with a fixed global k the
    index would be bounded away from 1 for every node whose degree differs from
    it, so the metric could never reach its own maximum.
    """
    n = pos.shape[0]
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)          # a node is not its own neighbour

    scores = []
    for i in range(n):
        deg = int(A[i].sum())
        if deg == 0:
            continue
        graph_nbrs = set(np.flatnonzero(A[i] > 0).tolist())
        layout_nbrs = set(np.argsort(dist[i])[:deg].tolist())
        inter = len(graph_nbrs & layout_nbrs)
        union = len(graph_nbrs | layout_nbrs)
        scores.append(inter / union if union else 0.0)

    return float(np.mean(scores)) if scores else float("nan")


def edge_crossings(pos: np.ndarray, edges: np.ndarray) -> int:
    """
    Number of crossing edge pairs. Edges sharing an endpoint are skipped, since
    they meet by construction rather than by crossing.

    Uses the standard orientation test: segments AB and CD cross when A and B
    fall on opposite sides of CD and C and D fall on opposite sides of AB.
    Collinear-overlap cases are not counted; they have measure zero for
    continuous coordinates.
    """
    def orient(p, q, r):
        return ((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])
                - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))

    m = edges.shape[0]
    if m < 2:
        return 0
    i, j = np.triu_indices(m, k=1)
    a, b = edges[i, 0], edges[i, 1]
    c, d = edges[j, 0], edges[j, 1]

    share = (a == c) | (a == d) | (b == c) | (b == d)
    keep = ~share
    if not keep.any():
        return 0
    a, b, c, d = a[keep], b[keep], c[keep], d[keep]

    A_, B_, C_, D_ = pos[a], pos[b], pos[c], pos[d]
    d1, d2 = orient(C_, D_, A_), orient(C_, D_, B_)
    d3, d4 = orient(A_, B_, C_), orient(A_, B_, D_)
    return int((((d1 > 0) != (d2 > 0)) & ((d3 > 0) != (d4 > 0))).sum())


def min_edge_angle(pos: np.ndarray, A: np.ndarray) -> float:
    """
    Mean over nodes of the smallest angle between any two edges incident on that
    node, in degrees. Higher is more readable; the ideal for a degree-k node is
    360/k.
    """
    n = pos.shape[0]
    vals = []
    for i in range(n):
        nbrs = np.flatnonzero(A[i] > 0)
        if nbrs.size < 2:
            continue
        v = pos[nbrs] - pos[i]
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1e-12
        u = v / norm
        cos = np.clip(u @ u.T, -1.0, 1.0)
        np.fill_diagonal(cos, -1.0)         # exclude self-pairs from the max
        vals.append(np.degrees(np.arccos(cos.max())))
    return float(np.mean(vals)) if vals else float("nan")


def community_silhouette(pos: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette coefficient of the layout coordinates under the planted community
    assignment. LFR supplies ground-truth labels, so separation is measured
    directly rather than inferred by clustering the drawing.

    Ranges [-1, 1]; higher means communities occupy more distinct regions.
    Returns NaN when fewer than two communities are present.
    """
    uniq = np.unique(labels)
    if uniq.size < 2 or pos.shape[0] <= uniq.size:
        return float("nan")

    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    scores = []
    for i in range(pos.shape[0]):
        own = labels == labels[i]
        own[i] = False
        if not own.any():
            continue
        a = dist[i, own].mean()
        b = min(dist[i, labels == c].mean() for c in uniq if c != labels[i])
        denom = max(a, b)
        if denom > 0:
            scores.append((b - a) / denom)
    return float(np.mean(scores)) if scores else float("nan")


def shortest_path_matrix(A: np.ndarray) -> np.ndarray:
    """
    All-pairs shortest-path distances by repeated BFS on the dense adjacency.
    Unreachable pairs are inf and are dropped by scale_normalized_stress.
    """
    n = A.shape[0]
    D = np.full((n, n), np.inf)
    adj = [np.flatnonzero(A[i] > 0) for i in range(n)]
    for s in range(n):
        D[s, s] = 0.0
        frontier, depth = [s], 0
        seen = np.zeros(n, dtype=bool)
        seen[s] = True
        while frontier:
            depth += 1
            nxt = []
            for u in frontier:
                for w in adj[u]:
                    if not seen[w]:
                        seen[w] = True
                        D[s, w] = depth
                        nxt.append(w)
            frontier = nxt
    return D


# ---------------------------------------------------------------------------
# Frame conversion
# ---------------------------------------------------------------------------

def denormalize(pos: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    """Normalized frame -> raw FR frame. Required before any force computation."""
    return pos * y_std + y_mean


def adjacency_from_edge_index(edge_index: np.ndarray, n: int) -> np.ndarray:
    """Dense unweighted symmetric adjacency with zero diagonal."""
    A = np.zeros((n, n), dtype=np.float64)
    A[edge_index[0], edge_index[1]] = 1.0
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A
