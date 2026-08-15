"""
gst/encodings.py

Precomputed graph encodings for the Graph Soft-Tokenizer (GST).
All encodings are derived purely from graph topology — no learned parameters here.

Encodings produced:
  - LapPE:        top-k Laplacian eigenvectors  [N, k]   (global PE)
  - eigenvalues:  top-k Laplacian eigenvalues   [k]      (global SE, graph-level)
  - RRWP_node:    diagonal of M^1..M^k          [N, k]   (local SE per node)
  - RRWP_edge:    M^m[i,j] for each edge        [E, k]   (relative PE per edge)
  - fiedler_SE:   phi2[i] * phi2[j] per edge    [E, 1]   (relative SE per edge)

All encodings are precomputed once by preprocess_encodings.py and stored
directly in the PyG Data objects. All output shapes are [N, k] or [E, k] —
no [N, N, k] tensors, so PyG batching works correctly with no issues.
"""

import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, degree
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Laplacian PE + Global SE (eigenvalues)
# ---------------------------------------------------------------------------

def compute_laplacian_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k Laplacian eigenvectors (LapPE) and eigenvalues (Global SE).

    The graph Laplacian L = D - A. We take the k eigenvectors corresponding
    to the k smallest non-zero eigenvalues. These give each node a globally
    consistent position signal — nodes close in the graph get similar values.

    Sign ambiguity: eigenvectors are sign-ambiguous (phi and -phi are both valid).
    We return raw eigenvectors here. SignNet (sign_net.py) handles the ambiguity
    during the forward pass.

    Args:
        edge_index: [2, E] edge indices (undirected)
        num_nodes:  N
        k:          number of eigenvectors to keep (default 10)

    Returns:
        eigvecs:    [N, k] float32 — raw Laplacian eigenvectors
        eigvals:    [k]    float32 — corresponding eigenvalues
    """
    # Build scipy sparse Laplacian
    # to_scipy_sparse_matrix returns the adjacency matrix
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    adj = adj.astype(np.float32)

    # Degree matrix
    deg_arr = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(deg_arr)

    # Laplacian
    L = D - adj
    L = L.astype(np.float64)  # eigs needs float64

    # We need the eigenvectors of the k SMALLEST NON-ZERO eigenvalues.
    # eigvecs[:, 0] must come out as the Fiedler vector — compute_fiedler_se()
    # depends on that, and so does the "where am I in the graph" signal.
    if num_nodes <= 1000:
        # Dense eigh is exact, deterministic, and faster than shift-invert
        # at these sizes (graphs here are N=20-100). Returns ascending order.
        eigvals_all, eigvecs_all = np.linalg.eigh(L.toarray())
    else:
        n_req = min(k + 1, num_nodes - 1)
        # CAREFUL: when `sigma` is passed, `which` selects on the SHIFTED
        # spectrum 1/(lambda - sigma), not on lambda. So 'LM' = eigenvalues
        # CLOSEST to sigma = the smallest ones, which is what we want.
        # ('SM' here would return the LARGEST eigenvalues — the long-standing
        # bug this replaces, which made lap_pe the highest-frequency modes
        # and left fiedler_se with no community signal.)
        # sigma is slightly negative so that L - sigma*I stays non-singular
        # (lambda_0 = 0 is always an exact eigenvalue of a graph Laplacian).
        eigvals_all, eigvecs_all = spla.eigsh(L, k=n_req, which='LM', sigma=-1e-3)
        order = np.argsort(eigvals_all)
        eigvals_all = eigvals_all[order]
        eigvecs_all = eigvecs_all[:, order]

    # Drop the trivial lambda_0 = 0 constant eigenvector, keep the next k
    eigvals = eigvals_all[1:k + 1]
    eigvecs = eigvecs_all[:, 1:k + 1]

    # Pad to exactly k if we got fewer eigenvectors
    if eigvecs.shape[1] < k:
        pad_cols = k - eigvecs.shape[1]
        eigvecs = np.concatenate(
            [eigvecs, np.zeros((num_nodes, pad_cols), dtype=eigvecs.dtype)],
            axis=1
        )
        eigvals = np.concatenate(
            [eigvals, np.zeros(pad_cols, dtype=eigvals.dtype)]
        )

    eigvecs = torch.from_numpy(eigvecs.astype(np.float32))  # [N, k]
    eigvals = torch.from_numpy(eigvals.astype(np.float32))  # [k]

    return eigvecs, eigvals


# RRWP — Random Walk Positional Encoding
def compute_rrwp(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RRWP node and edge features.

    The random walk matrix M = D^{-1} A.
    - RRWP_node[i]    = [M^1[i,i], M^2[i,i], ..., M^k[i,i]]  diagonal entries
    - RRWP_edge[i,j]  = [M^1[i,j], M^2[i,j], ..., M^k[i,j]]  for each edge

    No sign ambiguity — return probabilities are always non-negative.

    Args:
        edge_index: [2, E] edge indices
        num_nodes:  N
        k:          number of random walk steps (default 16)

    Returns:
        rrwp_node: [N, k]   float32 — diagonal entries (local SE)
        rrwp_edge: [E, k]   float32 — edge entries (relative PE)
    """
    device = edge_index.device

    # Build dense adjacency matrix
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32, device=device)
    row, col = edge_index[0], edge_index[1]
    A[row, col] = 1.0
    A[col, row] = 1.0  # ensure undirected

    # Degree-normalised random walk matrix M = D^{-1} A
    deg = A.sum(dim=1)  # [N]
    deg_inv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
    M = deg_inv.unsqueeze(1) * A  # [N, N]  — row-normalised

    # Accumulate powers
    rrwp_node = torch.zeros(num_nodes, k, dtype=torch.float32, device=device)
    rrwp_edge = torch.zeros(edge_index.shape[1], k, dtype=torch.float32, device=device)

    M_pow = M.clone()  # M^1
    for step in range(k):
        # Node features: diagonal entries M^(step+1)[i,i]
        rrwp_node[:, step] = M_pow.diagonal()

        # Edge features: M^(step+1)[i,j] for each edge (i,j)
        rrwp_edge[:, step] = M_pow[row, col]

        # Advance: M^(step+2) = M^(step+1) @ M
        if step < k - 1:
            M_pow = M_pow @ M

    return rrwp_node, rrwp_edge


# ---------------------------------------------------------------------------
# Fiedler Relative SE
# ---------------------------------------------------------------------------

def compute_fiedler_se(
    edge_index: torch.Tensor,
    eigvecs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Fiedler-vector-based relative structural encoding per edge.

    fiedler_SE[i,j] = phi2[i] * phi2[j]

    Positive value → nodes i and j are in the same community.
    Negative value → nodes i and j are in different communities.

    The Fiedler vector (phi2) is the eigenvector of the second-smallest
    Laplacian eigenvalue — it partitions the graph into two communities
    via its sign. This is the cheapest and most direct community signal.

    Args:
        edge_index: [2, E]
        eigvecs:    [N, k] — column 0 is the Fiedler vector (phi2 in 0-indexed)
                             (eigvecs[:, 0] corresponds to smallest non-zero eigenvalue)

    Returns:
        fiedler_se: [E, 1] float32
    """
    # eigvecs[:, 0] = first non-trivial eigenvector = Fiedler vector
    phi2 = eigvecs[:, 0]  # [N]

    row, col = edge_index[0], edge_index[1]
    fiedler_se = (phi2[row] * phi2[col]).unsqueeze(1)  # [E, 1]

    return fiedler_se.float()


# ---------------------------------------------------------------------------
# Convenience: compute all encodings for one graph
# ---------------------------------------------------------------------------

def compute_all_encodings(
    edge_index: torch.Tensor,
    num_nodes: int,
    lap_k: int = 10,
    rw_k: int = 16,
) -> dict:
    """
    Compute all GST encodings for a single graph.

    Returns a dict with keys:
        'eigvecs'    : [N, lap_k]  — raw Laplacian eigenvectors (LapPE input to SignNet)
        'eigvals'    : [lap_k]     — Laplacian eigenvalues (Global SE)
        'rrwp_node'  : [N, rw_k]  — RRWP diagonal (local SE node features)
        'rrwp_edge'  : [E, rw_k]  — RRWP off-diagonal (relative PE edge features)
        'fiedler_se' : [E, 1]     — Fiedler product (relative SE edge features)

    All tensors are float32. Device matches edge_index.device.
    """
    eigvecs, eigvals = compute_laplacian_pe(edge_index, num_nodes, k=lap_k)

    # Move eigvecs to same device as edge_index for Fiedler computation
    device = edge_index.device
    eigvecs = eigvecs.to(device)
    eigvals = eigvals.to(device)

    rrwp_node, rrwp_edge = compute_rrwp(edge_index, num_nodes, k=rw_k)

    fiedler_se = compute_fiedler_se(edge_index, eigvecs)

    return {
        'eigvecs':    eigvecs,    # [N, lap_k]
        'eigvals':    eigvals,    # [lap_k]
        'rrwp_node':  rrwp_node,  # [N, rw_k]
        'rrwp_edge':  rrwp_edge,  # [E, rw_k]
        'fiedler_se': fiedler_se, # [E, 1]
    }