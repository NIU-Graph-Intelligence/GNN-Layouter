"""
glide/executor/structural.py

Structural embeddings for the equivariant executor (PAPER_PLAN.md §4, Q3):
computed once per graph from topology alone (Laplacian PE through SignNet,
random-walk features) and cached for the whole rollout. Only the geometric
neighbourhood, state features (pos, tau), and the readout are recomputed per
step -- see EquivariantExecutor in model.py.

Reuses glide/stage1/gst/encodings.py (raw eigenvector/RRWP computation) and
glide/stage1/gst/sign_net.py (sign-invariant processing) rather than forking
them: same math, already validated by Stage 1 training, no reason to
duplicate it. Neither needed changes for this use.
"""

import torch
import torch.nn as nn

from glide.stage1.gst.encodings import compute_all_encodings
from glide.stage1.gst.sign_net import SignNet


def _laplacian_eigvecs_fast(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    """Plain-Lanczos Laplacian eigenvectors (Q13 large-N path).

    Same spectrum target as stage1's shift-invert mode (the k smallest
    non-trivial eigenpairs) but computed with `eigsh(which='SM')` and no
    shift, which avoids the expensive per-iteration linear solve that makes
    shift-invert crawl at N>=10^5. Verified to agree with the shift-invert
    result to float noise and to split two-community graphs correctly (the
    Fiedler sign test) at N=10^5. Float64 sparse -- scipy's requirement.
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from torch_geometric.utils import to_scipy_sparse_matrix

    adj = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=num_nodes).astype(np.float32)
    deg = np.array(adj.sum(axis=1)).flatten()
    L = (sp.diags(deg) - adj).astype(np.float64)

    n_req = min(k + 1, num_nodes - 1)
    eigvals_all, eigvecs_all = spla.eigsh(L, k=n_req, which="SM")
    order = np.argsort(eigvals_all)
    eigvecs_all = eigvecs_all[:, order]

    # drop the trivial lambda_0 = 0 constant eigenvector, keep next k
    eigvecs = eigvecs_all[:, 1 : k + 1]
    if eigvecs.shape[1] < k:
        pad = np.zeros((num_nodes, k - eigvecs.shape[1]), dtype=eigvecs.dtype)
        eigvecs = np.concatenate([eigvecs, pad], axis=1)
    return torch.from_numpy(eigvecs.astype(np.float32))


def compute_rrwp_diag(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    """RRWP diagonal entries (rrwp_node) via sparse matrix powers -- the
    executor only ever uses the node-level diagonal, never the dense [N,N,C]
    edge tensor, so this avoids the O(N^2) dense matrix the stage1
    compute_rrwp builds. Diagonal of M^t for t=1..k, M = D^{-1}A
    row-normalised, matching stage1's definition.

    IMPORTANT (Q13/Q14 finding, 2026-08-13): this is NOT scalable past
    N~10^4. Random-walk probabilities mix fast on well-connected graphs --
    measured: at N=10^4 the sparse power M^t reaches 100% density by t=8
    (a 10000x10000 dense float32 matrix = 400 MB, and the matmul cost
    balloons correspondingly), so the loop below is ~36 s at N=10^4 and
    effectively infeasible at N=10^5. The stage1 dense version has the same
    wall. This is a MATH limit of random-walk PEs, not an implementation
    artifact -- replacing it (bounded-walk sampling for the diagonal, or a
    different positional encoding) changes the paper's §4 structural
    encodings and is Lei's call, flagged as the Q14 blocker in
    the experiment queue. For benchmark timing at N>10^4 use
    `--skip_encoding` (random node_feat) rather than paying this wall.
    """
    import numpy as np
    import scipy.sparse as sp
    from torch_geometric.utils import to_scipy_sparse_matrix

    adj = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=num_nodes).astype(np.float32)
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    M = sp.diags(deg_inv) @ adj  # row-normalised, [N, N] sparse

    diags = np.zeros((num_nodes, k), dtype=np.float32)
    M_pow = M
    for t in range(k):
        d = M_pow.diagonal()
        diags[:, t] = np.asarray(d).reshape(-1).astype(np.float32)
        if t < k - 1:
            M_pow = M_pow @ M
    return torch.from_numpy(diags)


class StructuralEncoder(nn.Module):
    """
    Topology -> per-node feature vector. Call once per graph; the result is
    the `node_feat` argument EquivariantExecutor expects at every step of a
    rollout on that graph. Depends only on `edge_index` and `num_nodes` --
    never on positions -- so it cannot leak coordinate information into a
    quantity that is supposed to be rigid-motion invariant (see model.py's
    equivariance argument, which relies on that).
    """

    def __init__(
        self,
        out_dim: int = 64,
        lap_k: int = 10,
        rw_k: int = 16,
        sign_net_hidden: int = 64,
        fast_eigsh: bool = False,
        use_rrwp: bool = True,
    ):
        super().__init__()
        self.lap_k = lap_k
        self.rw_k = rw_k
        self.fast_eigsh = fast_eigsh
        self.use_rrwp = use_rrwp

        self.sign_net = SignNet(k=lap_k, hidden_dim=sign_net_hidden, out_dim=out_dim)
        if use_rrwp:
            self.rrwp_proj = nn.Linear(rw_k, out_dim)
            self.combine = nn.Sequential(
                nn.Linear(2 * out_dim, out_dim),
                nn.SiLU(),
            )
        else:
            # LapPE-only (Q14 structural-encoding decision, 2026-08-13): the
            # RRWP branch is not created at all, so there is no wasted
            # parameter and no possibility of a large-N path silently
            # depending on it. SignNet output is already [N, out_dim]; the
            # light head keeps a trainable projection layer.
            self.combine = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.SiLU(),
            )

    def forward(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """General path: compute raw eigenvectors/RRWP from scratch. Use for
        any graph that doesn't already have them cached (new graphs at
        eval/large-N time -- Q13 onward).

        `fast_eigsh=True` (Q13 benchmark / large-N eval only): uses a plain
        Lanczos `eigsh(which='SM')` for the Laplacian spectrum instead of the
        shift-invert mode in glide/stage1/gst/encodings.py. Verified identical
        results (SM-no-shift vs shift-invert agree to float noise on the same
        small graphs) and ~15x faster at N=10^5 (7.2s vs >120s). The stage1
        frozen path is left untouched -- this flag is a property of the
        executor's own StructuralEncoder, not a change to shared stage1 code.

        `use_rrwp=False`: LapPE-only. RRWP is never computed, so this path is
        scalable to arbitrary N (eigsh is the only cost, ~7s at N=10^5). This
        is the Q14-proposed large-N encoding; see EXPERIMENT_QUEUE.md.
        """
        if self.fast_eigsh:
            eigvecs = _laplacian_eigvecs_fast(edge_index, num_nodes, self.lap_k)
        else:
            from glide.stage1.gst.encodings import compute_all_encodings

            enc = compute_all_encodings(edge_index, num_nodes, lap_k=self.lap_k, rw_k=self.rw_k)
            if self.use_rrwp:
                return self.encode_precomputed(enc["eigvecs"], enc["rrwp_node"])
            return self.encode_precomputed(enc["eigvecs"], None)
        if self.use_rrwp:
            rrwp_node = compute_rrwp_diag(edge_index, num_nodes, self.rw_k)
            return self.encode_precomputed(eigvecs, rrwp_node)
        return self.encode_precomputed(eigvecs, None)

    def encode_precomputed(self, eigvecs: torch.Tensor, rrwp_node: torch.Tensor = None) -> torch.Tensor:
        """Fast path: `comm_5k_v2_with_encodings.pt` already stores raw
        eigenvectors (`lap_pe`) and RRWP (`rrwp_node`) per graph
        (glide/stage1/preprocess_encodings.py) -- reuse them instead of
        recomputing an eigendecomposition per training step. With
        `use_rrwp=False`, `rrwp_node` is ignored and only LapPE is used."""
        lap_pe = self.sign_net(eigvecs)       # [N, out_dim]
        if not self.use_rrwp:
            return self.combine(lap_pe)       # [N, out_dim]
        rw_feat = self.rrwp_proj(rrwp_node)   # [N, out_dim]
        return self.combine(torch.cat([lap_pe, rw_feat], dim=-1))  # [N, out_dim]
