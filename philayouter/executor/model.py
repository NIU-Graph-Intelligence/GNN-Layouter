"""
philayouter/executor/model.py

Equivariant executor for a force-directed layout algorithm -- one learned
step of X_{t+1} = Phi(G, X_t, tau_t).

Every term of the FR update maps onto one component:

    attraction (topological)        -> _EdgeContextMP over E
    repulsion  (geometric, in X_t)  -> _EdgeContextMP over a kNN graph
                                        rebuilt from the current positions
                                        every step (philayouter/executor/geometric.py)
    temperature tau_t decaying      -> multiplicative scalar conditioning
    displacement                    -> equivariant readout:

        dX[v] = tau_t * sum_u ((x_v-x_u)/||x_v-x_u||) *
                phi(h_v, h_u, ||x_v-x_u||, is_topo, is_geo, tau_t, c)

Every argument of phi is invariant to rigid motion (distance, the two
membership flags, tau_t, the teacher-conditioning embedding c) and the
prefactor (x_v-x_u)/||x_v-x_u|| is equivariant, so the model is equivariant
by construction at every parameter setting -- not something training has to
learn. Verified numerically in check_equivariance.py.

Node features (`node_feat`) must be structural encodings only (Laplacian PE,
random-walk features, degree, ...) -- never raw coordinates. Raw coordinates
only enter through relative-position terms (distances, unit vectors), which
is what makes the equivariance guarantee hold; if a future edit threads `pos`
into `h` directly, that guarantee silently breaks.

`pos` and `tau` are assumed already normalized by k = sqrt(1/N) at the
caller, per the scale-normalization convention -- this module has no notion of
graph size and must not be given one (that's the point).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .geometric import build_knn_graph, merge_edge_sets

EPS = 1e-8


class _EdgeContextMP(MessagePassing):
    """
    Aggregates neighbour features into h, conditioned on an invariant
    per-edge scalar (distance in the current drawing). Used for both the
    topological (attraction) and geometric (repulsion) neighbourhoods --
    same layer shape, separate weights, per the component table above.
    """

    def __init__(self, hidden_dim: int):
        super().__init__(aggr="add")
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor,
                chunk: int = None) -> torch.Tensor:
        n = h.shape[0]
        if chunk is not None:
            # Chunked aggregation: process edges in blocks and accumulate with
            # index_add, so peak memory is bounded by (chunk * message dim)
            # instead of (E * message dim). Needed for N ~ 10^6 where E = N*k_geo
            # materializes 10^7+ per-edge messages (the large-N memory wall).
            out = torch.zeros(n, self.edge_mlp[0].out_features, device=h.device)
            row, col = edge_index
            for s in range(0, col.shape[0], chunk):
                e = min(s + chunk, col.shape[0])
                hi, hj = h[col[s:e]], h[row[s:e]]
                dist = (pos[col[s:e]] - pos[row[s:e]]).norm(dim=-1, keepdim=True).clamp_min(EPS)
                msg = self.edge_mlp(torch.cat([hi, hj, dist], dim=-1))
                out.index_add_(0, col[s:e], msg)
            return out
        return self.propagate(edge_index, size=(n, n), h=h, pos=pos)

    def message(self, h_i, h_j, pos_i, pos_j):
        dist = (pos_i - pos_j).norm(dim=-1, keepdim=True).clamp_min(EPS)
        return self.edge_mlp(torch.cat([h_i, h_j, dist], dim=-1))


class EquivariantExecutor(nn.Module):
    """One learned step of the layout operator. See module docstring.

    `use_stride=True` enables the Phi_k compression conditioning:
    `stride` (how many algorithm steps one forward pass
    covers) becomes an extra scalar argument to phi. It must be False for
    Phi_1 (single-step checkpoints were trained with the smaller phi input) --
    changing phi_in_dim would silently invalidate those state dicts.
    """

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 64,
        num_teachers: int = 1,
        k_geo: int = 10,
        use_stride: bool = False,
        knn_backend: str = "brute",
        use_tau: bool = True,
        use_geo_mp: bool = True,
        use_equiv_readout: bool = True,
        chunk: int = None,
    ):
        super().__init__()
        self.k_geo = k_geo
        self.use_stride = use_stride
        self.knn_backend = knn_backend
        self.use_tau = use_tau
        self.use_geo_mp = use_geo_mp
        self.use_equiv_readout = use_equiv_readout
        # Chunk size for per-edge processing (message passing AND the readout).
        # None = original monolithic path (bit-identical numerics for existing
        # checkpoints); an int bounds peak GPU memory to O(chunk) edge messages
        # so N=10^6 runs without the 24 GB OOM (the large-N memory wall).
        self.chunk = chunk

        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.attraction_mp = _EdgeContextMP(hidden_dim)
        self.repulsion_mp = _EdgeContextMP(hidden_dim)
        self.combine = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.teacher_embed = nn.Embedding(num_teachers, hidden_dim)

        # phi(h_v, h_u, dist, is_topo, is_geo, tau_t, c [, stride]) -> scalar
        phi_in_dim = 2 * hidden_dim + 1 + 2 + (1 if use_tau else 0) + hidden_dim
        if use_stride:
            phi_in_dim += 1
        # Ablation (non-equivariant readout): phi outputs a per-edge
        # [E, 2] displacement VECTOR instead of a scalar times the unit
        # vector. The model can then learn arbitrary directions, so the
        # equivariance guarantee is deliberately broken. Same phi_in_dim.
        out_dim_phi = 1 if use_equiv_readout else 2
        self.phi = nn.Sequential(
            nn.Linear(phi_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim_phi),
        )

    def forward(
        self,
        node_feat: torch.Tensor,       # [N, node_feat_dim] structural encodings only
        edge_index_topo: torch.Tensor,  # [2, E] topological edges (both directions)
        pos: torch.Tensor,              # [N, 2] current positions, k-normalized
        tau: torch.Tensor,              # scalar, [N] or [N, 1]: current temperature, k-normalized
        teacher_id: torch.Tensor = None,  # scalar or [N] long: which teacher/algorithm (c)
        stride: torch.Tensor = None,    # scalar or [N]: steps compressed into this forward pass (Phi_k)
    ) -> torch.Tensor:                  # [N, 2] displacement dX
        n = pos.shape[0]
        device = pos.device
        chunk = self.chunk

        if tau.dim() == 0:
            tau = tau.expand(n).unsqueeze(-1)
        elif tau.dim() == 1:
            tau = tau.unsqueeze(-1)

        if teacher_id is None:
            teacher_id = torch.zeros(n, dtype=torch.long, device=device)
        elif teacher_id.dim() == 0:
            teacher_id = teacher_id.expand(n)

        if self.use_stride:
            if stride is None:
                stride = torch.ones(n, dtype=torch.float32, device=device)
            elif stride.dim() == 0:
                stride = stride.expand(n)
            stride = stride.unsqueeze(-1).float()  # [N, 1]

        h0 = self.node_encoder(node_feat)

        if self.use_geo_mp:
            edge_index_geo = build_knn_graph(pos, k=min(self.k_geo, n - 1), backend=self.knn_backend)
            h_repl = self.repulsion_mp(h0, edge_index_geo, pos, chunk=chunk)
        else:
            # Ablation (no geometric rewiring): message passing on the
            # topological E only. No kNN is built; repulsion gets zero
            # contribution and is_geo is False everywhere (phi still sees the
            # dimension, so phi_in_dim is unchanged and the checkpoint layout
            # is stable).
            edge_index_geo = torch.empty((2, 0), dtype=torch.long, device=device)
            h_repl = torch.zeros_like(h0)

        h_attr = self.attraction_mp(h0, edge_index_topo, pos, chunk=chunk)
        h = self.combine(torch.cat([h0, h_attr, h_repl], dim=-1))

        row, col, is_topo, is_geo = merge_edge_sets(edge_index_topo, edge_index_geo, n)
        # row = source u, col = target v; dX[v] accumulates contributions from u.
        # Chunked readout: phi_in concatenation is O(E * phi_in_dim), which is
        # the 24 GB OOM allocation at N=10^6. Process edge blocks instead.

        c_all = self.teacher_embed(teacher_id[col])  # [E, hidden_dim] (cached; small)

        def _process(lo: int, hi: int) -> torch.Tensor:
            rs, cs = row[lo:hi], col[lo:hi]
            delta = pos[cs] - pos[rs]  # x_v - x_u, [E', 2]
            dist = delta.norm(dim=-1, keepdim=True).clamp_min(EPS)
            unit = delta / dist
            phi_in = [
                h[cs],
                h[rs],
                dist,
                is_topo[lo:hi].unsqueeze(-1).float(),
                is_geo[lo:hi].unsqueeze(-1).float(),
            ]
            if self.use_tau:
                phi_in.append(tau[cs])
            phi_in.append(c_all[lo:hi])
            if self.use_stride:
                phi_in.append(stride[cs])
            phi = self.phi(torch.cat(phi_in, dim=-1))  # [E', 1] or [E', 2] (ablation)
            if self.use_equiv_readout:
                return unit * phi  # equivariant by construction
            # Ablation: phi outputs a full displacement vector per edge.
            return phi

        dX = torch.zeros(n, 2, device=device, dtype=pos.dtype)
        if chunk is None:
            contrib = _process(0, col.shape[0])
            dX.index_add_(0, col, contrib)
        else:
            for s in range(0, col.shape[0], chunk):
                e = min(s + chunk, col.shape[0])
                contrib = _process(s, e)
                dX.index_add_(0, col[s:e], contrib)
        if self.use_tau:
            dX = tau * dX

        return dX
