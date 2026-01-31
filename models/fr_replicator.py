# fr_replicator.py
# One-step (or multi-step) FR replicator with latent edge stiffness (topology-only).
# Paste this into your project and import FRUnrolled + compute_layout_loss where needed.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

__all__ = [
    "FRUnrolled",
    "compute_layout_loss",                # batched loss (with distillation)
    "aligned_mse_loss_batched",          # optional: alignment-only loss
]

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def infer_batch_from_edge_index(num_nodes: int,
                                edge_index: torch.Tensor,
                                device: torch.device) -> torch.Tensor:
    """
    Infer per-node graph IDs by connected components (undirected).
    Returns: batch (N,) with labels 0..G-1
    """
    row, col = edge_index
    N = num_nodes

    adj = [[] for _ in range(N)]
    # NOTE: edge_index can be already bidirectional; that's fine.
    row_cpu = row.detach().cpu().tolist()
    col_cpu = col.detach().cpu().tolist()
    for u, v in zip(row_cpu, col_cpu):
        if u != v:
            adj[u].append(v)
            adj[v].append(u)

    comp_id = [-1] * N
    comp = 0
    for n in range(N):
        if comp_id[n] != -1:
            continue
        stack = [n]
        comp_id[n] = comp
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if comp_id[w] == -1:
                    comp_id[w] = comp
                    stack.append(w)
        comp += 1

    return torch.tensor(comp_id, dtype=torch.long, device=device)


# ------------------------------------------------------------
# Physics (FR-style) with learned latent edge weights
# ------------------------------------------------------------

def _compute_fr_forces(coords, edge_index, batch,
                       k_mode="sqrt_inv_n", k_constant=0.1,
                       edge_weight=None,        # <- NEW: learned latent stiffness
                       eps=1e-2):               # <- match NetworkX 0.01 clamp
    device = coords.device
    N = coords.size(0)

    # per-graph k
    ones = torch.ones(N, device=device)
    num_nodes_per_graph = scatter_sum(ones, batch, dim=0)
    if k_mode == "sqrt_inv_n":
        k_per_graph = 1.0 / torch.sqrt(num_nodes_per_graph.clamp(min=1))
    else:
        k_per_graph = torch.full_like(num_nodes_per_graph, float(k_constant))
    k_node = k_per_graph[batch]

    # ---- Attraction (edge-based; scale by edge_weight if provided)
    row, col = edge_index
    diff = coords[row] - coords[col]                 # (E,2)
    dist = torch.norm(diff, dim=1) + eps            # (E,)
    k_e  = k_node[row]                               # (E,)
    attr_vec_row = -(dist / k_e).unsqueeze(1) * diff  # (E,2)
    if edge_weight is not None:
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(1)
        attr_vec_row = attr_vec_row * edge_weight
    attr_vec_col = -attr_vec_row

    F_attr = torch.zeros_like(coords)
    F_attr.index_add_(0, row, attr_vec_row)
    F_attr.index_add_(0, col, attr_vec_col)

    # ---- Repulsion (per component; unweighted)
    F_rep = torch.zeros_like(coords)
    for g in batch.unique(sorted=True):
        idx_g = torch.nonzero(batch == g, as_tuple=False).view(-1)
        ng = idx_g.numel()
        if ng <= 1:
            continue
        coords_g = coords[idx_g]
        diff_g = coords_g.unsqueeze(1) - coords_g.unsqueeze(0)   # (ng,ng,2)
        dist_g = torch.norm(diff_g, dim=2) + eps                 # (ng,ng)
        k_g = k_per_graph[g]
        mag_rep_g = (k_g * k_g) / dist_g
        mag_rep_g.fill_diagonal_(0.0)
        dir_g = diff_g / dist_g.unsqueeze(2)
        F_rep_g = (mag_rep_g.unsqueeze(2) * dir_g).sum(dim=1)    # (ng,2)
        F_rep[idx_g] += F_rep_g

    return F_attr + F_rep


# ------------------------------------------------------------
# Learnable pieces
# ------------------------------------------------------------

class EdgeWeightHead(nn.Module):
    """
    Predict latent stiffness w_hat >= 0 from topology-only edge features φ(i,j).
    """
    def __init__(self, in_dim: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, e_feat: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(e_feat)) + 1e-6   # (E,1)


class GNNResidual(nn.Module):
    """
    Simple neighbor-aggregation residual: returns per-node (dx,dy).
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index  # messages from col -> row
        m = self.msg_mlp(h[col])          # (E, H)

        N = h.size(0)
        agg = torch.zeros(N, m.size(1), device=h.device)
        agg.index_add_(0, row, m)

        deg = torch.bincount(row, minlength=N).to(h.device).clamp(min=1).float().unsqueeze(1)
        agg = agg / deg

        return self.update_mlp(torch.cat([h, agg], dim=1))  # (N,2)


class NeighborAggregator(nn.Module):
    """
    Message passing layer that returns node embeddings with the same dimension.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        m = self.msg_mlp(h[col])

        agg = torch.zeros(h.size(0), m.size(1), device=h.device)
        agg.index_add_(0, row, m)

        deg = torch.bincount(row, minlength=h.size(0)).to(h.device).clamp(min=1).float().unsqueeze(1)
        agg = agg / deg

        return self.update_mlp(torch.cat([h, agg], dim=1))


class EdgeWeightPredictor(nn.Module):
    """
    Learns a latent stiffness per undirected edge using node message passing.
    """
    def __init__(self,
                 node_input_dim: int,
                 embed_dim: int = 64,
                 num_layers: int = 2,
                 head_hidden: int = 64,
                 include_degree: bool = True):
        super().__init__()
        self.include_degree = include_degree
        self.input_proj = nn.Linear(node_input_dim, embed_dim)
        self.layers = nn.ModuleList([NeighborAggregator(embed_dim) for _ in range(num_layers)])

        head_in = 2 * embed_dim
        if include_degree:
            head_in += 2  # deg_i, deg_j
        self.edge_head = EdgeWeightHead(in_dim=head_in, hidden=head_hidden)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(node_features)
        for layer in self.layers:
            h = h + layer(h, edge_index)

        row, col = edge_index
        N = node_features.size(0)
        deg = torch.bincount(row, minlength=N).to(node_features.device).float()

        num_edges = edge_index.size(1)
        if num_edges % 2 != 0:
            raise ValueError("edge_index must contain bidirectional edges for symmetry.")
        undirected = num_edges // 2

        canonical_edges = edge_index[:, :undirected]
        src = canonical_edges[0]
        dst = canonical_edges[1]

        pair_feat = [h[src], h[dst]]
        if self.include_degree:
            pair_feat.extend([deg[src].unsqueeze(1), deg[dst].unsqueeze(1)])
        edge_feat = torch.cat(pair_feat, dim=1)

        w_half = self.edge_head(edge_feat)  # (E/2, 1)
        return torch.cat([w_half, w_half], dim=0)


class FRCell(nn.Module):
    def __init__(self,
                 in_dim,
                 k_mode="sqrt_inv_n",
                 k_constant=0.1,
                 clamp_step=0.1,
                 learn_step=True,
                 use_residual=False,
                 residual_hidden=64):
        super().__init__()
        self.k_mode = k_mode
        self.k_constant = k_constant
        self.clamp_step = clamp_step
        self.use_residual = use_residual

        if learn_step:
            self.alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("alpha", torch.tensor(1.0))

        if use_residual:
            self.residual_gnn = GNNResidual(in_dim=in_dim, hidden_dim=residual_hidden)
        else:
            self.residual_gnn = None

    def forward(self, h_t, edge_index, batch, edge_weight=None):
        coords_t = h_t[:, :2]
        feat_t   = h_t[:, 2:]

        delta_fr = _compute_fr_forces(
            coords=coords_t,
            edge_index=edge_index,
            batch=batch,
            k_mode=self.k_mode,
            k_constant=self.k_constant,
            edge_weight=edge_weight,
            eps=1e-2
        )

        delta_total = self.alpha * delta_fr

        if self.use_residual:
            delta_total = delta_total + self.residual_gnn(h_t, edge_index)

        step_norm = torch.norm(delta_total, dim=1, keepdim=True)
        scale = (self.clamp_step / (step_norm + 1e-9)).clamp(max=1.0)
        delta_total = delta_total * scale

        coords_next = coords_t + delta_total
        return torch.cat([coords_next, feat_t], dim=1)


class FRUnrolled(nn.Module):
    """
    Unroll K steps of FRCell; return final coords.
    Input x: (N, F), with LAST 2 = initial coords, FIRST F-2 = node features.
    """
    def __init__(self,
                 input_dim: int,
                 *,
                 steps: int = 3,
                 k_mode: str = "sqrt_inv_n",
                 k_constant: float = 0.1,
                 clamp_step: float = 0.05,
                 learn_step: bool = True,
                 use_residual: bool = True,
                 residual_hidden: int = 64,
                 learn_edge_weight: bool = True,
                 edge_head_hidden: int = 32,
                 edge_embed_dim: int = 64,
                 edge_gnn_layers: int = 2,
                 include_degree_in_edge: bool = True,
                 center_output: bool = True):
        super().__init__()
        self.steps = steps
        self.center_output = center_output
        self.learn_edge_weight = learn_edge_weight

        self.cell = FRCell(
            in_dim=input_dim,
            k_mode=k_mode,
            k_constant=k_constant,
            clamp_step=clamp_step,
            learn_step=learn_step,
            use_residual=use_residual,
            residual_hidden=residual_hidden,
        )
        self.edge_predictor = None
        if learn_edge_weight:
            self.edge_predictor = EdgeWeightPredictor(
                node_input_dim=input_dim,
                embed_dim=edge_embed_dim,
                num_layers=edge_gnn_layers,
                head_hidden=edge_head_hidden,
                include_degree=include_degree_in_edge
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        device = x.device
        N = x.size(0)
        batch = infer_batch_from_edge_index(N, edge_index, device=device)

        feat_no_xy = x[:, :-2]  # Node features (N, F-2)
        init_xy    = x[:, -2:]  # Initial coords (N, 2)
        h = torch.cat([init_xy, feat_no_xy], dim=1)

        edge_weight = None
        if self.edge_predictor is not None:
            edge_weight = self.edge_predictor(h, edge_index)

        for _ in range(self.steps):
            h = self.cell(h, edge_index, batch, edge_weight=edge_weight)

        coords_final = h[:, :2]

        return coords_final
