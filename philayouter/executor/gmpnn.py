"""
philayouter/executor/gmpnn.py

Q25 (the experiment queue): Triplet-GMPNN executor baseline (Ibarz et al.
2022, "A Generalist Neural Algorithmic Learner", LoG 2022).

The strongest standard NAR processor. This is the control that answers "we did
not win because we used a stronger architecture": same trajectories, same
state/output encoding, same teacher-forcing exposure as the EquivariantExecutor,
only the processor is swapped to a Triplet-GMPNN.

Triplet-GMPNN uses TRIPLET message passing (centre - neighbour - node
triplets) over the topological graph, the form the NAR literature found
strongest on CLRS-30. The message on edge v<-u is a function of the centre h_v,
the neighbour h_u, AND a summary of u's own neighbourhood (the "triplet"
w in N(u)):

    m(v,u) = f( h_v, h_u,  max_w in N(u) g(h_u, h_w) )

computed in two propagation passes: first aggregate a per-node neighbour
summary s_u = max_w g(h_u, h_w), then send m(v,u) = f(h_v, h_u, s_u) over the
edges. Max aggregation both times, matching the NAR convention.

Deliberately NOT equivariant and NOT geometrically rewired -- a plain (strong)
GNN step operator -- so comparing against EquivariantExecutor isolates what
algorithmic alignment buys, holding architecture strength at or above the
executor's.

Usage: same training loop as train_mpnn.py (per-graph, all steps in one
loss/update, k-unit normalization).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class _NeighbourSummary(MessagePassing):
    """First pass: s_u = max_w in N(u) of g(h_u, h_w)."""

    def __init__(self, hidden_dim: int):
        super().__init__(aggr="max")
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index):
        return self.propagate(edge_index, size=(h.shape[0], h.shape[0]), h=h)

    def message(self, h_i, h_j):
        return self.edge_mlp(torch.cat([h_i, h_j], dim=-1))


class _TripletSend(MessagePassing):
    """Second pass: m(v,u) = f(h_v, h_u, s_u) where s_u is u's neighbour
    summary from the first pass."""

    def __init__(self, hidden_dim: int):
        super().__init__(aggr="max")
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, s, edge_index):
        return self.propagate(edge_index, size=(h.shape[0], h.shape[0]), h=h, s=s)

    def message(self, h_i, h_j, s_j):
        return self.edge_mlp(torch.cat([h_i, h_j, s_j], dim=-1))


class TripletGMPNN(nn.Module):
    """Triplet-GMPNN step operator (Q25). Two-pass triplet message passing."""

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.summary_layers = nn.ModuleList(
            [_NeighbourSummary(hidden_dim) for _ in range(num_layers)]
        )
        self.send_layers = nn.ModuleList(
            [_TripletSend(hidden_dim) for _ in range(num_layers)]
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, node_feat, edge_index, pos, tau, teacher_id=None, stride=None):
        h = self.node_encoder(node_feat)
        for summ, send in zip(self.summary_layers, self.send_layers):
            s = summ(h, edge_index)  # per-node neighbour summary
            h = h + send(h, s, edge_index)  # triplet update, residual
        return self.output_mlp(h)  # [N, 2] -- no equivariance
