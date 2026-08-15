"""
glide/executor/mpnn.py

Q20 (the experiment queue): MPNN-max Neural Executor baseline
(Velicovic et al. 2020, "Neural Execution of Graph Algorithms", ICLR 2020).

The cleanest NAR baseline: one algorithm step per forward pass, the predicted
state fed back into the next step, message passing with MAX aggregation. This
is exactly an UNALIGNED step operator -- a plain MPNN with NO equivariant
readout and NO geometric kNN rewiring -- trained on the same FR/FA2/KK
trajectories, same state/output encoding, same teacher-forcing exposure as
the EquivariantExecutor.

Why it exists (PAPER_PLAN.md §9): it isolates what algorithmic alignment
buys. The equivariant executor's hypothesis class CONTAINS the algorithm
(constructionally), which is what makes step supervision sample-efficient. A
plain MPNN on the same data tests whether alignment is the reason -- if the
MPNN reaches the same fidelity it does, alignment bought nothing.

This is deliberately NOT the EquivariantExecutor with flags flipped; it is a
separate, ordinary MPNN processor:
  - topological message passing only (no geometric kNN)
  - node update = max over neighbours of an edge-MLP of (h_u, dist)
  - output = an MLP on the node state -> [N, 2] displacement (no unit-vector
    equivariance, no tau prefactor)
This maximises the contrast: same data, same loss, same training loop, only
the inductive bias differs.

Usage: pass the class into a training loop that otherwise matches train.py
(per-graph, all steps summed into one loss/update, k-unit normalization).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .geometric import build_knn_graph

EPS = 1e-8


class _MaxMP(MessagePassing):
    """Max aggregation over topological neighbours, edge-conditioned on
    distance (so it can at least *see* geometry through edge lengths)."""

    def __init__(self, hidden_dim: int):
        super().__init__(aggr="max")
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index, pos):
        return self.propagate(edge_index, size=(h.shape[0], h.shape[0]), h=h, pos=pos)

    def message(self, h_i, h_j, pos_i, pos_j):
        dist = (pos_i - pos_j).norm(dim=-1, keepdim=True).clamp_min(EPS)
        return self.edge_mlp(torch.cat([h_i, h_j, dist], dim=-1))


class MpnnExecutor(nn.Module):
    """Unaligned neural executor (Q20). One step X_{t+1}=X_t + dX, dX from a
    plain MPNN with max aggregation; no equivariant readout, no geometric
    rewiring, no temperature conditioning of the readout."""

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.layers = nn.ModuleList([_MaxMP(hidden_dim) for _ in range(num_layers)])
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, node_feat, edge_index, pos, tau, teacher_id=None, stride=None):
        h = self.node_encoder(node_feat)
        for layer in self.layers:
            h = h + layer(h, edge_index, pos)  # residual
        return self.output_mlp(h)  # [N, 2] -- NO unit-vector constraint
