"""
glide/executor/forgetnet.py

Q26 (the experiment queue): G-ForgetNet executor baseline (Bohde et al.
2024, "On the Markov Property of Neural Algorithmic Reasoning", ICLR 2024).

Three variants on the same data:
  (a) history-KEEPING processor: a hidden state carried across rollout steps
  (b) history-FREE processor: each step from (G, X_t, tau_t) only  <-- this is
      exactly the EquivariantExecutor; it is Markov by construction
  (c) gated-history G-ForgetNet: a gated memory that decides how much of the
      previous hidden state to keep.

If history-free (our executor) performs at least as well as gated-history,
that STRENGTHENS the paper's interpretation: the layout operator
X_{t+1}=Phi(G,X_t,tau_t) is Markov by construction, and the learned model
agrees with it -- remembering the past is neither necessary nor helpful.

G-ForgetNet processor (simplified to the gated-memory idea, on the same
topological MPNN backbone as MpnnExecutor):

  z_v = sigmoid(W_z [h_v^t, m_v, s_v])            # update gate
  h_v^{t+1} = (1 - z_v) * h_v^t + z_v * tanh(W_h m_v)   # gated carry

where m_v is the MPNN-aggregated message and s_v is the carried state.
A forget gate r_v = sigmoid(W_r [m_v, s_v]) modulates the message read:
  m_v = max_u edge_mlp(h_u, r_v * s_u ...)

This is the "gated-history" variant (c): the memory is retained unless the
gate closes it. The contrast with (b) is the Q26 control.

Training passes the hidden state across steps within a graph's rollout
(h_state from step t feeds step t+1), which is the one structural difference
from the Markov executor's training loop. The hidden state resets per graph.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

EPS = 1e-8


class _GatedMP(MessagePassing):
    """Message passing with an edge-gated read of the carried state: the
    message on edge v<-u is edge_mlp(h_v, h_u, s_v, s_u) where s are the
    carried per-node hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__(aggr="max")
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, s, edge_index):
        return self.propagate(edge_index, size=(h.shape[0], h.shape[0]), h=h, s=s)

    def message(self, h_i, h_j, s_i, s_j):
        return self.edge_mlp(torch.cat([h_i, h_j, s_i, s_j], dim=-1))


class GForgetNet(nn.Module):
    """Gated-history executor (Q26c). h_state is carried across rollout steps
    and combined with the fresh message under a learned gate."""

    HAS_HISTORY = True

    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.mp_layers = nn.ModuleList([_GatedMP(hidden_dim) for _ in range(num_layers)])
        self.update_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def init_state(self, node_feat: torch.Tensor) -> torch.Tensor:
        """Per-graph initial hidden state: zero vector on the node_encoder dim."""
        return torch.zeros(node_feat.shape[0], self.node_encoder.out_features,
                           device=node_feat.device)

    def forward(self, node_feat, edge_index, pos, tau, teacher_id=None, stride=None,
                h_state: torch.Tensor = None):
        """One step. `h_state` is the carried memory from the previous step
        (None on the first step of a graph's rollout -> init to zero). Returns
        (dX, h_next)."""
        if h_state is None:
            h_state = self.init_state(node_feat)
        h = self.node_encoder(node_feat)
        for layer in self.mp_layers:
            msg = layer(h, h_state, edge_index)  # gated read of carried state
            h = h + msg
        # update gate: how much of the carried state to keep vs the new message
        gate = torch.sigmoid(self.update_gate(torch.cat([h, h_state], dim=-1)))
        h_next = (1 - gate) * h_state + gate * torch.tanh(h)
        dX = self.output_mlp(h_next)
        return dX, h_next
