"""
gst/sign_net.py

SignNet: sign-invariant processing of Laplacian eigenvectors.

Problem: Laplacian eigenvectors are sign-ambiguous — if phi is an eigenvector,
so is -phi. Two identical graphs could receive opposite-sign LapPE, making them
look different to the model even though they are structurally identical.

Solution (from Lim et al. 2022): process eigenvectors through a network that
produces the same output regardless of the sign of each eigenvector column.

Architecture:
  For each eigenvector column phi_k (one at a time):
    - Compute rho(phi_k) + rho(-phi_k)   ← sum over both signs, invariant by design
    - rho is a 2-layer MLP per eigenvector
  Sum results across all k eigenvectors through a DeepSet aggregation.
  Final MLP maps to hidden dimension d.

This is the DeepSet variant described in the GPS paper (Table 1, SignNetDeepSets).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignNet(nn.Module):
    """
    Sign-invariant network for processing Laplacian eigenvectors.

    Input:  eigvecs [N, k] — raw Laplacian eigenvectors (sign-ambiguous)
    Output: lap_pe  [N, d] — sign-invariant positional encoding

    Architecture per eigenvector column phi_k:
        inner_mlp(phi_k) + inner_mlp(-phi_k)   ← sign invariant

    Aggregation across k eigenvectors:
        sum over k                              ← DeepSet aggregation

    Final:
        outer_mlp(aggregated)  →  [N, d]
    """

    def __init__(
        self,
        k: int = 10,        # number of eigenvectors
        hidden_dim: int = 64,
        out_dim: int = 256,  # must match GST hidden_dim d
    ):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Inner MLP: processes one eigenvector column at a time
        # Input is a single scalar per node (one column of eigvecs)
        self.inner_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Outer MLP: maps aggregated representation to output dim
        self.outer_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, eigvecs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eigvecs: [N, k] — Laplacian eigenvectors, potentially sign-ambiguous

        Returns:
            lap_pe: [N, out_dim] — sign-invariant node-level positional encoding
        """
        N, k = eigvecs.shape

        # inner_mlp acts on one scalar at a time, so every (node, eigenvector)
        # entry is an independent sample: flatten all k columns into the batch
        # dimension and run the MLP once instead of once per column.
        phi = eigvecs.t().reshape(-1, 1)                    # [k*N, 1]

        # Sign-invariant: rho(phi) + rho(-phi)
        # Same output regardless of which sign was assigned to this eigenvector
        both = self.inner_mlp(phi) + self.inner_mlp(-phi)   # [k*N, hidden_dim]

        # DeepSet aggregation: sum across the k eigenvectors
        agg = both.view(k, N, self.hidden_dim).sum(dim=0)   # [N, hidden_dim]

        # Final projection to output dimension
        lap_pe = self.outer_mlp(agg)                        # [N, out_dim]

        return lap_pe