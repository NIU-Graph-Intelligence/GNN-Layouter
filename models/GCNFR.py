import torch
from torch import nn
from torch_scatter import scatter_mean

class NodeModel(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.norm_before_rep = nn.LayerNorm(self.in_feat)
        self.norm_after_rep = nn.LayerNorm(self.in_feat)

        # Simplified MLP blocks; adjust as needed.
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(in_feat, in_feat, bias=False),
            nn.ReLU(),
            nn.Linear(in_feat, in_feat, bias=False),
        )

        self.node_mlp_2 = nn.Sequential(
            nn.Linear(2 * in_feat, in_feat, bias=False),
            nn.ReLU(),
            nn.Linear(in_feat, out_feat, bias=False)
        )

        # One learnable weight per feature‐dimension
        self.w = nn.Parameter(torch.ones(in_feat))

    def forward(self, x, edge_index, init_coords=None):

        if init_coords is not None:
            # First layer concatenates raw features + coords
            x_input = torch.cat([x, init_coords], dim=1)
        else:
            x_input = x

        row, col = edge_index
        neighbor_msg = x_input[row]  # [E, in_feat]
        neighbor_msg = self.node_mlp_1(neighbor_msg)  # [E, in_feat]

        # scatter_mean: average over all edges pointing to each node
        agg_neighbors = scatter_mean(
            neighbor_msg,
            col,
            dim=0,
            dim_size=x_input.size(0)
        )  # [N, in_feat]

        agg_neighbors = self.norm_before_rep(agg_neighbors)

        repulsion = (x_input - agg_neighbors) * self.w  # [N, in_feat]
        fx = x_input + repulsion  # [N, in_feat]

        fx = self.norm_after_rep(fx)

        out_final = torch.cat([fx, agg_neighbors], dim=1)  # [N, 2*in_feat]
        return self.node_mlp_2(out_final)  # [N, out_feat]

class ForceGNN(nn.Module):
        def __init__(self, in_feat, hidden_dim, out_feat, num_layers):
            super().__init__()
            self.layers = nn.ModuleList()

            # First layer: input feature size (+2 for init coords)
            self.layers.append(NodeModel(in_feat, hidden_dim))
            # Hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(NodeModel(hidden_dim, hidden_dim))

            # Last layer: output dimension
            self.layers.append(NodeModel(hidden_dim, out_feat))

        def forward(self, x, edge_index, init_coords=None):
            # Layer 0: concat (x, init_coords) once
            h = self.layers[0](x, edge_index, init_coords)

            # 2) Hidden layers (1 … num_layers-2) with skip
            for i in range(1, len(self.layers) - 1):
                h_new = self.layers[i](h, edge_index, init_coords=None)  # [N, hidden_feat]
                h = h_new + h  # safe because both are [N, hidden_feat]

            # 3) Final layer: no skip-add (h is [N, hidden_feat], final maps → [N, out_feat])
            coords_out = self.layers[-1](h, edge_index, init_coords=None)  # [N, out_feat=2]
            return coords_out





