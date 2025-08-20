import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

class NodeModel(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        
        # Force processing components
        self.norm_before_rep = nn.LayerNorm(in_feat)
        self.norm_after_rep = nn.LayerNorm(in_feat)
        self.w = nn.Parameter(torch.ones(in_feat))

        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(in_feat, in_feat, bias=False),
            nn.ReLU(),
            nn.Linear(in_feat, in_feat, bias=False),
        )
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * in_feat, in_feat, bias=False),
            nn.ReLU(),
            nn.Linear(in_feat, out_feat, bias=False)
        )

    def process_force_messages(self, x_input, neighbor_msg, col, size):
        """Process force messages and return force features"""
        # Aggregate neighbor messages
        agg_neighbors = scatter_mean(neighbor_msg, col, dim=0, dim_size=size)
        agg_neighbors = self.norm_before_rep(agg_neighbors)
        
        # Calculate repulsion
        repulsion = (x_input - agg_neighbors) * self.w
        
        # Final processing
        fx = x_input + repulsion
        fx = self.norm_after_rep(fx)
        
        return fx, agg_neighbors

    def forward(self, x, edge_index):
        """Standard PyG interface"""
        row, col = edge_index
        
        # Process messages
        neighbor_msg = self.message_mlp(x[row])
        
        # Process forces
        fx, agg_neighbors = self.process_force_messages(
            x, neighbor_msg, col, x.size(0)
        )
        
        # Output processing
        out_final = torch.cat([fx, agg_neighbors], dim=1)
        return self.output_mlp(out_final)

class ForceGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=4):
        super().__init__()
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.layers.append(NodeModel(input_dim, hidden_dim))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(NodeModel(hidden_dim, hidden_dim))
        
        # Final layer: hidden_dim -> 2 (coordinates)
        self.layers.append(NodeModel(hidden_dim, 2))

    def forward(self, x, edge_index):
        """Standard PyG interface"""
        # First layer
        h = self.layers[0](x, edge_index)
        
        # Hidden layers with residual connections
        for layer in self.layers[1:-1]:
            h_new = layer(h, edge_index)
            h = h_new + h  # Residual connection
        
        # Final layer
        coords = self.layers[-1](h, edge_index)
        return coords