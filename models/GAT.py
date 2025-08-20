import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        
        self.dropout = dropout
        self.heads = heads
        
        # Build GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer for coordinates
        self.output_layer = nn.Linear(hidden_dim, 2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index):
        h = x.float()
        
        # GAT layers with normalization and residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            
            # Residual connection (skip first layer due to dimension change)
            if i > 0:
                h_new = h_new + h
            h = h_new
        
        # Output coordinates
        coords = self.output_layer(h)
        return coords