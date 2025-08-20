import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.dropout = dropout
        
        # Build GIN layers
        self.convs = nn.ModuleList()
        
        # First layer
        first_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(first_mlp, train_eps=True))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
        
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
        
        # First layer
        h = F.relu(self.convs[0](h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Remaining layers with residual connections
        for conv in self.convs[1:]:
            h_new = F.relu(conv(h, edge_index))
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # Residual connection
        
        # Output coordinates
        coords = self.output_layer(h)
        return coords