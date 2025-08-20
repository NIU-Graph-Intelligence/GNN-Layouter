import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class ChebNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, K=3, dropout=0.1):
        super().__init__()
        
        self.dropout = dropout
        
        # Build Chebyshev convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(ChebConv(input_dim, hidden_dim, K=K))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(ChebConv(hidden_dim, hidden_dim, K=K))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
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
        h = self.convs[0](h, edge_index)
        h = F.relu(self.bns[0](h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Remaining layers with residual connections
        for i in range(1, len(self.convs)):
            h_new = self.convs[i](h, edge_index)
            h_new = F.relu(self.bns[i](h_new))
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new + h  # Residual connection
        
        # Output coordinates
        coords = self.output_layer(h)
        return coords