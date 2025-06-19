import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
import torch.nn.functional as F
import torch


class GNN_Model_GIN(nn.Module):
    def __init__(self, max_nodes, hidden_channels=128):
        super().__init__()
        input_dim = max_nodes + 1  # +1 for positional feature

        # Smaller hidden channels and more regularization
        self.dropout_rate = 0.4  # Increased dropout
        
        # First GIN layer with reduced dimension
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),  # LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_channels, hidden_channels)
        ), train_eps=True)

        # Second GIN layer
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_channels, hidden_channels)
        ), train_eps=True)

        # Third GIN layer
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_channels, hidden_channels)
        ), train_eps=True)
        
        # Fourth GIN layer for better message passing
        self.conv4 = GINConv(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_channels, hidden_channels)
        ), train_eps=True)
        
        # Attention-like weighting mechanism (similar to GAT)
        self.attention = nn.Sequential(
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

        # More direct angle prediction with wider MLP
        self.angle_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.5),  # Higher dropout for stronger regularization
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh()  # Output in [-1, 1] range for angle prediction
        )

        # Radius prediction with constrained output
        self.radius_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()  # Output in [0, 1] range for scaling
        )

        # Initialize weights with smaller values
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use smaller weights initialization for better gradient flow
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, edge_index):
        # Initial node features with stronger regularization
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)
        
        # Second layer with residual
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, p=self.dropout_rate, training=self.training)
        h2 = h2 + h1  # Residual connection
        
        # Third layer with residual
        h3 = F.relu(self.conv3(h2, edge_index))
        h3 = F.dropout(h3, p=self.dropout_rate, training=self.training)
        h3 = h3 + h2  # Residual connection
        
        # Fourth layer with residual 
        h4 = F.relu(self.conv4(h3, edge_index))
        h4 = F.dropout(h4, p=self.dropout_rate, training=self.training)
        h4 = h4 + h3  # Residual connection

        # Apply attention weights to focus on important nodes
        attention_weights = self.attention(h4)
        h_weighted = h4 * attention_weights
        
        # Circular coordinate prediction with separate angle and radius
        angles = self.angle_mlp(h_weighted) * torch.pi  # Scale to [-π, π]
        
        # Force radius to be close to 1 with small variations (0.9 to 1.1)
        radius = 0.9 + 0.2 * self.radius_mlp(h_weighted)
        
        # Convert angle and radius to coordinates
        coords = torch.zeros(x.size(0), 2, device=x.device)
        coords[:, 0] = radius.squeeze() * torch.cos(angles.squeeze())  # x = r * cos(θ)
        coords[:, 1] = radius.squeeze() * torch.sin(angles.squeeze())  # y = r * sin(θ)
        
        # Normalize to make sure layout is centered
        coords = coords - coords.mean(dim=0, keepdim=True)
        
        return coords
