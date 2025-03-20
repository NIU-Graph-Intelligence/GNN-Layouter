import torch.nn as nn
from torch_geometric.nn import GCN, GCNConv
import torch.nn.functional as F


class GNN_Model_1(nn.Module):
    def __init__(self, max_nodes=50, hidden_channels=128): # 50 (one-hot) + 1 ( positional features )
        super().__init__()

        input_dim = max_nodes + 1
        # GNN layers
        self.conv1 = GCNConv(input_dim, hidden_channels) # 53 -> 64
        self.conv2 = GCNConv(hidden_channels, hidden_channels) # 64 -> 64
        self.conv3 = GCNConv(hidden_channels, hidden_channels)


        self.pos_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, 2)
        )

        self.radius_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus()
        )

    def forward(self, x, edge_index):
        # Initial GNN layers with residual connections
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.3, training=self.training)

        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.3, training=self.training)
        h2 = h2 + h1  # Residual connection

        h3 = self.conv3(h2, edge_index)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=0.3, training=self.training)
        h3 = h3 + h2  # Residual connection

        # Predict radius and coordinates separately
        radius = self.radius_mlp(h3)
        coords = self.pos_mlp(h3)

        # Normalize coordinates to unit circle
        coords_norm = F.normalize(coords, p=2, dim=1)

        # Scale by predicted radius
        coords = coords_norm * radius

        return coords


