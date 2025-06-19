import torch.nn as nn
from torch_geometric.nn import GCN, GCNConv
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, max_nodes, hidden_channels=192):
        super().__init__()
        input_dim = max_nodes + 1

        # Graph convolution layers - added conv4 to match successful model
        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)  # Added this layer

        # Position MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_channels // 2, 2)
        )

        # Radius prediction
        self.radius_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus()
        )

    def forward(self, x, edge_index):
        # Graph convolutions with residual connections
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.3, training=self.training)

        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.3, training=self.training)
        h2 = h1 + h2

        h3 = self.conv3(h2, edge_index)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=0.3, training=self.training)
        h3 = h2 + h3

        h4 = self.conv4(h3, edge_index)
        h4 = F.relu(h4)
        h4 = F.dropout(h4, p=0.3, training=self.training)
        h4 = h3 + h4

        # Predict normalized positions and radius
        radius = self.radius_mlp(h4)
        coords = self.pos_mlp(h4)

        coords_norm = F.normalize(coords, p=2, dim=1)

        coords = coords_norm * radius

        return coords
        
        