import torch.nn as nn
from torch_geometric.nn import ChebConv
import torch.nn.functional as F


class GNN_ChebConv(nn.Module):
    def __init__(self, max_nodes, hidden_channels=64):
        super().__init__()
        # input_dim = max_nodes + 1
        input_dim = max_nodes # Without Positional Index

        self.conv1 = ChebConv(input_dim, hidden_channels, K=2)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=2)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = ChebConv(hidden_channels, hidden_channels, K=2)
        self.bn3 = nn.BatchNorm1d(hidden_channels)

        self.pos_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_channels, 2)
        )

        self.radius_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, edge_index):
        h1 = self.bn1(F.relu(self.conv1(x, edge_index)))
        h1 = F.dropout(h1, p=0.4, training=self.training)

        h2 = self.bn2(F.relu(self.conv2(h1, edge_index)))
        h2 = F.dropout(h2, p=0.4, training=self.training)
        h2 = h2 + h1

        h3 = self.bn3(F.relu(self.conv3(h2, edge_index)))
        h3 = F.dropout(h3, p=0.4, training=self.training)
        h3 = h3 + h2

        radius = self.radius_mlp(h3)
        coords = F.normalize(self.pos_mlp(h3), p=2, dim=1) * radius
        return coords

