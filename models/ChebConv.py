import torch.nn as nn
from torch_geometric.nn import ChebConv
import torch.nn.functional as F
from models.mlp_layers import MLPFactory, WeightInitializer
from models.coordinate_layers import CoordinateNormalizer


class GNN_ChebConv(nn.Module):
    def __init__(self, input_dim, hidden_channels=64, num_layers=3, K=2, dropout=0.4, use_residual=True):

        super().__init__()

        self.dropout = dropout
        self.use_residual = use_residual

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(ChebConv(input_dim, hidden_channels, K=K))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(ChebConv(hidden_channels, hidden_channels, K=K))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.pos_mlp = MLPFactory.create('position',
            in_channels=hidden_channels,
            hidden_channels=[hidden_channels],
            dropout_rate=0.6,
            use_layer_norm=True,
            output_dim = 2
        )

        self.radius_mlp = MLPFactory.create('radius',
            in_channels=hidden_channels,
            hidden_channels=[hidden_channels // 2],
            dropout_rate=0.6,
            use_layer_norm=True,
            constrained_output = True
        )

        WeightInitializer.xavier_uniform(self)

    def forward(self, x, edge_index):
        h = x

        h = self.convs[0](h, edge_index)
        h = F.relu(self.bns[0](h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        for i in range(1, len(self.convs)):
            h_new = self.convs[i](h, edge_index)
            h_new = F.relu(self.bns[i](h_new))
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if self.use_residual:
                h_new = h_new + h
            h = h_new


        coords = CoordinateNormalizer.normalize_with_radius(self.pos_mlp(h), self.radius_mlp(h))
        return coords