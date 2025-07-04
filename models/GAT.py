import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from models.mlp_layers import MLPFactory, ConvolutionalBlock, WeightInitializer
from models.coordinate_layers import PolarCoordinates, CartesianCoordinates


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_channels=64, heads=8, num_layers=3, dropout=0.2):
        
        super().__init__()

        # Input dimension setup
        self.dropout = dropout
        self.heads = heads
        self.num_layers = num_layers

        head_dim = hidden_channels // heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        conv, norm = ConvolutionalBlock.create_attention_block(
            input_dim,head_dim, heads, dropout
        )
        self.convs.append(conv)
        self.norms.append(norm)

        # Hidden GAT layers (hidden_channels to hidden_channels)
        for _ in range(num_layers - 1):
            conv, norm = ConvolutionalBlock.create_attention_block(
                hidden_channels,head_dim, heads, dropout
            )
            self.convs.append(conv)
            self.norms.append(norm)

        # Angle prediction using modular MLP
        self.angle_mlp = MLPFactory.create(
            'angle',
            in_channels=hidden_channels,
            hidden_channels=[hidden_channels],
            dropout_rate=0.2,
            use_layer_norm=True
        )

        # Radius prediction using modular MLP
        self.radius_mlp = MLPFactory.create(
            'radius',
            in_channels=hidden_channels,
            hidden_channels=[hidden_channels // 2],
            dropout_rate=0.2,
            use_layer_norm=True,
            constrained_output=True  # Uses Sigmoid for constrained output
        )

        WeightInitializer.xavier_uniform(self)

    def forward(self, x, edge_index):
       
        h = x
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index)
            h_new = self.norms[i](h_new)
            h_new = F.relu(h_new)
            if i > 0:
                h_new = h_new + h
            else:
                h = h_new

        h = F.normalize(h, p=2, dim=1)

        # Predict angles and radius
        angles = self.angle_mlp(h)
        radius = self.radius_mlp(h)

        # Convert to Cartesian coordinates using modular components
        coords = PolarCoordinates.to_cartesian(
            angles=angles,
            radius=radius,
            constrain_radius=True,
            radius_range=(0.9, 1.1)  # GAT's specific radius constraints
        )

        # Final normalization
        coords = CartesianCoordinates.normalize_coordinates(coords, center=True)
        
        return coords

