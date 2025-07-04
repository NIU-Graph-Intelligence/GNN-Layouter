import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
import torch.nn.functional as F
import torch
from models.mlp_layers import MLPFactory, ConvolutionalBlock, WeightInitializer
from models.coordinate_layers import PolarCoordinates, CartesianCoordinates


class GNN_Model_GIN(nn.Module):
    def __init__(self, input_dim, hidden_channels=128, num_layers=4, dropout=0.4):
        super().__init__()
        # input_dim = max_nodes
        self.dropout = dropout

        self.convs = nn.ModuleList()

        self.convs.append(GINConv(
            ConvolutionalBlock.create_gin_block(input_dim, hidden_channels, dropout),
            train_eps=True
        ))
        
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(
                ConvolutionalBlock.create_gin_block(hidden_channels, hidden_channels, dropout),
                train_eps=True
            ))
        
        self.attention = MLPFactory.create('attention', in_channels=hidden_channels)
        

        self.angle_mlp = MLPFactory.create('angle',
            in_channels=hidden_channels,
            hidden_channels=[hidden_channels * 2, hidden_channels],
            dropout_rate=0.5,
            use_layer_norm=True
        )

        self.radius_mlp = MLPFactory.create('radius',
            in_channels=hidden_channels,
            hidden_channels=[hidden_channels],
            dropout_rate=0.5,
            use_layer_norm=True,
            constrained_output=True  # For Sigmoid activation
        )

        # Replace manual weight init with WeightInitializer
        WeightInitializer.xavier_uniform(self, gain=0.5)

    def forward(self, x, edge_index):

        h = x

        # First layer - no residual due to dimension change
        h = F.relu(self.convs[0](h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Remaining layers with residual connections
        for i in range(1, len(self.convs)):
            h_new = F.relu(self.convs[i](h, edge_index))
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # Residual connection (same dimensions)
    

        attention_weights = self.attention(h)
        h_weighted = h * attention_weights
        
        # Circular coordinate prediction with separate angle and radius
        coords = PolarCoordinates.to_cartesian(
            self.angle_mlp(h_weighted), 
            self.radius_mlp(h_weighted),
            constrain_radius = True,
            radius_range=(0.9, 1.1)
        )
        coords = CartesianCoordinates.normalize_coordinates(coords, center=False)
        
        return coords
