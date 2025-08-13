import torch.nn as nn
from torch_geometric.nn import ChebConv
import torch.nn.functional as F
from models.mlp_layers import MLPFactory, WeightInitializer
from models.coordinate_layers import CoordinateNormalizer

# Try to import config, but don't fail if not available (for backwards compatibility)
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from config_utils.config_manager import get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False


class GNN_ChebConv(nn.Module):
    def __init__(self, input_dim, hidden_channels=None, num_layers=None, K=None, dropout=None, use_residual=None):

        super().__init__()

        # Load config defaults if available
        if _CONFIG_AVAILABLE:
            try:
                config = get_config()
                model_config = config.get_model_config('ChebConv')
                hidden_channels = hidden_channels or model_config.get('hidden_channels', 64)
                num_layers = num_layers or model_config.get('num_layers', 3)
                K = K or model_config.get('K', 2)
                dropout = dropout or model_config.get('dropout', 0.4)
                use_residual = use_residual if use_residual is not None else model_config.get('use_residual', True)
            except Exception:
                # If config loading fails, use provided values or defaults
                hidden_channels = hidden_channels or 64
                num_layers = num_layers or 3
                K = K or 2
                dropout = dropout or 0.4
                use_residual = use_residual if use_residual is not None else True
        else:
            # Fallback defaults if config not available
            hidden_channels = hidden_channels or 64
            num_layers = num_layers or 3
            K = K or 2
            dropout = dropout or 0.4
            use_residual = use_residual if use_residual is not None else True

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