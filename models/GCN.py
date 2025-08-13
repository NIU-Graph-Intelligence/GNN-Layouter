import torch.nn as nn
from torch_geometric.nn import GCN, GCNConv
import torch.nn.functional as F
from models.mlp_layers import MLPFactory, WeightInitializer
from models.coordinate_layers import CoordinateNormalizer

# Import config manager
try:
    from config_utils.config_manager import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_channels=None, dropout_rate=None):
        super().__init__()
        
        # Load config defaults if available
        if CONFIG_AVAILABLE:
            config = get_config()
            gcn_config = config.get_model_config('GCN')
            
            # Apply config defaults with fallback to hardcoded values
            hidden_channels = hidden_channels or gcn_config.get('hidden_channels', 192)
            dropout_rate = dropout_rate or gcn_config.get('dropout_rate', 0.3)
        else:
            # Fallback defaults if config not available
            hidden_channels = hidden_channels or 192
            dropout_rate = dropout_rate or 0.3
        
        # input_dim = max_nodes # Store for reference
        self.dropout_rate = dropout_rate

        # Add input projection layer to map input_dim -> hidden_channels
        self.input_proj = nn.Linear(input_dim, hidden_channels)

        # First conv layer takes one-hot encoded input
        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels),  # Input is one-hot encoded
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
            GCNConv(hidden_channels, hidden_channels),
        ])

        # Position MLP - outputs 2D coordinates
        self.pos_mlp = MLPFactory.create('position',
            in_channels=hidden_channels,
            hidden_channels=[hidden_channels, hidden_channels // 2],
            dropout_rate=0.4,
            use_layer_norm=False,  # Uses BatchNorm instead
            output_dim=2  # 2D coordinates
        )

        # Radius prediction
        self.radius_mlp = MLPFactory.create('radius',
            in_channels=hidden_channels,
            hidden_channels=[hidden_channels // 2],
            dropout_rate=0.3,
            use_layer_norm=False,  # Uses BatchNorm
            constrained_output=True
        )

        # Add weight initialization
        WeightInitializer.xavier_uniform(self)

    def forward(self, x, edge_index):
        # Ensure input is float
        x = x.float()
        

        # Graph convolutions with residual connections
        h = self.input_proj(x)
        for conv in self.convs:
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout_rate, training=self.training)
            h = h + h_new  # Residual connection
        
        # Predict normalized positions and radius
        coords = CoordinateNormalizer.normalize_with_radius(self.pos_mlp(h), self.radius_mlp(h), center = False)

        return coords
        
        