import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, max_nodes, hidden_channels=64, heads=8):  # Reduced hidden_channels, increased heads
        """
        Args:
            max_nodes (int): Maximum number of nodes. Used to define input feature dimension.
            hidden_channels (int): Dimensionality for hidden representations.
            heads (int): Number of attention heads.
        """
        super().__init__()

        # Original input feature dimension.
        # Optionally, if you have additional features (e.g., degree or Laplacian eigenvectors),
        # you can increase this dimension accordingly.
        input_dim = max_nodes + 1
        # input_dim = max_nodes

        head_dim = hidden_channels // heads

        # Simplified architecture with fewer layers but more heads
        self.conv1 = GATConv(input_dim, head_dim, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_channels, head_dim, heads=heads, dropout=0.2)
        self.conv3 = GATConv(hidden_channels, head_dim, heads=heads, dropout=0.2)

        # Layer norms after each convolution
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)

        # Angle prediction branch (new)
        self.angle_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )

        # Radius prediction branch
        self.radius_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus()
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, edge_index):
        # Initial node features
        h1 = self.norm1(self.conv1(x, edge_index))
        h1 = F.relu(h1)

        # Second layer with residual
        h2 = self.norm2(self.conv2(h1, edge_index))
        h2 = F.relu(h2) + h1

        # Third layer with residual
        h3 = self.norm3(self.conv3(h2, edge_index))
        h3 = F.relu(h3) + h2

        # Modified coordinate prediction
        h3 = F.normalize(h3, p=2, dim=1)  # Normalize features

        # Predict angles (ensure full range coverage)
        angles = self.angle_mlp(h3) * torch.pi  # Scale to [-π, π]
        
        # Predict radius deviation from unit circle
        radius_dev = self.radius_mlp(h3)
        radius = 1.0 + 0.1 * torch.tanh(radius_dev)  # Constrain radius near 1
        
        # Convert to coordinates
        coords = torch.zeros(x.size(0), 2, device=x.device)
        coords[:, 0] = radius.squeeze() * torch.cos(angles.squeeze())
        coords[:, 1] = radius.squeeze() * torch.sin(angles.squeeze())
        
        # Ensure the layout is centered
        coords = coords - coords.mean(dim=0, keepdim=True)
        
        # Scale to maintain unit circle
        coords = F.normalize(coords, p=2, dim=1)
        
        return coords

