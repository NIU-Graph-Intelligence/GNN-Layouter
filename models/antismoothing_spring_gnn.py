import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean, scatter_add


class AntiGCNConv(nn.Module):
    """Anti-smoothing convolution that pushes nodes away from neighbor averages"""
    def __init__(self, hidden_dim, alpha=0.5, use_edge_weight=False):
        super().__init__()
        self.alpha = alpha  # strength of anti-smoothing
        self.use_edge_weight = use_edge_weight
        
        # Transform features before anti-smoothing
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_transform = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable anti-smoothing strength
        self.anti_strength = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, x, edge_index, edge_weight=None):
        row, col = edge_index
        
        # Transform input features
        x_transformed = self.linear(x)
        
        # Compute neighbor messages
        neighbor_msgs = self.neighbor_transform(x_transformed[row])
        
        # Aggregate neighbor information
        if edge_weight is not None and self.use_edge_weight:
            neighbor_msgs = neighbor_msgs * edge_weight.view(-1, 1)
        
        # Average of neighbor features
        neighbor_mean = scatter_mean(neighbor_msgs, col, dim=0, dim_size=x.size(0))
        
        # Anti-smoothing: move away from neighbor average
        # The idea is to amplify differences rather than smooth them
        anti_smooth_output = x_transformed - torch.sigmoid(self.anti_strength) * neighbor_mean
        
        return anti_smooth_output


class AttractionBranch(nn.Module):
    """Standard GCN branch for attractive forces"""
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers for attraction (smoothing)
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x.float()))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Multi-layer GCN for smooth attractive forces
        for gcn in self.gcn_layers:
            h_new = F.relu(gcn(h, edge_index))
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new  # Residual connection
        
        # Final projection
        attraction_features = self.output_proj(h)
        return attraction_features


class RepulsionBranch(nn.Module):
    """Anti-GCN branch for repulsive forces"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        # Input projection  
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Anti-GCN layers for repulsion
        self.anti_gcn_layers = nn.ModuleList([
            AntiGCNConv(hidden_dim, alpha=0.3 + 0.2 * i) for i in range(num_layers)
        ])  # Increasing anti-smoothing strength in deeper layers
        
        # Feature enhancement between anti-GCN layers
        self.feature_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers - 1)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x.float()))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Multi-layer Anti-GCN for repulsive forces
        for i, anti_gcn in enumerate(self.anti_gcn_layers):
            h = anti_gcn(h, edge_index)
            h = F.relu(h)
            
            # Feature enhancement between layers
            if i < len(self.feature_enhancers):
                h = self.feature_enhancers[i](h)
        
        # Final projection
        repulsion_features = self.output_proj(h)
        return repulsion_features


class GlobalRepulsionBranch(nn.Module):
    """Global attention branch for long-range repulsive forces"""
    def __init__(self, input_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention for global interactions
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Post-attention processing
        self.post_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x.float()))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Global attention (all nodes attend to all nodes)
        h_input = h.unsqueeze(0)  # Add batch dimension: [1, N, hidden_dim]
        
        # Self-attention for global repulsive interactions
        h_attended, attention_weights = self.global_attention(h_input, h_input, h_input)
        h_attended = h_attended.squeeze(0).contiguous()  # Remove batch dimension: [N, hidden_dim]
        
        # Post-attention processing
        h_processed = self.post_attention(h_attended)
        
        # Residual connection
        h_global = h + h_processed
        
        # Final projection
        global_features = self.output_proj(h_global)
        return global_features


class ThreeWayFusion(nn.Module):
    """Fusion module to combine attraction, repulsion, and global features"""
    def __init__(self, hidden_dim, output_dim=2, dropout=0.1):
        super().__init__()
        
        # Individual branch processing
        self.attraction_head = nn.Linear(hidden_dim, hidden_dim // 2)
        self.repulsion_head = nn.Linear(hidden_dim, hidden_dim // 2)
        self.global_head = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Cross-branch interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Final coordinate prediction
        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3 // 2, hidden_dim),  # 3 * hidden_dim//2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(3))
        
    def forward(self, attraction_features, repulsion_features, global_features):
        # Process each branch
        h_attr = F.relu(self.attraction_head(attraction_features))
        h_repul = F.relu(self.repulsion_head(repulsion_features))
        h_global = F.relu(self.global_head(global_features))
        
        # Stack for cross-attention
        stacked_features = torch.stack([h_attr, h_repul, h_global], dim=1)  # [N, 3, hidden_dim//2]
        
        # Cross-branch attention
        attended_features, _ = self.cross_attention(stacked_features, stacked_features, stacked_features)
        
        # Apply learnable weights
        weights = F.softmax(self.fusion_weights, dim=0)
        weighted_features = attended_features * weights.view(1, 3, 1)  # Broadcast weights
        
        # Ensure tensor is contiguous before reshaping
        weighted_features = weighted_features.contiguous()
        
        # Flatten and concatenate
        fused_features = weighted_features.reshape(weighted_features.size(0), -1)  # [N, 3*hidden_dim//2]
        
        # Predict final coordinates
        coordinates = self.coord_predictor(fused_features)
        return coordinates


class AntiSmoothingSpringGNN(nn.Module):
    """Anti-smoothing GNN that explicitly models attractive and repulsive forces"""
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        
        # Three branches for different types of forces
        self.attraction_branch = AttractionBranch(input_dim, hidden_dim, dropout=dropout)
        self.repulsion_branch = RepulsionBranch(input_dim, hidden_dim, dropout=dropout)
        self.global_repulsion_branch = GlobalRepulsionBranch(input_dim, hidden_dim, dropout=dropout)
        
        # Fusion module
        self.fusion_module = ThreeWayFusion(hidden_dim, output_dim=2, dropout=dropout)
        
        # Final coordinate normalization
        self.coord_norm = nn.LayerNorm(2)
        
    def forward(self, x, edge_index):
        # Process each branch
        attraction_features = self.attraction_branch(x, edge_index)
        repulsion_features = self.repulsion_branch(x, edge_index) 
        global_features = self.global_repulsion_branch(x, edge_index)
        
        # Fuse all branches and predict coordinates
        coordinates = self.fusion_module(attraction_features, repulsion_features, global_features)
        
        # Normalize coordinates
        coordinates = self.coord_norm(coordinates)
        
        return coordinates