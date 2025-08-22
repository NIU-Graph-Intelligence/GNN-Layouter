import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class LocalForceNet(nn.Module):
    """Local force computation using 1-hop neighbors"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        # Process node features and initial coordinates
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.coord_proj = nn.Linear(2, hidden_dim // 4)  # initial coordinates
        
        # GCN for local message passing
        self.gcn = GCNConv(hidden_dim + hidden_dim // 4, hidden_dim)
        
        # Output local force vector
        self.force_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, x, edge_index, initial_coords):
        # Combine node features with coordinate information
        h_node = self.node_proj(x.float())
        h_coord = self.coord_proj(initial_coords.float())
        h_combined = torch.cat([h_node, h_coord], dim=-1)
        
        # Local message passing
        h = self.gcn(h_combined, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Output local force
        local_force = self.force_head(h)
        return local_force


class MediumForceNet(nn.Module):
    """Medium-range force computation using 2-3 hop neighbors"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.coord_proj = nn.Linear(2, hidden_dim // 4)
        
        # Multi-hop GCN layers
        self.gcn1 = GCNConv(hidden_dim + hidden_dim // 4, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        
        # Distance-aware attention for medium range
        self.distance_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        self.force_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, x, edge_index, initial_coords):
        # Combine features
        h_node = self.node_proj(x.float())
        h_coord = self.coord_proj(initial_coords.float())
        h_combined = torch.cat([h_node, h_coord], dim=-1)
        
        # Multi-hop message passing (2-3 hops)
        h = F.relu(self.gcn1(h_combined, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = F.relu(self.gcn2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = F.relu(self.gcn3(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Distance-aware attention for medium-range interactions
        h_attended, _ = self.distance_attention(h.unsqueeze(1), h.unsqueeze(1), h.unsqueeze(1))
        h_attended = h_attended.squeeze(1)
        
        # Output medium-range force
        medium_force = self.force_head(h_attended)
        return medium_force


class GlobalForceNet(nn.Module):
    """Global force computation with top-k sampling for efficiency"""
    def __init__(self, input_dim, hidden_dim, top_k=32, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.coord_proj = nn.Linear(2, hidden_dim // 4)
        
        # Global attention components
        self.query_proj = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        
        # Distance encoding for global interactions
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        self.force_head = nn.Linear(hidden_dim, 2)
        
    def forward(self, x, edge_index, initial_coords):
        batch_size = x.size(0)
        
        # Combine features
        h_node = self.node_proj(x.float())
        h_coord = self.coord_proj(initial_coords.float())
        h_combined = torch.cat([h_node, h_coord], dim=-1)
        
        # Project to query, key, value
        Q = self.query_proj(h_combined)
        K = self.key_proj(h_combined) 
        V = self.value_proj(h_combined)
        
        # Compute pairwise distances
        coords = initial_coords.float()
        pairwise_dist = torch.cdist(coords, coords, p=2)  # [N, N]
        
        # Top-k sampling for efficiency
        global_forces = []
        for i in range(batch_size):
            # Get top-k nearest neighbors for node i
            distances_i = pairwise_dist[i]  # [N]
            _, top_k_indices = torch.topk(distances_i, min(self.top_k, batch_size), largest=False)
            
            # Compute attention only with top-k neighbors
            q_i = Q[i:i+1]  # [1, hidden_dim]
            k_topk = K[top_k_indices]  # [top_k, hidden_dim]
            v_topk = V[top_k_indices]  # [top_k, hidden_dim]
            
            # Distance-based attention weights
            dist_topk = distances_i[top_k_indices].unsqueeze(-1)  # [top_k, 1]
            dist_encoding = self.distance_encoder(dist_topk)  # [top_k, hidden_dim//4]
            
            # Compute attention scores with distance bias
            attention_scores = torch.mm(q_i, k_topk.t()) / (self.hidden_dim ** 0.5)  # [1, top_k]
            
            # Apply distance-based weighting (closer nodes have stronger influence)
            distance_weights = 1.0 / (dist_topk.squeeze(-1) + 1e-6)  # [top_k]
            attention_scores = attention_scores + distance_weights.unsqueeze(0)
            
            attention_weights = F.softmax(attention_scores, dim=-1)  # [1, top_k]
            
            # Aggregate global information
            global_info = torch.mm(attention_weights, v_topk)  # [1, hidden_dim]
            global_forces.append(global_info)
        
        # Stack all global forces
        global_features = torch.cat(global_forces, dim=0)  # [N, hidden_dim]
        global_features = F.dropout(global_features, p=self.dropout, training=self.training)
        
        # Output global force
        global_force = self.force_head(global_features)
        return global_force


class ForceCombiner(nn.Module):
    """Learns how to combine local, medium, and global forces"""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        
        # Force combination network
        self.force_combiner = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 3 forces * 2D each = 6
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Final 2D coordinates
        )
        
        # Learnable force weights
        self.force_weights = nn.Parameter(torch.ones(3))  # weights for 3 forces
        
    def forward(self, f_local, f_medium, f_global):
        # Apply learnable weights
        weights = F.softmax(self.force_weights, dim=0)
        
        weighted_local = weights[0] * f_local
        weighted_medium = weights[1] * f_medium  
        weighted_global = weights[2] * f_global
        
        # Concatenate all forces
        combined_forces = torch.cat([weighted_local, weighted_medium, weighted_global], dim=-1)
        
        # Learn final coordinate update
        coordinates = self.force_combiner(combined_forces)
        return coordinates


class MultiScaleSpringGNN(nn.Module):
    """Multi-scale spring layout GNN with different force ranges"""
    def __init__(self, input_dim, hidden_dim=64, top_k=32, dropout=0.1):
        super().__init__()
        
        # Calculate node feature dimension (input_dim - 2 for initial coords)
        self.node_feature_dim = max(1, input_dim - 2)  # At least 1 dimension
        
        # Different scale force networks
        self.local_force = LocalForceNet(self.node_feature_dim, hidden_dim, dropout)      # 1-hop neighbors
        self.medium_force = MediumForceNet(self.node_feature_dim, hidden_dim, dropout)    # 2-3 hop neighbors  
        self.global_force = GlobalForceNet(self.node_feature_dim, hidden_dim, top_k, dropout)  # Global with top-k
        
        # Force combination module
        self.force_combiner = ForceCombiner(hidden_dim, dropout)
                
    def forward(self, x, edge_index):
        # Extract initial coordinates from last 2 dimensions
        if x.size(1) < 2:
            raise ValueError(f"Input features must have at least 2 dimensions for initial coordinates, got {x.size(1)}")
        
        initial_coords = x[:, -2:]  # Last 2 columns
        node_features = x[:, :-2] if x.size(1) > 2 else torch.zeros(x.size(0), 1, device=x.device)  # All except last 2 columns
        
        # Compute forces at different scales
        f_local = self.local_force(node_features, edge_index, initial_coords)
        f_medium = self.medium_force(node_features, edge_index, initial_coords) 
        f_global = self.global_force(node_features, edge_index, initial_coords)
        
        # Combine forces and predict final coordinates
        coordinates = self.force_combiner(f_local, f_medium, f_global)
        
        # No normalization
        return coordinates