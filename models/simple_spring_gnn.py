import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AttractionGCN(nn.Module):
    """Handles attractive forces on edges using GCN smoothing property"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))
        
        self.force_head = nn.Linear(hidden_dim, 2)
    
    def forward(self, x, edge_index):
        h = x.float()
        
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        attraction_force = self.force_head(h)
        return attraction_force


class RepulsionNet(nn.Module):
    """Handles global repulsive forces using simplified global attention"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.force_head = nn.Linear(hidden_dim, 2)
    
    def forward(self, x, edge_index):
        h = self.node_proj(x.float())
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        Q = self.query_proj(h)
        K = self.key_proj(h)
        V = self.value_proj(h)
        
        attention_scores = torch.mm(Q, K.t()) / (self.hidden_dim ** 0.5)
        repulsion_weights = F.softmax(attention_scores, dim=-1)
        repulsion_features = torch.mm(repulsion_weights, V)
        
        repulsion_force = self.force_head(repulsion_features)
        return repulsion_force


class ForceBalanceNet(nn.Module):
    """Learns how to balance attractive and repulsive forces, outputs final coordinates"""
    def __init__(self, input_dim, output_dim=2, hidden_dim=64, dropout=0.1):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, combined_forces):
        coordinates = self.mlp(combined_forces)
        return coordinates


class SimpleIterativeGNN(nn.Module):
    """Simple iterative GNN with learnable steps"""
    def __init__(self, input_dim, hidden_dim=64, num_steps=3, dropout=0.1):
        super().__init__()
        self.num_steps = num_steps
        self.input_dim = input_dim
        
        # Each step has its own GCN + MLP
        self.step_gcns = nn.ModuleList([
            GCNConv(input_dim, hidden_dim) for _ in range(num_steps)
        ])
        
        self.step_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)
            ) for _ in range(num_steps)
        ])
        
        # Learnable step size for each iteration
        self.step_sizes = nn.Parameter(torch.ones(num_steps) * 0.1)
        
        # Optional: shared layers for efficiency
        self.use_shared_layers = False
        if self.use_shared_layers:
            self.shared_gcn = GCNConv(input_dim, hidden_dim)
            self.shared_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)
            )
    
    def forward(self, x, edge_index):
        # Extract initial coordinates
        coords = x[:, -2:].clone()
        current_x = x.clone()
        
        for step in range(self.num_steps):
            if self.use_shared_layers:
                # Use shared layers
                h = F.relu(self.shared_gcn(current_x, edge_index))
                coord_update = self.shared_mlp(h)
            else:
                # Use step-specific layers
                h = F.relu(self.step_gcns[step](current_x, edge_index))
                coord_update = self.step_mlps[step](h)
            
            # Apply learnable step size
            coords = coords + self.step_sizes[step] * coord_update
            
            # Update feature vector with new coordinates
            current_x = torch.cat([current_x[:, :-2], coords], dim=-1)
        
        return coords


class SimpleSpringGNN(nn.Module):
    """Main model: Simple iterative approach with GCN fallback"""
    def __init__(self, input_dim, hidden_dim=64, num_steps=3, dropout=0.1):
        super().__init__()
        
        # Simple iterative component
        self.iterative_model = SimpleIterativeGNN(input_dim, hidden_dim, num_steps, dropout)
        
        # Simple GCN baseline
        self.gcn_baseline = nn.Sequential(
            GCNConv(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            GCNConv(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            GCNConv(hidden_dim, 2)
        )
        
        # Learnable blending weight
        self.blend_weight = nn.Parameter(torch.tensor(0.5))
        
        # Whether to use blending or just iterative
        self.use_blending = True
    
    def forward(self, x, edge_index):
        # Iterative approach
        iterative_coords = self.iterative_model(x, edge_index)
        
        if not self.use_blending:
            return iterative_coords
        
        # GCN baseline
        h = x
        for i, layer in enumerate(self.gcn_baseline):
            if isinstance(layer, GCNConv):
                h = layer(h, edge_index)
            else:
                h = layer(h)
        gcn_coords = h
        
        # Add residual connection for GCN
        gcn_coords = gcn_coords + x[:, -2:]
        
        # Blend the two approaches
        alpha = torch.sigmoid(self.blend_weight)
        final_coords = alpha * iterative_coords + (1 - alpha) * gcn_coords
        
        return final_coords
    
    def set_blending(self, use_blending):
        """Enable/disable blending with GCN baseline"""
        self.use_blending = use_blending
    
    def get_step_sizes(self):
        """Get current step sizes for debugging"""
        return self.iterative_model.step_sizes.data


# Training utilities
def create_model(input_dim=4, hidden_dim=64, num_steps=3, dropout=0.1):
    """Factory function to create model"""
    return SimpleSpringGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        dropout=dropout
    )


def training_step(model, batch, criterion, optimizer, clip_grad=True):
    """Single training step with gradient clipping"""
    x, edge_index, target_coords = batch
    
    # Forward pass
    pred_coords = model(x, edge_index)
    
    # Compute loss
    loss = criterion(pred_coords, target_coords)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping for stability
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


def progressive_training(model, train_loader, val_loader, num_epochs=100):
    """Progressive training: start with fewer steps, then increase"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Start with 1 step, gradually increase
    original_steps = model.iterative_model.num_steps
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Progressive steps: increase every 20 epochs
        current_steps = min(1 + epoch // 20, original_steps)
        model.iterative_model.num_steps = current_steps
        
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            loss = training_step(model, batch, criterion, optimizer)
            train_losses.append(loss)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, edge_index, target_coords = batch
                pred_coords = model(x, edge_index)
                val_loss = criterion(pred_coords, target_coords).item()
                val_losses.append(val_loss)
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if epoch % 10 == 0:
            step_sizes = model.get_step_sizes()
            print(f"Epoch {epoch:3d} | Steps: {current_steps} | "
                  f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                  f"Step sizes: {step_sizes[:3].tolist()}")
    
    # Restore original number of steps
    model.iterative_model.num_steps = original_steps
    
    return best_val_loss


def simple_training(model, train_loader, val_loader, num_epochs=100):
    """Simple training loop"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            loss = training_step(model, batch, criterion, optimizer)
            train_losses.append(loss)
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    x, edge_index, target_coords = batch
                    pred_coords = model(x, edge_index)
                    val_loss = criterion(pred_coords, target_coords).item()
                    val_losses.append(val_loss)
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)
            
            print(f"Epoch {epoch:3d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")


# Example usage
if __name__ == "__main__":
    # Create model
    model = create_model(input_dim=4, hidden_dim=64, num_steps=3)
    
    # Print model info
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Step sizes: {model.get_step_sizes()}")
    
    # Test forward pass
    x = torch.randn(10, 4)  # 10 nodes, 4 features (last 2 are coordinates)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    with torch.no_grad():
        output = model(x, edge_index)
        print(f"Input coords shape: {x[:, -2:].shape}")
        print(f"Output coords shape: {output.shape}")
        print(f"Coordinate change: {torch.norm(output - x[:, -2:], dim=1).mean().item():.6f}")