# training/losses.py
import torch
import torch.nn.functional as F

def circular_layout_loss(pred_coords, true_coords):
    """
    Circular layout loss with multiple components for better circular structure.
    
    Args:
        pred_coords: Predicted coordinates [num_nodes, 2]
        true_coords: True coordinates [num_nodes, 2]
    """
    # Handle shape mismatch
    if pred_coords.shape[0] != true_coords.shape[0]:
        min_size = min(pred_coords.shape[0], true_coords.shape[0])
        pred_coords = pred_coords[:min_size]
        true_coords = true_coords[:min_size]
    
    batch_size = pred_coords.size(0)
    
    # 1. Basic coordinate loss
    coord_loss = F.mse_loss(pred_coords, true_coords)
    
    # 2. Radius consistency - all points should be at the same distance from center
    pred_radii = torch.norm(pred_coords, dim=1)
    target_radius = torch.ones_like(pred_radii)  # Unit circle
    radius_loss = F.mse_loss(pred_radii, target_radius)
    
    # 3. Angular distribution loss - ensure uniform angular distribution
    pred_angles = torch.atan2(pred_coords[:, 1], pred_coords[:, 0])
    pred_angles_sorted, _ = torch.sort(pred_angles)
    
    # Compute differences between consecutive angles
    angle_diffs = pred_angles_sorted[1:] - pred_angles_sorted[:-1]
    target_diff = 2 * torch.pi / batch_size
    target_diffs = torch.ones_like(angle_diffs) * target_diff
    spacing_loss = F.mse_loss(angle_diffs, target_diffs)
    
    # 4. Center constraint - ensure layout is centered at origin
    center_loss = torch.mean(torch.abs(torch.mean(pred_coords, dim=0)))
    
    # 5. Pairwise distance loss - maintain proper spacing
    pred_dists = torch.cdist(pred_coords, pred_coords)
    true_dists = torch.cdist(true_coords, true_coords)
    distance_loss = F.mse_loss(pred_dists, true_dists)
    
    # Combine losses with weights
    total_loss = (0.1 * coord_loss +
                  0.3 * radius_loss +
                  0.3 * spacing_loss +
                  0.2 * center_loss +
                  0.1 * distance_loss)
    
    return total_loss

def force_directed_loss(pred_coords, true_coords):
    """
    Force-directed layout loss (simple MSE).
    
    Args:
        pred_coords: Predicted coordinates [num_nodes, 2]
        true_coords: True coordinates [num_nodes, 2]
    """
    return F.mse_loss(pred_coords, true_coords)

def compute_loss_per_graph(pred_coords, true_coords, batch, loss_type='force_directed'):
    """
    Compute loss for batched graphs with different sizes.
    
    Args:
        pred_coords: Predicted coordinates [total_nodes_in_batch, 2]
        true_coords: True coordinates [total_nodes_in_batch, 2]
        batch: Batch tensor indicating which graph each node belongs to
        loss_type: 'circular' or 'force_directed'
    """
    losses = []
    
    for graph_id in batch.unique():
        mask = (batch == graph_id)
        pred_graph = pred_coords[mask]
        true_graph = true_coords[mask]
        
        if loss_type == 'circular':
            loss = circular_layout_loss(pred_graph, true_graph)
        elif loss_type == 'force_directed':
            loss = force_directed_loss(pred_graph, true_graph)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        losses.append(loss)
    
    return torch.stack(losses).mean()