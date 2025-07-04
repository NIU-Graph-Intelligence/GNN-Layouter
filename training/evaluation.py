import torch
import torch.nn.functional as F
import traceback
from torch.nn import MSELoss


def evaluate(model, loader, device, loss_type='circular'):

    if loss_type == 'circular':
        return evaluate_circular(model, loader, device)
    elif loss_type == 'forceGNN':
        return evaluate_forcegnn(model, loader, device)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")



def forceGNN_loss(pred_coords, true):

    batch_size = pred_coords.shape[0]
    nodes_per_graph = pred_coords.shape[1]
    
    # Reshape predictions to match target shape
    pred_coords_flat = pred_coords.reshape(-1, 2)
    
    criterion = MSELoss()
    loss = criterion(pred_coords_flat, true)
    return loss


def evaluate_forcegnn(model, loader, device):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in loader:
            try:
                batch = batch.to(device)
                # Model takes the entire batch and returns [batch_size, nodes_per_graph, 2]
                pred_coords = model(batch.x, batch.edge_index, batch.batch, batch.init_coords)
                
                # Calculate force residual loss with reshaped predictions
                loss = forceGNN_loss(pred_coords, batch.original_y)

                total_loss += loss.item()
                count += 1
            except Exception as e:
                print(f"Error evaluating batch: {str(e)}")
                traceback.print_exc()  # Print full traceback for debugging
                continue

    return total_loss / count if count > 0 else float('inf')



def evaluate_circular(model, loader, device):

    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in loader:
            try:
                batch = batch.to(device)
                pred_coords = model(batch.x, batch.edge_index)
                loss = circular_layout_loss(pred_coords, batch.y, batch.x)
                total_loss += loss.item()
                count += 1
            except Exception as e:
                print(f"Error evaluating batch: {str(e)}")
                continue

    return total_loss / count if count > 0 else float('inf')


def circular_layout_loss(pred_coords, true_coords, x):

    # Check for shape mismatch and fix if needed
    if pred_coords.shape[0] != true_coords.shape[0]:
        print(f"Warning: Shape mismatch in circular_layout_loss. Pred: {pred_coords.shape}, True: {true_coords.shape}")
        # If pred has more nodes than true, trim pred to match true
        if pred_coords.shape[0] > true_coords.shape[0]:
            pred_coords = pred_coords[:true_coords.shape[0]]
        # If true has more nodes than pred, trim true to match pred
        else:
            true_coords = true_coords[:pred_coords.shape[0]]

    batch_size = pred_coords.size(0)

    # 1. Basic coordinate loss (reduced weight)
    coord_loss = F.mse_loss(pred_coords, true_coords)

    # 2. Radius consistency - all points should be at the same distance from center
    pred_radii = torch.norm(pred_coords, dim=1)
    target_radius = torch.ones_like(pred_radii)  # We want unit circle
    radius_loss = F.mse_loss(pred_radii, target_radius)

    # 3. Angular distribution loss - ensure uniform angular distribution
    pred_angles = torch.atan2(pred_coords[:, 1], pred_coords[:, 0])
    pred_angles_sorted, _ = torch.sort(pred_angles)

    # Compute the differences between consecutive angles
    angle_diffs = pred_angles_sorted[1:] - pred_angles_sorted[:-1]
    target_diff = 2 * torch.pi / batch_size  # Expected angle difference for uniform distribution
    target_diffs = torch.ones_like(angle_diffs) * target_diff
    spacing_loss = F.mse_loss(angle_diffs, target_diffs)

    # 4. Center constraint - ensure the layout is centered at origin
    center_loss = torch.mean(torch.abs(torch.mean(pred_coords, dim=0)))

    # 5. Pairwise distance loss - maintain proper spacing between nodes
    pred_dists = torch.cdist(pred_coords, pred_coords)
    true_dists = torch.cdist(true_coords, true_coords)
    distance_loss = F.mse_loss(pred_dists, true_dists)

    # Combine losses with appropriate weights
    total_loss = (0.1 * coord_loss +
                  0.3 * radius_loss +
                  0.3 * spacing_loss +
                  0.2 * center_loss +
                  0.1 * distance_loss)

    return total_loss

