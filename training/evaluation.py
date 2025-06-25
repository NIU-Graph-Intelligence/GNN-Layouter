import torch
import torch.nn.functional as F
import traceback
from torch.nn import MSELoss


def evaluate(model, loader, device, loss_type='circular'):
    """
    Router function that calls the appropriate evaluation function based on loss_type
    """
    if loss_type == 'circular':
        return evaluate_circular(model, loader, device)
    elif loss_type == 'fr':
        return evaluate_fr(model, loader, device)
    elif loss_type == 'fa2':
        return evaluate_fa2(model, loader, device)
    elif loss_type == 'forceGNN':
        return evaluate_forcegnn(model, loader, device)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# def normalize(z, eps=1e-6):
#     """
#     Center at zero mean and scale to unit std along each axis.
#     """
#     z = z - z.mean(dim=0, keepdim=True)
#     z = z / (z.std(dim=0, keepdim=True) + eps)
#     return z
#
# def procrustes_loss(pred, true, eps=1e-6):
#     """
#     1 - (trace(pred^T true)^2) / (trace(pred^T pred) * trace(true^T true))
#     Both pred & true should be zero-mean and unit-variance.
#     """
#     # [B is 2 dims here]
#     num = torch.trace(pred.T @ true).pow(2)
#     den = (torch.trace(pred.T @ pred) * torch.trace(true.T @ true)).clamp(min=eps)
#     return 1 - num / den
#
# def structure_loss(pred, true, α=1.0, β=10.0, γ=0, δ=1.0):
#     """
#     pred, true:  [N,2] raw coords output / ground-truth original_y
#     α: coord MSE weight
#     β: stress / pairwise MSE weight
#     γ: tiny repulsion weight
#     δ: Procrustes alignment weight
#     """
#     # 1) normalize both to focus only on shape of the embedding
#     p = normalize(pred)
#     t = normalize(true)
#
#     # 2) coordinate MSE
#     l_coord  = F.mse_loss(p, t)
#
#     # 3) stress = pairwise distance MSE
#     dp = torch.cdist(p, p)  # [N,N]
#     dt = torch.cdist(t, t)
#     l_stress = F.mse_loss(dp, dt)
#
#     # 4) tiny repulsion to avoid collapse
#     #    encourage distances > 0
#     # d = dp + 1e-6
#     # l_repel = - (torch.log(d).mean())
#     l_repel = 0
#
#     # 5) Procrustes alignment (rotation/reflection)
#     l_proc   = procrustes_loss(p, t)
#
#     return (
#         α * l_coord
#       + β * l_stress
#       + γ * l_repel
#       + δ * l_proc,
#       l_coord, l_stress, l_repel, l_proc
#     )


def fr_force_residual_loss(pred_pos, true_pos, lambda_coord = 1.0, lambda_pairwise=0.5):

    coord_loss = F.mse_loss(pred_pos, true_pos)

    # init_loss = 0.0
    # if init_coords is not None:
    #     init_loss = F.mse_loss(pred_pos, init_coords)
    #
    pairwise_loss = 0
    if true_pos is not None:
        pred_dist = torch.cdist(pred_pos, pred_pos)
        true_dist = torch.cdist(true_pos, true_pos)
        pairwise_loss = F.mse_loss(pred_dist, true_dist)

    total_loss = lambda_coord * coord_loss  + lambda_pairwise * pairwise_loss
#
    # return coord_loss
    return total_loss


#     # 1) coordinate MSE
#     l_coord = F.mse_loss(pred, true)
#
#     # 2) stress
#     pdist_pred = torch.cdist(pred, pred)
#     pdist_true = torch.cdist(true, true)
#     l_stress = F.mse_loss(pdist_pred, pdist_true)
#
#     # 3) tiny repulsion to keep things apart
#     d = pdist_pred + 1e-6
#     l_repel = -(torch.log(d).mean())
#
#     return α * l_coord + β * l_stress + γ * l_repel

def forceGNN_loss(pred_coords, true):
    """
    Compute MSE loss between predicted and true coordinates.
    
    Args:
        pred_coords: Tensor of shape [batch_size, nodes_per_graph, 2]
        true: Tensor of shape [batch_size * nodes_per_graph, 2]
    """
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


def fa2_force_residual_loss(pred_pos, true_pos, lambda_coord = 1.0, lambda_pairwise=0.5):
    coord_loss = F.mse_loss(pred_pos, true_pos)
    pairwise_loss = 0
    if true_pos is not None:
        pred_dist = torch.cdist(pred_pos, pred_pos)
        true_dist = torch.cdist(true_pos, true_pos)
        pairwise_loss = F.mse_loss(pred_dist, true_dist)

    total_loss = lambda_coord * coord_loss + lambda_pairwise * pairwise_loss

    return total_loss


def evaluate_fa2(model, loader, device):
    """
    Evaluate average FA2‐force‐residual loss over dataset.
    """
    model.eval()
    total, count = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            # remove any global drift
            pred_centered = pred - pred.mean(dim=0, keepdim=True)

            loss = fa2_force_residual_loss(pred_centered, batch.original_y,)
            total += loss.item()
            count += 1

    return total / count if count else float('inf')


def evaluate_fr(model, loader, device):
    
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in loader:
            try:
                batch = batch.to(device)
                pred_coords = model(batch)  # Model takes the entire batch
                
                # Center the predicted coordinates
                pred_centered = pred_coords - pred_coords.mean(dim=0, keepdim=True)

                # Calculate force residual loss
                loss = fr_force_residual_loss(pred_centered, batch.original_y)

                total_loss += loss.item()
                count += 1
            except Exception as e:
                print(f"Error evaluating batch: {str(e)}")
                traceback.print_exc()  # Print full traceback for debugging
                continue

    return total_loss / count if count > 0 else float('inf')




def evaluate_circular(model, loader, device):
    """
    Evaluation function specifically for circular layout
    """
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
    """
    Enhanced loss function with stronger geometric constraints for circular layout
    """
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

