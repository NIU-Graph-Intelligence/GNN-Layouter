import torch
import torch.nn.functional as F

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_coords = model(batch.x, batch.edge_index)
            loss = circular_layout_loss(pred_coords, batch.y, batch.x)
            total_loss += loss.item() * batch.num_nodes

    return total_loss / len(loader.dataset)

def circular_layout_loss(pred_coords, true_coords, x):
    """
    Enhanced loss function with better circular geometry constraints
    """
    # L2 coordinate loss
    coord_loss = F.mse_loss(pred_coords, true_coords)

    # Normalize predicted and true coordinates
    pred_norm = F.normalize(pred_coords, p=2, dim=1)
    true_norm = F.normalize(true_coords, p=2, dim=1)

    # Circular geometry loss
    circular_loss = F.mse_loss(pred_norm, true_norm)

    # Radius consistency
    pred_radii = torch.norm(pred_coords, dim=1)
    true_radii = torch.norm(true_coords, dim=1)
    radius_loss = F.mse_loss(pred_radii, true_radii)

    # Angular preservation loss
    pred_angles = torch.atan2(pred_coords[:, 1], pred_coords[:, 0])
    true_angles = torch.atan2(true_coords[:, 1], true_coords[:, 0])
    angle_loss = F.mse_loss(torch.sin(pred_angles), torch.sin(true_angles)) + \
                 F.mse_loss(torch.cos(pred_angles), torch.cos(true_angles))

    # Total loss with weighted components
    total_loss = coord_loss + \
                 0.3 * circular_loss + \
                 0.1 * radius_loss + \
                 0.1 * angle_loss

    return total_loss


