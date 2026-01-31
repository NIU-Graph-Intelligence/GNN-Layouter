
# training/evaluation.py
import torch
from .losses import compute_loss_per_graph

def evaluate_model(model, data_loader, device, loss_type='force_directed'):
    """
    Evaluate model on given data loader.
    
    Args:
        model: PyTorch model
        data_loader: PyTorch Geometric DataLoader
        device: torch.device
        loss_type: 'circular' or 'force_directed'
    
    Returns:
        Average loss across all batches
    """
    model.eval()
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for batch in data_loader:
            try:
                batch = batch.to(device)
                
                # Forward pass
                pred_coords = model(batch.x, batch.edge_index)
                
                # Compute loss
                loss = compute_loss_per_graph(
                    pred_coords, 
                    batch.y,  # Use normalized coordinates
                    batch.batch, 
                    loss_type
                )
                
                total_loss += loss.item()
                count += 1
                
            except Exception as e:
                print(f"Error evaluating batch: {str(e)}")
                continue
    
    return total_loss / count if count > 0 else float('inf')

def compute_metrics(model, data_loader, device):
    """
    Compute additional metrics for analysis.
    
    Args:
        model: PyTorch model
        data_loader: PyTorch Geometric DataLoader
        device: torch.device
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    mse_errors = []
    mae_errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            try:
                batch = batch.to(device)
                pred_coords = model(batch.x, batch.edge_index)
                
                # Compute per-graph metrics
                for graph_id in batch.batch.unique():
                    mask = (batch.batch == graph_id)
                    pred_graph = pred_coords[mask]
                    true_graph = batch.y[mask]
                    
                    mse = torch.mean((pred_graph - true_graph) ** 2).item()
                    mae = torch.mean(torch.abs(pred_graph - true_graph)).item()
                    
                    mse_errors.append(mse)
                    mae_errors.append(mae)
                    
            except Exception as e:
                continue
    
    if not mse_errors:
        return {"mse": float('inf'), "mae": float('inf')}
    
    return {
        "mse": sum(mse_errors) / len(mse_errors),
        "mae": sum(mae_errors) / len(mae_errors),
        "mse_std": torch.tensor(mse_errors).std().item(),
        "mae_std": torch.tensor(mae_errors).std().item()
    }