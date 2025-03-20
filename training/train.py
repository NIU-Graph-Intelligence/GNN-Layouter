import torch
import torch.nn.functional as F
from .evaluation import evaluate, circular_layout_loss
# from visualization import *
# import sys
import os
# from ..main import *
import matplotlib.pyplot as plt
import pickle

def train_model(model, train_loader, val_loader, batch_size, num_epochs=2000, lr=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Model type: {type(model)}")
    print(f"Model: {model}")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        min_lr=1e-5,
        verbose=True
    )
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 300

    # Lists to store metrics for plotting
    epochs_list = []
    train_losses = []
    val_losses = []
    learning_rates = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Train on individual samples
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred_coords = model(batch.x, batch.edge_index)
            loss = circular_layout_loss(pred_coords, batch.y, batch.x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)

        # Store metrics for plotting
        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # Pass val_loss to scheduler.step()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            try:
                current_dir = os.getcwd()
                save_dir = os.path.join(current_dir, 'results', 'metrics')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'best_model_{batch_size}.pt')
                torch.save(model.state_dict(), save_path)

                # Save the metrics data for future use
                metrics_data = {
                    'epochs': epochs_list,
                    'train_loss': train_losses,
                    'val_loss': val_losses,
                    'learning_rate': learning_rates
                }
                metrics_path = os.path.join(save_dir, 'training_metrics.pkl')
                with open(metrics_path, 'wb') as f:
                    pickle.dump(metrics_data, f)

            except Exception as e:
                print(f"Error saving model or metrics: {str(e)}")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(
                f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # Only save the final visualization at the end of training
    save_final_training_curves(epochs_list, train_losses, val_losses, learning_rates, batch_size)
    return model


def save_final_training_curves(epochs, train_losses, val_losses, learning_rates, batch_size):
    """
    Save the final visualization of training metrics.
    """
    try:
        # Create directory if it doesn't exist
        current_dir = os.getcwd()
        save_dir = os.path.join(current_dir, 'results', 'plots')
        os.makedirs(save_dir, exist_ok=True)

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot learning rate
        ax2.plot(epochs, learning_rates, 'g-')
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')  # Learning rate often best viewed on log scale
        ax2.grid(True)

        # Save the figure
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'training_curves_batch{batch_size}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved final training curves to {save_path}")
    except Exception as e:
        print(f"Error saving training curves: {str(e)}")


# def train_model(model, train_loader, val_loader, num_epochs=2000, lr=0.0005):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     print(f"Model type: {type(model)}")
#     print(f"Model: {model}")
#     model = model.to(device)
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)
#
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Changed scheduler
#         optimizer,
#         mode='min',
#         factor=0.5,
#         patience=20,
#         min_lr=1e-5,
#         verbose=True
#     )
#     best_val_loss = float('inf')
#     patience_counter = 0
#     max_patience = 300
#
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#
#         # Train on individual samples
#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#
#             pred_coords = model(batch.x, batch.edge_index)
#             loss = circular_layout_loss(pred_coords, batch.y, batch.x)
#
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#
#             total_loss += loss.item()
#
#         train_loss = total_loss / len(train_loader)
#         val_loss = evaluate(model, val_loader, device)
#
#         # Pass val_loss to scheduler.step()
#         scheduler.step(val_loss)
#
#         if epoch % 20 == 0 or epoch == num_epochs - 1:
#             # visualize_layout(model, next(iter(train_loader)), epoch, device)
#             pass
#
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             try:
#                 current_dir = os.getcwd()
#                 print("Tansih", current_dir)
#                 save_dir = os.path.join(current_dir,'results','metrics')
#                 os.makedirs(save_dir, exist_ok=True)
#                 save_path = os.path.join(save_dir, 'best_model.pt')
#                 torch.save(model.state_dict(), save_path)
#             except Exception as e:
#                 print(f"Error saving model: {str(e)}")
#         else:
#             patience_counter += 1
#
#         if patience_counter >= max_patience:
#             print(f"Early stopping at epoch {epoch}")
#             break
#
#         if epoch % 10 == 0:
#             print(
#                 f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
#
#     return model


# def train_epoch(model, loader, optimizer, device):
#     model.train()
#     total_loss = 0
#
#     for batch in loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()
#
#         # Get predictions
#         pred_coords = model(batch.x, batch.edge_index)
#
#         # Calculate pairwise Euclidean distance loss
#         loss = circular_layout_loss(pred_coords, batch.y, batch.x)
#
#         # Backward pass
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item() * batch.num_nodes
#
#     return total_loss / len(loader.dataset)
