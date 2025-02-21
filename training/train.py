import torch
import torch.nn.functional as F
from .evaluation import evaluate, circular_layout_loss
# from visualization import *
# import sys
import os
# from ..main import *

def train_model(model, train_loader, val_loader, num_epochs=2000, lr=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Model type: {type(model)}")
    print(f"Model: {model}")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Changed scheduler
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

        # Pass val_loss to scheduler.step()
        scheduler.step(val_loss)

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            # visualize_layout(model, next(iter(train_loader)), epoch, device)
            pass

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            try:
                current_dir = os.getcwd()
                print("Tansih", current_dir)
                save_dir = os.path.join(current_dir,'results','metrics')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'best_model.pt')
                torch.save(model.state_dict(), save_path)
            except Exception as e:
                print(f"Error saving model: {str(e)}")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(
                f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    return model


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Get predictions
        pred_coords = model(batch.x, batch.edge_index)

        # Calculate pairwise Euclidean distance loss
        loss = circular_layout_loss(pred_coords, batch.y, batch.x)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_nodes

    return total_loss / len(loader.dataset)
