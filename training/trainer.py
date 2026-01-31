# training/trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
from .losses import compute_loss_per_graph
from .evaluation import evaluate_model
from visualize import GraphVisualizer
import networkx as nx


class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device, 
                 loss_type: str = 'force_directed',
                 **config):
        """
        Unified trainer for all layout types.
        
        Args:
            model: PyTorch model
            device: torch.device
            loss_type: 'circular' or 'force_directed'
            **config: Training configuration parameters
        """
        self.model = model
        self.device = device
        self.loss_type = loss_type
        
        # Training configuration
        self.lr = config.get('lr', 0.001)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.epochs = config.get('epochs', 1000)
        self.min_epochs = config.get('min_epochs', 100)
        self.max_patience = config.get('max_patience', 50)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        print(f"Trainer initialized for {loss_type} layout")
        print(f"Config: lr={self.lr}, epochs={self.epochs}, weight_decay={self.weight_decay}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        count = 0
        
        for batch in train_loader:
            try:
                batch = batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                pred_coords = self.model(batch.x, batch.edge_index)
                
                # Compute loss
                # loss = compute_loss_per_graph(
                #     pred_coords, 
                #     batch.y, 
                #     batch.batch, 
                #     self.loss_type
                # )
                loss = compute_loss_per_graph(pred_coords, true_coords=batch.y,
                                             batch=batch.batch, loss_type = 'distill', edge_index=batch.edge_index
                                             , lambda_edge=0.2)

                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                count += 1
                
            except Exception as e:
                print(f"Error in training batch: {str(e)}")
                continue
        
        return total_loss / count if count > 0 else float('inf')
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        return evaluate_model(self.model, val_loader, self.device, self.loss_type)
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              save_dir: str,
              save_filename: str,
              model_name: str,
              preview: bool = False,
              preview_every: int = 10,
              monitor_samples: Optional[list] = None) -> Optional[str]:
        """
        Train the model with flexible file naming.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            save_filename: Specific filename for the checkpoint
            model_name: Name for logging purposes
            preview: If True, save a two-graph snapshot at epoch 0 and every 10 epochs
            preview_every: Interval (in epochs) for saving previews
            monitor_samples: List of up to two PyG Data graphs to preview consistently
            
        Returns:
            Path to best checkpoint if successful, None otherwise
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_filename)
        
        print(f"Training {model_name} for {self.loss_type} layout")
        print(f"Saving checkpoints to {save_path}")
        
        # --- Preview setup ---
        previews_dir = os.path.join(os.path.dirname(save_path), "previews")
        if preview and monitor_samples:
            os.makedirs(previews_dir, exist_ok=True)
            gv = GraphVisualizer(device=self.device)

            def _save_preview(epoch: int):
                """
                Render one fixed graph side-by-side:
                left  = ground-truth coordinates (data.y)
                right = current model prediction
                """
                if not monitor_samples:
                    return

                data = monitor_samples[0]   # use only the first sample
                # Safety: ensure we have GT coordinates
                if not hasattr(data, "y") or data.y is None:
                    print(f"[preview] no ground-truth 'y' on sample; skipping epoch {epoch}")
                    return

                # Forward pass for prediction
                try:
                    self.model.eval()
                    with torch.no_grad():
                        pred = self.model(data.x.to(self.device), data.edge_index.to(self.device))
                    pos_pred = pred.detach().cpu().numpy()
                    if pos_pred.shape[1] > 2:    # if model outputs >2 dims, take first 2
                        pos_pred = pos_pred[:, :2]
                except Exception as e:
                    print(f"[preview] forward failed at epoch {epoch}: {e}")
                    return
                finally:
                    self.model.train()

                # Ground truth positions
                pos_true = data.y.detach().cpu().numpy()
                if pos_true.shape[1] > 2:
                    pos_true = pos_true[:, :2]

                # Build NX graph & colors via your helpers
                try:
                    gv = GraphVisualizer(device=self.device)
                    G = gv.create_networkx_graph(data)
                    node_colors, _ = gv.get_node_colors(data)
                except Exception as e:
                    print(f"[preview] NX/color failed at epoch {epoch}: {e}")
                    return

                # Two-panel figure: GT (left) vs Pred (right)
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                ax_true, ax_pred = axes

                nx.draw(
                    G,
                    pos={i: pos_true[i] for i in range(data.num_nodes)},
                    ax=ax_true,
                    node_color=node_colors,
                    edge_color='gray',
                    alpha=0.85,
                    node_size=50,
                    width=0.4,
                    with_labels=False
                )
                title_id = getattr(data, 'graph_id', None) or "Sample"
                layout_type = getattr(data, 'layout_type', 'ground-truth')
                ax_true.set_title(f"{title_id} — Ground Truth ({layout_type})")
                ax_true.set_aspect('equal'); ax_true.axis('off')

                nx.draw(
                    G,
                    pos={i: pos_pred[i] for i in range(data.num_nodes)},
                    ax=ax_pred,
                    node_color=node_colors,
                    edge_color='gray',
                    alpha=0.85,
                    node_size=50,
                    width=0.4,
                    with_labels=False
                )
                ax_pred.set_title(f"{title_id} — Prediction (epoch {epoch})")
                ax_pred.set_aspect('equal'); ax_pred.axis('off')

                fig.suptitle(f"{self.loss_type.capitalize()} preview — epoch {epoch}", y=0.98)

                out_path = os.path.join(previews_dir, f"epoch_{epoch:05d}.png")
                plt.savefig(out_path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                print(f"[preview] saved {out_path}")


            # Initial snapshot (epoch 0)
            try:
                _save_preview(0)
            except Exception as e:
                print(f"[preview] initial snapshot skipped: {e}")


        for epoch in range(self.epochs):
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch < 10:
                print(f'Epoch {epoch+1}/{self.epochs} - '
                      f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(save_path)
                if (epoch + 1) % 10 == 0 or epoch < 10:
                    print(f'  -> Validation improved! Saved checkpoint.')
            else:
                self.patience_counter += 1
            
            # Early stopping check
            if epoch >= self.min_epochs and self.patience_counter >= self.max_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

            # Periodic preview
            if preview and monitor_samples and ((epoch + 1) % preview_every == 0):
                try:
                    _save_preview(epoch + 1)
                except Exception as e:
                    print(f"[preview] skipped at epoch {epoch + 1}: {e}")

        return save_path if os.path.exists(save_path) else None
    
    def save_checkpoint(self, save_path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'loss_type': self.loss_type,
            'config': {
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'epochs': self.epochs
            }
        }
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, load_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from {load_path}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
    
    def plot_training_curves(self, save_path: str):
        """Plot and save training curves"""
        if not self.train_losses or not self.val_losses:
            print("No training history to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', alpha=0.8)
        plt.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress ({self.loss_type} layout)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = save_path.replace('.pt', '_training_curve.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Training curve saved to {plot_path}')