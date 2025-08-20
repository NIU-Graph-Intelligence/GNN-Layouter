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
                loss = compute_loss_per_graph(
                    pred_coords, 
                    batch.y, 
                    batch.batch, 
                    self.loss_type
                )
                
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
              model_name: str) -> Optional[str]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            model_name: Name for saving files
            
        Returns:
            Path to best checkpoint if successful, None otherwise
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_{self.loss_type}_best.pt')
        
        print(f"Training {model_name} for {self.loss_type} layout")
        print(f"Saving checkpoints to {save_path}")
        
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