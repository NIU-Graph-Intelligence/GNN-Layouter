import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Optional
from abc import ABC, abstractmethod
from evaluation import evaluate, circular_layout_loss, forceGNN_loss
from torch.optim import lr_scheduler

class BaseTrainer:
    def __init__(self, model, device, config):
        
        self.model = model
        self.device = device
        self.config = config
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    @abstractmethod
    def get_loss_type(self) -> str:
        """Return the type of loss function to use."""
        pass
        
    def save_checkpoint(self, save_path: str):
        """Save model checkpoint and training state."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, save_path)
        
    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']

    def validate(self, val_loader: DataLoader) -> float:

        return evaluate(self.model, val_loader, self.device, self.get_loss_type())

class CircularLayoutTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, device: torch.device, config: Dict):

        super().__init__(model, device, config)
    
    def get_loss_type(self) -> str:
        return 'circular'
    
    def train_epoch(self, train_loader: DataLoader) -> float:

        self.model.train()
        total_loss = 0
        count = 0
        
        for batch in train_loader:
            try:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                pred_coords = self.model(batch.x, batch.edge_index)
                loss = circular_layout_loss(pred_coords, batch.y, batch.x)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                count += 1
            except Exception as e:
                print(f"Error in training batch: {str(e)}")
                continue
            
        return total_loss / count if count > 0 else float('inf')
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              save_dir: str, model_name: str, batch_size: int) -> Optional[str]:

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_bs{batch_size}_best.pt')
        
        print(f"Training {model_name} with batch size {batch_size}")
        print(f"Saving checkpoints to {save_path}")
        
        for epoch in range(self.config['num_epochs']):
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print progress
            print(f'Epoch {epoch+1}/{self.config["num_epochs"]} - '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(save_path)
                print(f'Validation loss improved. Saved checkpoint to {save_path}')
            else:
                self.patience_counter += 1
            
            # Early stopping check
            if epoch >= self.config['min_epochs'] and self.patience_counter >= self.config['max_patience']:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        return save_path if os.path.exists(save_path) else None
    
    def visualize_results(self, save_path: str, batch_size: int, model_name: str):

        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Training Progress (Batch Size: {batch_size})')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = save_path.replace('.pt', '_training_curve.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Training curve saved to {plot_path}')

class ForceDirectedTrainer(BaseTrainer):
    def __init__(self, model: nn.Module, device: torch.device, config: Dict):

        super().__init__(model, device, config)
        
        # Use AdamW optimizer with different default parameters
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.0001),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # Add learning rate scheduler
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=500,
            min_lr=1e-6,
            verbose=True
        )
    
    def get_loss_type(self) -> str:
        return 'forceGNN'
    
    def train_epoch(self, train_loader: DataLoader) -> float:

        self.model.train()
        total_loss = 0
        count = 0
        
        for batch in train_loader:
            try:
                # Move batch to device and ensure all tensors are on correct device
                batch = batch.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                pred_coords = self.model(batch.x, batch.edge_index, batch.batch, batch.init_coords)
                loss = forceGNN_loss(pred_coords, batch.original_y)
                
                # Backward pass
                loss.backward()
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                count += 1
            except Exception as e:
                print(f"Error in training batch: {str(e)}")
                # Print device information for debugging
                print(f"Device information:")
                print(f"Model device: {next(self.model.parameters()).device}")
                print(f"Batch x device: {batch.x.device}")
                print(f"Batch edge_index device: {batch.edge_index.device}")
                print(f"Batch init_coords device: {batch.init_coords.device}")
                print(f"Batch original_y device: {batch.original_y.device}")
                continue
            
        return total_loss / count if count > 0 else float('inf')
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              save_dir: str, model_name: str, batch_size: int) -> Optional[str]:

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'ForceDirected_{model_name}_bs{batch_size}_best.pt')
        
        print(f"Training Force-Directed {model_name} with batch size {batch_size}")
        print(f"Saving checkpoints to {save_path}")
        
        for epoch in range(self.config['num_epochs']):
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Print progress with current learning rate
            print(f'Epoch {epoch+1}/{self.config["num_epochs"]} - '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(save_path)
                print(f'Validation loss improved. Saved checkpoint to {save_path}')
            else:
                self.patience_counter += 1
            
            # Early stopping check with minimum epochs requirement
            if epoch >= self.config['min_epochs'] and self.patience_counter >= self.config['max_patience']:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        return save_path if os.path.exists(save_path) else None
    
    def visualize_results(self, save_path: str, batch_size: int, model_name: str):

        # Create a figure with subplots for loss curves and sample predictions
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)
        
        # Plot loss curves
        ax_loss = fig.add_subplot(gs[0, :])
        ax_loss.plot(self.train_losses, label='Training Loss')
        ax_loss.plot(self.val_losses, label='Validation Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'Force-Directed {model_name} Training Progress (Batch Size: {batch_size})')
        ax_loss.legend()
        ax_loss.grid(True)
        
        # Save plot
        plot_path = save_path.replace('.pt', '_training_curve.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Training curve saved to {plot_path}')