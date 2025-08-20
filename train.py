#!/usr/bin/env python3
"""
Main training script for graph layout models.
Usage: python train.py --model GCN --layout_type force_directed --data_path data/processed/community_5k_force_directed.pt
"""

import os
import sys
import argparse
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from data.dataset import load_dataset, create_data_loaders, get_dataset_info
from training.trainer import Trainer
from training.evaluation import evaluate_model, compute_metrics
from config_utils import load_training_config

# PyTorch Geometric standard models
from torch_geometric.nn import GCNConv, GATConv, GINConv, ChebConv, SAGEConv
import torch.nn as nn
import torch.nn.functional as F

# Custom models (cleaned versions)
CUSTOM_MODELS = {}
try:
    from models.GCN import GCN
    CUSTOM_MODELS['CustomGCN'] = GCN
except ImportError:
    pass

try:
    from models.GAT import GAT
    CUSTOM_MODELS['CustomGAT'] = GAT
except ImportError:
    pass

try:
    from models.GIN import GIN
    CUSTOM_MODELS['CustomGIN'] = GIN
except ImportError:
    pass

try:
    from models.ChebConv import ChebNet
    CUSTOM_MODELS['CustomChebNet'] = ChebNet
except ImportError:
    pass

try:
    from models.GCNFR import ForceGNN
    CUSTOM_MODELS['ForceGNN'] = ForceGNN
except ImportError:
    pass

# Standard PyG models
class StandardGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, 2))  # Output 2D coordinates
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class StandardGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_dim * heads, 2, heads=1, dropout=dropout))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class StandardSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, 2))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

# Combined model registry
MODEL_CLASSES = {
    # Standard PyG models
    'GCN': StandardGCN,
    'GAT': StandardGAT,
    'GraphSAGE': StandardSAGE,
    
    # Custom models (if available)
    **CUSTOM_MODELS
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train graph layout models')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True, 
                       help='Model architecture to use (use --list-models to see available models)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models and exit')
    parser.add_argument('--layout_type', type=str, required=True,
                       choices=['circular', 'force_directed'],
                       help='Type of layout to train')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the dataset file')
    
    # Optional arguments (will override config file)
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='Weight decay for optimizer')
    
    # Other options
    parser.add_argument('--config', type=str, default='training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """Setup computation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device

def create_model(model_name: str, input_dim: int, config: dict) -> torch.nn.Module:
    """Create model instance"""
    if model_name not in MODEL_CLASSES:
        available_models = list(MODEL_CLASSES.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
    
    model_class = MODEL_CLASSES[model_name]
    model_config = config.get('models', {}).get(model_name, {})
    
    # Special handling for ForceGNN - remove batch and init_coords parameters
    if model_name == 'ForceGNN':
        print("Note: ForceGNN will use standard (x, edge_index) interface")
        print("Initial coordinates should be included in the feature matrix x")
    
    # Create model with configuration
    try:
        model = model_class(input_dim=input_dim, **model_config)
    except TypeError as e:
        # Fallback: try with minimal parameters
        print(f"Warning: Could not create {model_name} with config {model_config}")
        print(f"Error: {e}")
        print("Trying with minimal parameters...")
        model = model_class(input_dim=input_dim)
    
    print(f"Created {model_name} model with input_dim={input_dim}")
    if model_config:
        print(f"Model config: {model_config}")
    
    return model

def main():
    args = parse_args()
    
    # Handle --list-models
    if args.list_models:
        print("Available models:")
        print("\nStandard PyG models:")
        for name in ['GCN', 'GAT', 'GraphSAGE']:
            if name in MODEL_CLASSES:
                print(f"  {name}")
        
        print("\nCustom models:")
        for name in CUSTOM_MODELS:
            print(f"  {name}")
        return 0
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load configuration
    try:
        config = load_training_config(args.config)
        print(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        print(f"Configuration file {args.config} not found, using defaults")
        config = {}
    
    # Get layout-specific configuration
    layout_config = config.get(args.layout_type, {})
    
    # Override with command line arguments
    training_params = {
        'batch_size': args.batch_size or layout_config.get('batch_size', 32),
        'lr': args.lr or layout_config.get('lr', 0.001),
        'epochs': args.epochs or layout_config.get('epochs', 1000),
        'weight_decay': args.weight_decay or layout_config.get('weight_decay', 0.01),
        'min_epochs': layout_config.get('min_epochs', 100),
        'max_patience': layout_config.get('max_patience', 50)
    }
    
    print(f"Training parameters: {training_params}")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = load_dataset(args.data_path)
    
    # Print dataset information
    dataset_info = get_dataset_info(dataset)
    print("Dataset information:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    data_config = config.get('data', {})
    splits = [data_config.get('train_split', 0.8), 
              data_config.get('val_split', 0.1), 
              data_config.get('test_split', 0.1)]
    random_state = data_config.get('random_state', 42)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=training_params['batch_size'],
        splits=splits,
        random_state=random_state
    )
    
    # Create model
    input_dim = dataset[0].x.shape[1]
    model = create_model(args.model, input_dim, config)
    model = model.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        loss_type=args.layout_type,
        **training_params
    )
    
    # Setup save directory
    save_dir = os.path.join(args.results_dir, args.layout_type, args.model)
    os.makedirs(save_dir, exist_ok=True)
    
    # Train model
    print("\nStarting training...")
    checkpoint_path = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=save_dir,
        model_name=args.model
    )
    
    if checkpoint_path:
        print(f"\nTraining completed successfully!")
        print(f"Best model saved to: {checkpoint_path}")
        
        # Plot training curves
        trainer.plot_training_curves(checkpoint_path)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        trainer.load_checkpoint(checkpoint_path)
        test_loss = trainer.validate(test_loader)
        test_metrics = compute_metrics(trainer.model, test_loader, device)
        
        print(f"Test Loss: {test_loss:.6f}")
        print("Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        # Save final results
        results = {
            'args': vars(args),
            'training_params': training_params,
            'dataset_info': dataset_info,
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'best_val_loss': trainer.best_val_loss,
            'total_epochs': len(trainer.train_losses)
        }
        
        results_path = checkpoint_path.replace('.pt', '_results.pt')
        torch.save(results, results_path)
        print(f"Results saved to: {results_path}")
        
    else:
        print("Training failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())