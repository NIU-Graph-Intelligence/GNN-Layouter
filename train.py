#!/usr/bin/env python3
"""
Main training script for graph layout models.
Usage: python train.py --model GCN --layout_type force_directed --data_path data/processed/community_5k_force_directed.pt
"""

import os
import sys
import argparse
import torch
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from data.dataset import load_dataset, create_data_loaders, get_dataset_info
from training.trainer import Trainer
from training.evaluation import evaluate_model, compute_metrics
from config_utils import load_training_config

# Import models from unified registry
from models.registry import MODEL_REGISTRY, get_available_models, get_model_class


def get_config_string_from_yaml(model_name, model_config):
    """
    Extract hyperparameters from YAML config to create unique filename suffix.
    
    Args:
        model_name: Name of the model
        model_config: Dictionary of model hyperparameters from YAML
    
    Returns:
        String representation of hyperparameters for filename
    """
    if not model_config:
        return "default"
    
    param_parts = []
    for param, value in sorted(model_config.items()):
        # Abbreviate common parameter names for shorter filenames
        param_abbrev = {
            'hidden_dim': 'h',
            'hidden_channels': 'h',
            'num_layers': 'l', 
            'heads': 'head',
            'dropout': 'd',
            'dropout_rate': 'd',
            'weight_decay': 'wd',
            'learning_rate': 'lr',
            'batch_size': 'bs',
            'epochs': 'ep',
            'K': 'k',
            'top_k': 'topk'
        }.get(param, param)
        
        # Format float values to avoid overly long decimals
        if isinstance(value, float):
            value_str = f"{value:.3f}".rstrip('0').rstrip('.')
        else:
            value_str = str(value)
            
        param_parts.append(f"{param_abbrev}{value_str}")
    
    return "_".join(param_parts)

def parse_args():
    parser = argparse.ArgumentParser(description='Train graph layout models')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True, 
                       help='Model architecture to use')
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
    try:
        model_class = get_model_class(model_name)
    except ValueError as e:
        available_models = get_available_models()
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
    
    model_config = config.get('models', {}).get(model_name, {})
    
    # Special handling for spring layout models that may need initial coordinates
    spring_models = ['MultiScaleSpringGNN']
    if model_name in spring_models:
        print(f"Note: {model_name} expects initial coordinates in the input")
        print("Make sure your dataset includes initial coordinates in the feature matrix")
    
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
    
    # Generate config string for unique filename
    model_config = config.get('models', {}).get(args.model, {})
    config_str = get_config_string_from_yaml(args.model, model_config)
    
    # Setup save directory with hyperparameter-based filename
    save_dir = os.path.join(args.results_dir, args.layout_type, args.model)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create unique filename based on hyperparameters
    if config_str == "default":
        save_filename = f'{args.model}_{args.layout_type}_default_best.pt'
    else:
        save_filename = f'{args.model}_{args.layout_type}_{config_str}_best.pt'
        
    print(f"Model will be saved as: {save_filename}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        loss_type=args.layout_type,
        **training_params
    )
    
    # Train model with updated save path
    print("\nStarting training...")
    checkpoint_path = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=save_dir,
        save_filename=save_filename,  # Pass the filename separately
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
        
        # Prepare results for saving
        results = {
            'args': vars(args),
            'training_params': training_params,
            'model_config': model_config,
            'config_string': config_str,
            'dataset_info': dataset_info,
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'best_val_loss': trainer.best_val_loss,
            'total_epochs': len(trainer.train_losses)
        }
        
        # Save as .pt format (for programmatic reading)
        results_path = checkpoint_path.replace('.pt', '_results.pt')
        torch.save(results, results_path)
        
        # Save as .json format (for human reading)
        json_path = checkpoint_path.replace('.pt', '_results.json')
        
        # Convert tensors to serializable format for JSON
        json_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                json_results[key] = value.item() if value.numel() == 1 else value.tolist()
            else:
                json_results[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        print(f"Human-readable results saved to: {json_path}")
        
        print(f"Results saved to: {results_path}")
        print(f"Human-readable results saved to: {json_path}")
        
        # Generate visualization command
        print("\n" + "="*60)
        print("🎨 To visualize the trained model, run:")
        print("="*60)
        viz_command = f"python visualize.py --model_path {checkpoint_path} --data_path {args.data_path}"
        print(viz_command)
        print("="*60)
        
    else:
        print("Training failed!")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())