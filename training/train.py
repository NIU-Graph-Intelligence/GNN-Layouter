import sys
print(sys.executable)
import torch
import os
import sys
import argparse


# Add the project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Import all models using absolute imports
from models.GCN import GCN
from models.ChebConv import GNN_ChebConv
from models.GAT import GAT
from models.GIN import GNN_Model_GIN
from models.GCNFR import ForceGNN
from data.dataset import circular_data_loader, force_directed_data_loader
from training.trainer import CircularLayoutTrainer, ForceDirectedTrainer

# Import config manager
from config_utils.config_manager import get_config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dictionary mapping model names to their classes
MODEL_CLASSES = {
    'GCN': GCN,
    'ChebConv': GNN_ChebConv,
    'GAT': GAT,
    'GIN': GNN_Model_GIN,
    'ForceGNN': ForceGNN
}

def parse_args():
    config = get_config()
    
    parser = argparse.ArgumentParser(description='Train graph layout models')
    parser.add_argument('--model', type=str, default='ForceGNN', choices=MODEL_CLASSES.keys(),
                      help='Model architecture to use')
    parser.add_argument('--layout_type', type=str, default='force_directed', choices=['circular', 'force_directed'],
                      help='Type of layout to train')
    
    # Optional overrides - use config defaults if not provided
    parser.add_argument('--data_path', type=str, default=None,
                      help='Path to the dataset (uses config default if not provided)')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size for training (uses config default if not provided)')
    parser.add_argument('--num_epochs', type=int, default=None,
                      help='Number of epochs to train (uses config default if not provided)')
    parser.add_argument('--learning_rate', type=float, default=None,
                      help='Initial learning rate (uses config default if not provided)')
    parser.add_argument('--weight_decay', type=float, default=None,
                      help='Weight decay for optimizer (uses config default if not provided)')
    
    args = parser.parse_args()
    
    # Apply config defaults for unspecified arguments
    training_config = config.get_training_config(args.layout_type)
    data_config = config.get_data_config()
    
    if args.data_path is None:
        args.data_path = config.get_data_path(args.layout_type)
    if args.batch_size is None:
        args.batch_size = training_config.get('batch_size', 256)
    if args.num_epochs is None:
        args.num_epochs = training_config.get('num_epochs', 2000)
    if args.learning_rate is None:
        args.learning_rate = training_config.get('learning_rate', 0.003)
    if args.weight_decay is None:
        args.weight_decay = training_config.get('weight_decay', 0.005)
    
    return args

def setup_model_and_data(args):
    """Setup model and data loaders."""
    config = get_config()
    data_config = config.get_data_config()
    
    try:
        data_dict = torch.load(args.data_path, map_location=DEVICE, weights_only=False)
        dataset = data_dict['dataset']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Get data loader configuration
    splits = data_config.get('splits', [0.8, 0.1, 0.1])
    random_state = data_config.get('random_state', 42)
    
    # Choose appropriate data loader based on layout type
    if args.layout_type == 'circular':
        train_loader, val_loader, test_loader = circular_data_loader(
            dataset,
            batch_size=args.batch_size,
            splits=splits,
            random_state=random_state
        )
    else:  # force_directed
        train_loader, val_loader, test_loader = force_directed_data_loader(
            dataset,
            batch_size=args.batch_size,
            splits=splits,
            random_state=random_state
        )
    
    # Get input dimensions from data
    sample_graph = dataset[0]
    input_dim = sample_graph.x.size(1)
    
    # Create model with configuration parameters
    model_class = MODEL_CLASSES[args.model]
    model_config = config.get_model_config(args.model)
    
    if args.model == 'ForceGNN':
        # ForceGNN needs additional input features from initial coordinates
        in_feat = input_dim + sample_graph.init_coords.shape[1]
        model = model_class(
            in_feat=in_feat,
            hidden_dim=model_config.get('hidden_dim', 32),
            out_feat=model_config.get('out_feat', 2),
            num_layers=model_config.get('num_layers', 4)
        )
    else:
        # Use model configuration for other models
        model = model_class(input_dim=input_dim, **model_config)
    
    model = model.to(DEVICE)

    # Force all model parameters to be on the correct device
    for name, param in model.named_parameters():
        if param.device != DEVICE:
            param.data = param.data.to(DEVICE)
    
    # Also ensure all buffers are on the correct device
    for name, buffer in model.named_buffers():
        if buffer.device != DEVICE:
            buffer.data = buffer.data.to(DEVICE)
            
    return model, train_loader, val_loader, test_loader


def main():
    # Parse arguments
    args = parse_args()
    config = get_config()
    
    print(f"Using device: {DEVICE}")
    print(f"Configuration loaded from: {config.config_path}")
    print(f"Training {args.model} for {args.layout_type} layout")

    # Setup model and data
    model, train_loader, val_loader, test_loader = setup_model_and_data(args)
    
    # Get training configuration from config
    training_config = config.get_training_config(args.layout_type)
    
    # Create trainer configuration - merge config defaults with command line args
    trainer_config = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'min_epochs': training_config.get('min_epochs', 1000),
        'max_patience': training_config.get('max_patience', 100)
    }
    
    # Initialize appropriate trainer
    if args.layout_type == 'circular':
        trainer = CircularLayoutTrainer(
            model=model,
            device=DEVICE,
            config=trainer_config
        )
    else:  # force_directed
        trainer = ForceDirectedTrainer(
            model=model,
            device=DEVICE,
            config=trainer_config
        )
    
    # Setup save directory using layout type and config paths
    results_base = config.get_path('results.base') or 'results'
    save_dir = os.path.join(results_base, args.layout_type, args.model)
    os.makedirs(save_dir, exist_ok=True)
    
    # Train model
    save_path = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=save_dir,
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    # Visualize results if training was successful
    if save_path:
        trainer.visualize_results(
            save_path=save_path,
            batch_size=args.batch_size,
            model_name=args.model
        )
        
        # Evaluate on test set
        trainer.load_checkpoint(save_path)
        test_loss = trainer.validate(test_loader)
        print(f"\nTest Loss: {test_loss:.6f}")

if __name__ == '__main__':
    main()
