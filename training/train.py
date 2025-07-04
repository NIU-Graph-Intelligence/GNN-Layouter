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
from trainer import CircularLayoutTrainer, ForceDirectedTrainer

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
    parser = argparse.ArgumentParser(description='Train graph layout models')
    parser.add_argument('--model', type=str, required=True, choices=MODEL_CLASSES.keys(),
                      help='Model architecture to use')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=2000,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                      help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005,
                      help='Weight decay for optimizer')
    parser.add_argument('--layout_type', type=str, default='circular', choices=['circular', 'force_directed'],
                      help='Type of layout to train')
    return parser.parse_args()

def setup_model_and_data(args):
    """Setup model and data loaders."""
    try:
        data_dict = torch.load(args.data_path, map_location=DEVICE)
        dataset = data_dict['dataset']
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Choose appropriate data loader based on layout type
    if args.layout_type == 'circular':
        train_loader, val_loader, test_loader = circular_data_loader(
            dataset,
            batch_size=args.batch_size,
            splits=(0.8, 0.1, 0.1),
            random_state=42
        )
    else:  # force_directed
        train_loader, val_loader, test_loader = force_directed_data_loader(
            dataset,
            batch_size=args.batch_size,
            splits=(0.8, 0.1, 0.1),
            random_state=42
        )
    
    # Get input dimensions from data
    sample_graph = dataset[0]
    input_dim = sample_graph.x.size(1)
    
    # Create model with appropriate parameters
    model_class = MODEL_CLASSES[args.model]
    if args.model == 'ForceGNN':
        # ForceGNN needs additional input features from initial coordinates
        in_feat = input_dim + sample_graph.init_coords.shape[1]
        model = model_class(
            in_feat=in_feat,
            hidden_dim=32,
            out_feat=2,
            num_layers=4
        )
    else:
        model = model_class(input_dim=input_dim)
    
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
    
    print(f"Using device: {DEVICE}")

    # Setup model and data
    model, train_loader, val_loader, test_loader = setup_model_and_data(args)
    
    # Ensure model device consistency
    
    # Create trainer configuration
    trainer_config = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'min_epochs': 500,
        'max_patience': 100
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
    
    # Setup save directory using layout type
    save_dir = os.path.join('results', args.layout_type, args.model)
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
