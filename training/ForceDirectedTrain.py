import torch
from torch.optim import lr_scheduler
import os
import matplotlib.pyplot as plt
import pickle
import sys
import argparse
import networkx as nx
import torch.nn.functional as F

# Add the project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Import models and evaluation functions
from models.GCNFR import ForceGNN
from evaluation import evaluate, forceGNN_loss
from data.dataset import data_loader

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Dictionary mapping model names to their classes
MODEL_CLASSES = {
    'ForceGNN': ForceGNN,
}


def train_model(model, train_loader, val_loader, batch_size, model_name, num_epochs=2000, lr=0.0001):
    print(f"Training {model_name}")
    print(f"Model: {model}")
    model = model.to(DEVICE)

    # Extract layout type from command line args
    layout_type = sys.argv[3]
    data_path = sys.argv[2]
    print(data_path)

    # Optimizer with AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=500,
        min_lr=1e-6,
        verbose=True
    )

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 500
    min_epochs = 200  # Ensure the model trains for at least this many epochs

    epochs_list = []
    train_losses = []
    val_losses = []
    learning_rates = []

    # Define save_path outside the conditional blocks to avoid UnboundLocalError
    current_dir = os.getcwd()
    save_dir = os.path.join(current_dir, 'results', 'metrics', model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'OneHotCustom{model_name}_bs{batch_size}_ep2000.pt')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad()

            # Forward pass - model now takes the entire batch
            pred_coords = model(data.x, data.edge_index, data.batch, data.init_coords)
            true = data.original_y.to(DEVICE)

            # Calculate loss
            loss = forceGNN_loss(pred_coords, true)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, DEVICE, loss_type='forceGNN')

        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_loss)

        if epoch >= min_epochs:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                try:
                    # Save the best model
                    torch.save(model.state_dict(), save_path)

                    # Save training metrics
                    metrics_data = {
                        'epochs': epochs_list,
                        'train_loss': train_losses,
                        'val_loss': val_losses,
                        'learning_rate': learning_rates
                    }
                    metrics_path = os.path.join(save_dir,
                                                f'OneHotCustomFinal_training_metrics_{model_name}_{layout_type}_batch{batch_size}.pkl')
                    with open(metrics_path, 'wb') as f:
                        pickle.dump(metrics_data, f)

                except Exception as e:
                    print(f"Error saving model or metrics: {str(e)}")
            else:
                patience_counter += 1

            if epoch >= 100 and patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(
                f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )

    # Save final visualization
    save_final_training_curves(epochs_list, train_losses, val_losses, learning_rates,
                               batch_size, save_path, layout_type, model_name, dataset=val_loader.dataset,
                               num_layers=len(model.layers))
    return model


def save_final_training_curves(epochs, train_losses, val_losses, learning_rates,
                               batch_size, best_model_path, layout_type, model_name, dataset=None, num_layers=4):
    try:
        fig = plt.figure(figsize=(24, 20))  
        gs = plt.GridSpec(4, 3, figure=fig)  
        ax_loss = fig.add_subplot(gs[:, 0])

        # Axes for predictions and ground truths
        ax_preds = [fig.add_subplot(gs[i, 1]) for i in range(4)]
        ax_gts = [fig.add_subplot(gs[i, 2]) for i in range(4)]

        for ax in ax_preds + ax_gts:
            ax.set_aspect('equal')

        ax_loss.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax_loss.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax_loss.set_title('Training and Validation Loss', fontsize=14)
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)
        in_channels = dataset[0].x.shape[1] + dataset[0].init_coords.shape[1]
        # in_channels = dataset[0].x.shape[1]
        if model_name == 'ForceGNN':
            model_instance = ForceGNN(
                in_feat=in_channels,
                hidden_dim=32,  # Match the training hidden dimension
                out_feat=2,
                num_layers=num_layers,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        model_instance.to(DEVICE)

        try:
            state_dict = torch.load(best_model_path, map_location=DEVICE)
            model_instance.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model state: {e}")

        model_instance.eval()

        if dataset and len(dataset) >= 4:
            sample_indices = [0, 1, 2, 3]

            for i, idx in enumerate(sample_indices):
                sample_data = dataset[idx].to(DEVICE)
                
                # Create batch tensor for single graph
                if sample_data.batch is None:
                    sample_data.batch = torch.zeros(sample_data.x.size(0), dtype=torch.long, device=DEVICE)
                
                with torch.no_grad():
                    pred_coords = model_instance(sample_data.x, sample_data.edge_index, sample_data.batch, sample_data.init_coords)
                    # For single graph, pred_coords will be [1, num_nodes, 2], so take first batch
                    pred_coords = pred_coords.squeeze(0)
                    pred_coords = pred_coords - pred_coords.mean(dim=0, keepdim=True)

                pred_coords_np = pred_coords.cpu().numpy()
                true_coords_np = sample_data.original_y.cpu().numpy()
                edge_index_np = sample_data.edge_index.cpu().numpy().T

                G = nx.Graph()
                G.add_nodes_from(range(sample_data.num_nodes))
                G.add_edges_from(edge_index_np)

                draw_graph_layout(G, pred_coords_np, edge_index_np, ax_preds[i], f"Graph {i + 1} Prediction")
                draw_graph_layout(G, true_coords_np, edge_index_np, ax_gts[i], f"Graph {i + 1} Ground Truth")
        else:
            print("Warning: Dataset not available or too small for visualization")
            for ax in ax_preds + ax_gts:
                ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        current_dir = os.getcwd()
        save_dir = os.path.join(current_dir, 'results', 'plots', model_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'PEFinal_Results_{model_name}_{layout_type}_batch{batch_size}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization to {save_path}")

    except Exception as e:
        print(f"Error saving visualizations: {str(e)}")
        import traceback
        traceback.print_exc()


def draw_graph_layout(G, coords, edge_list, ax, title):
    """Draw a graph layout on the given axes with the given coordinates"""
    coords = coords.copy()

    # Determine prediction vs ground truth
    if "Prediction" in title:
        node_color = 'red'
        edge_color = 'red'

        # Center and normalize
        coords = coords - coords.mean(axis=0)
        scale = max(coords[:, 0].ptp(), coords[:, 1].ptp())
        if scale > 1e-6:
            coords /= scale

        positions = {i: (coords[i, 0], coords[i, 1]) for i in range(len(coords))}

    else:  # Ground Truth - use stored original_y directly
        node_color = 'blue'
        edge_color = 'blue'

        # coords here is already original_y (unnormalized ground truth)
        positions = {i: (coords[i, 0], coords[i, 1]) for i in range(len(coords))}

    # Plot the graph using NetworkX
    nx.draw(G,
            pos=positions,
            ax=ax,
            node_color=node_color,
            edge_color=edge_color,
            node_size=80,
            width=0.5,
            alpha=0.7,
            with_labels=False)

    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal')


def main():
    parser = argparse.ArgumentParser(description='Train force-directed layout GNN models')
    parser.add_argument('--model', type=str, default='ForceGNN', choices=MODEL_CLASSES.keys(),
                        help='Model architecture to use')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the dataset file')
    parser.add_argument('--layout', type=str, default='FR', choices=['FR', 'FT2'],
                        help='Layout type to train for')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--hidden-channels', type=int, default=32,
                        help='Number of hidden channels in the model')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of GNN layers')

    args = parser.parse_args()

    # Store args in sys.argv for compatibility with existing code
    sys.argv = [sys.argv[0], args.model, args.data, args.layout, str(args.batch_size)]

    try:
        data_dict = torch.load(args.data)

        dataset = data_dict['dataset']

        # print(dataset[0].x + dataset[0].init_coords)
        in_channels = dataset[0].x.shape[1] + dataset[0].init_coords.shape[1]

        # Create data loaders
        train_loader, val_loader, test_loader = data_loader(dataset, batch_size=args.batch_size, splits=(0.8,0.1,0.1), random_state=42)

        # Initialize model
        if args.model == 'ForceGNN':
            model_instance = ForceGNN(in_channels, args.hidden_channels,2, args.num_layers)
        else:
            raise ValueError(f"Unknown model name: {args.model}")

        # Train the model
        trained_model = train_model(
            model=model_instance,
            train_loader=train_loader,
            val_loader=val_loader,
            batch_size=args.batch_size,
            model_name=args.model,
            num_epochs=args.epochs,
            lr=args.lr
        )

        # Save final model
        output_dir = os.path.join("results", "metrics", args.model)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"OneHotCustomWeights_{args.model}_{args.layout}_batch{args.batch_size}.pt")
        torch.save(trained_model.state_dict(), save_path)
        print(f"Final model saved to {save_path}")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

