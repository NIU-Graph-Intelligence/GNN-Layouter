import torch
from torch.optim import lr_scheduler
import os
import matplotlib.pyplot as plt
import pickle
import sys
import argparse
from torch_geometric.data import Data
import numpy as np
import matplotlib.gridspec as gridspec
import networkx as nx

# Add the project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Import all models using absolute imports
from models.GCN import GCN
from models.ChebConv import GNN_ChebConv
from models.GAT import GAT
from models.GIN import GNN_Model_GIN
from evaluation import evaluate, circular_layout_loss
from data.dataset import data_loader
from data.generate_layout_data import draw_circular_layouts

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Dictionary mapping model names to their classes
MODEL_CLASSES = {
    'GCN': GCN,
    'GNN_ChebConv': GNN_ChebConv,
    'GAT': GAT,
    'GIN': GNN_Model_GIN
}


def train_model(model, train_loader, val_loader, batch_size, model_name, num_epochs=2000, lr=0.002):
    print(f"Training {model_name}")
    print(f"Model: {model}")
    model = model.to(DEVICE)

    data_path = sys.argv[2]
    mode = "normalized" if "normalized" in data_path else "unnormalized"

    # Adjusted optimizer parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.005,  # Increased weight decay
        betas=(0.9, 0.999)
    )
    
    # More lenient scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=95,
        min_lr=1e-6,
        verbose=True
    )

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 100
    min_epochs = 500

    epochs_list = []
    train_losses = []
    val_losses = []
    learning_rates = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            # Add debug print to show shapes
            # print(f"Input shape: {batch.x.shape}, Edge index shape: {batch.edge_index.shape}")
            #
            pred_coords = model(batch.x, batch.edge_index)
            #
            # # Add debug print to show output and target shapes
            # print(f"Predicted coords shape: {pred_coords.shape}, Target coords shape: {batch.y.shape}")

            loss = circular_layout_loss(pred_coords, batch.y, batch.x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, DEVICE, loss_type='circular')

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
                    current_dir = os.getcwd()
                    save_dir = os.path.join(current_dir, 'results', 'metrics', model_name)
                    os.makedirs(save_dir, exist_ok=True)

                    save_path = os.path.join(save_dir, f'best_model_{model_name}_withPositionalFeature_{mode}_batch{batch_size}.pt')
                    torch.save(model.state_dict(), save_path)

                    metrics_data = {
                        'epochs': epochs_list,
                        'train_loss': train_losses,
                        'val_loss': val_losses,
                        'learning_rate': learning_rates
                    }
                    metrics_path = os.path.join(save_dir, f'training_metrics_{model_name}_withPositionalFeature_{mode}_batch{batch_size}.pkl')
                    with open(metrics_path, 'wb') as f:
                        pickle.dump(metrics_data, f)

                except Exception as e:
                    print(f"Error saving model or metrics: {str(e)}")
            else:
                patience_counter += 1

            if epoch >= 100 and patience_counter >= max_patience:  # Don't stop before 100 epochs
                print(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(
                f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )

    # Save final visualization
    save_final_training_curves(epochs_list, train_losses, val_losses, learning_rates, 
                             batch_size, save_path, mode, model_name)
    return model

def save_final_training_curves(epochs, train_losses, val_losses, learning_rates, 
                             batch_size, best_model_path, mode, model_name):
    try:
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(2, 4, figure=fig)
        
        ax_loss = fig.add_subplot(gs[:, :2])    # Training/validation loss (left, spans 2 rows)
        ax_er_pred = fig.add_subplot(gs[0, 2])  # ER Graph Prediction (top-middle)
        ax_er_true = fig.add_subplot(gs[0, 3])  # ER Graph Ground Truth (top-right)
        ax_ba_pred = fig.add_subplot(gs[1, 2])  # BA Graph Prediction (bottom-middle)
        ax_ba_true = fig.add_subplot(gs[1, 3])  # BA Graph Ground Truth (bottom-right)
        
        # Set aspect ratio to be equal for all graph subplots
        for ax in [ax_er_pred, ax_er_true, ax_ba_pred, ax_ba_true]:
            ax.set_aspect('equal')
        
        # Plot training and validation losses
        ax_loss.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax_loss.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax_loss.set_title('Training and Validation Loss', fontsize=14)
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)
        
        try:
            data_dict = torch.load(sys.argv[2])
            dataset = data_dict['dataset']
            max_nodes = data_dict['max_nodes']
        except Exception as e:
            print(f"Error loading dataset info: {e}")
            max_nodes = 41  # fallback value

        model_instance = MODEL_CLASSES[model_name](max_nodes=max_nodes)
        model_instance.to(DEVICE)
        
        try:
            state_dict = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        except TypeError:
            state_dict = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
            
        model_instance.load_state_dict(state_dict)
        model_instance.eval()

        G1 = nx.erdos_renyi_graph(n=max_nodes, p=0.3, seed=42)
        visualize_separate_layouts(G1, model_instance, max_nodes, 
                                  ax_er_pred, ax_er_true,
                                  "ER Graph Prediction", "ER Graph Ground Truth")
        
        G2 = nx.barabasi_albert_graph(n=max_nodes, m=3, seed=43)
        visualize_separate_layouts(G2, model_instance, max_nodes, 
                                  ax_ba_pred, ax_ba_true,
                                  "BA Graph Prediction", "BA Graph Ground Truth")
        
        # Ensure all graph subplots have the same limits
        # First, find the overall min and max for all plots
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        
        for ax in [ax_er_pred, ax_er_true, ax_ba_pred, ax_ba_true]:
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            x_min = min(x_min, x_lim[0])
            x_max = max(x_max, x_lim[1])
            y_min = min(y_min, y_lim[0])
            y_max = max(y_max, y_lim[1])
        
        # Set all limits to the same values
        for ax in [ax_er_pred, ax_er_true, ax_ba_pred, ax_ba_true]:
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save with model-specific name
        current_dir = os.getcwd()
        save_dir = os.path.join(current_dir, 'results', 'plots', model_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'training_results_{model_name}_{mode}_withPositionalFeature_batch{batch_size}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {save_path}")
        
    except Exception as e:
        print(f"Error saving visualizations: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full error traceback for debugging

def visualize_separate_layouts(G, model, max_nodes, ax_pred, ax_true, pred_title, true_title):
    """Helper function to visualize ground truth and prediction separately"""
    # Convert to PyG data
    adj = nx.to_numpy_array(G)
    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    
    # Create node features (one-hot encoding + positional feature)
    try:
        # Get the expected feature size from the model's first layer
        expected_feature_size = model.conv1.lin.in_features 
        
        # Create appropriately sized features with positional feature
        x = torch.zeros((max_nodes, expected_feature_size))
        # Add one-hot encoding for as many dimensions as possible
        one_hot_size = min(max_nodes, expected_feature_size - 1)  # -1 to leave room for positional feature
        for i in range(max_nodes):
            if i < one_hot_size:
                x[i, i] = 1.0
            # Add positional encoding as the last feature
            x[i, -1] = i / max_nodes
    except AttributeError:
        # Fallback if we can't determine the expected size
        x = torch.zeros((max_nodes, max_nodes + 1))  # +1 for positional feature
        for i in range(max_nodes):
            x[i, i] = 1.0
            # Add positional encoding as the last feature
            x[i, -1] = i / max_nodes
    
    # Generate circular layout as ground truth
    pos = nx.circular_layout(G)
    y = torch.tensor(np.array(list(pos.values())), dtype=torch.float)
    
    # Create PyG data object and predict
    data = Data(x=x, edge_index=edge_index, y=y).to(DEVICE)
    try:
        with torch.no_grad():
            pred_coords = model(data.x, data.edge_index)
        
        # Convert predictions to numpy
        pred_coords_np = pred_coords.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # 1. Plot prediction
        draw_graph_layout(G, pred_coords_np, edge_index.cpu().numpy().T, ax_pred, pred_title)
        
        # 2. Plot ground truth
        draw_graph_layout(G, y_np, edge_index.cpu().numpy().T, ax_true, true_title)
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        ax_pred.text(0.5, 0.5, f"Visualization error: {str(e)}", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax_pred.transAxes, fontsize=10, color='red')
        ax_true.text(0.5, 0.5, f"Visualization error: {str(e)}", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax_true.transAxes, fontsize=10, color='red')

def draw_graph_layout(G, coords, edge_list, ax, title):
    """Draw a graph layout on the given axes with the given coordinates"""
    # Use different colors based on whether this is a prediction or ground truth
    if "Prediction" in title:
        node_color = 'red'
        edge_color = 'r-'
    else:  # Ground Truth
        node_color = 'blue'
        edge_color = 'b-'
    
    # Plot nodes
    ax.scatter(coords[:, 0], coords[:, 1], color=node_color, s=80, alpha=0.7)
    
    # Plot edges
    for edge in edge_list:
        i, j = edge
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], edge_color, alpha=0.3)
    
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal')  # Ensure circular layouts appear circular

def usage():
    if len(sys.argv) != 4:
        print("Usage: python train.py <model_name> data/processed/modelInput.pt <batch_size>")
        print("Available models: GCN, ChebConv, GAT, GIN")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in MODEL_CLASSES:
        print(f"Error: Unknown model {model_name}")
        print("Available models: GCN, ChebConv, GAT, GIN")
        sys.exit(1)

    data_path = sys.argv[2]
    mode = "normalized" if "normalized" in data_path else "unnormalized"
    batch_size = int(sys.argv[3])

    try:
        data_dict = torch.load(data_path)
        dataset = data_dict['dataset']
        # Extract max_nodes from the num_nodes attribute of the first element in dataset
        # Instead of trying to access data_dict['max_nodes'] directly
        if isinstance(dataset, list) and len(dataset) > 0:
            # If dataset is a list, take the maximum num_nodes across all samples
            max_nodes = max([data.num_nodes for data in dataset])
        else:
            # If dataset is a single Data object
            max_nodes = dataset.num_nodes
        
        mode = "normalized" if "normalized" in data_path else "unnormalized"
        print(f"Dataset loaded from {data_path}")
        print(f"Maximum number of nodes from dataset: {max_nodes}")
        print(f"Running in {mode} mode")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # train_loader, val_loader = data_loader(dataset)
    train_loader, val_loader, test_loader = data_loader(dataset, batch_size=batch_size, splits=(0.8, 0.1, 0.1), random_state=42)

    ModelClass = MODEL_CLASSES[model_name]
    model_instance = ModelClass(max_nodes=max_nodes)
    
    trained_model = train_model(model_instance, train_loader, val_loader, 
                              batch_size=batch_size, model_name=model_name)

    # Save final model
    output_dir = os.path.join("results", "metrics", model_name)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"best_model_{model_name}_withPositionalFeature_{mode}_batch{batch_size}.pt")
    torch.save(trained_model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    usage()
