# from VirtualizationDataset.CircularLayout import ImprovedLayoutGNN
from data.generate_graph_data import BA_graph_dataset, ER_graph_dataset, WS_graph_dataset, save_graph_dataset
from data.generate_layout_data import load_graph_dataset, convert_to_adjacency_matrices, save_adjacency_matrices, draw_circular_layouts, draw_shell_layouts, draw_kamada_kawai, save_layouts
from data.data_preprocessing import process_dictionary_data, create_node_features, prepare_data, normalized_coordinates
import torch
from models.gnn_model_1 import GNN_Model_1
from training.train import *
from data.dataset import data_loader
from utils.visualization import *
import matplotlib.pyplot as plt
import os

def visualize_layout(model, sample, epoch, device):
    model.eval()
    with torch.no_grad():
        sample = sample.to(device)
        pred_coords = model(sample.x, sample.edge_index)

        # Convert to numpy for plotting
        pred_coords = pred_coords.cpu().numpy()
        true_coords = sample.y.cpu().numpy()
        edges = sample.edge_index.cpu().numpy().T

        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot predicted layout
        ax1.scatter(pred_coords[:, 0], pred_coords[:, 1])
        for edge in edges:
            ax1.plot(pred_coords[edge, 0], pred_coords[edge, 1], 'gray', alpha=0.5)
        ax1.set_title(f'Predicted Layout - Epoch {epoch}')

        # Plot true layout
        ax2.scatter(true_coords[:, 0], true_coords[:, 1])
        for edge in edges:
            ax2.plot(true_coords[edge, 0], true_coords[edge, 1], 'gray', alpha=0.5)
        ax2.set_title('True Layout')

        save_dir = '/checkpoints'
        plt.tight_layout()
        # Save in the visualization directory with full path
        save_path = os.path.join(save_dir, f'layout_epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()

        # Print confirmation
        print(f"Saved visualization to {save_path}")

def main(batch_size=1):

    ER_graph_dataset(num_samples=500, nodes=50)
    WS_graph_dataset(num_samples=4500, num_nodes=50, nearest_neighbors=(2, 10),
                     rewiring_probability=(0.3999999, 0.59999999))
    BA_graph_dataset(num_samples=3000, num_nodes= 50, num_edges=(2, 25))

    save_graph_dataset()

    # Load the dataset before using it
    merged_graph_dataset = load_graph_dataset()

    # Process and convert data
    adjacency_matrices = convert_to_adjacency_matrices(merged_graph_dataset)
    save_adjacency_matrices(adjacency_matrices)

    circular_layouts = draw_circular_layouts(merged_graph_dataset)
    save_layouts(circular_layouts)

    shell_layouts = draw_shell_layouts(merged_graph_dataset)
    save_layouts(shell_layouts)

    kamada_kawai = draw_kamada_kawai(merged_graph_dataset)
    save_layouts(kamada_kawai)

    adj_matrices, coordinates = process_dictionary_data(adjacency_matrices, circular_layouts)
    dataset = prepare_data(adj_matrices, coordinates)

    torch.save(dataset, 'data/processed/modelInput.pt')

    models = GNN_Model_1
    train_loader, val_loader = data_loader(dataset)
    model_instance = models()
    trained_model = train_model(model_instance, train_loader, val_loader)

    return trained_model

if __name__ == '__main__':
    model = main()
    test_model_generalization()
