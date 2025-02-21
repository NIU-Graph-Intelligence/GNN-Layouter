import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import os
from models.gnn_model_1 import GNN_Model_1
from data.data_preprocessing import create_node_features

root_dir = os.getcwd()
model_path = os.path.join(root_dir,"results","metrics", "best_model.pt")

output_dir = os.path.join(root_dir,"GNN-Layouter","results", "figures")
os.makedirs(output_dir, exist_ok=True)

# Verify and load
if os.path.exists(model_path):
    print("Model path is valid:", model_path)
else:
    print("Model path does not exist:", model_path)


def prepare_variable_size_graph(G):
    adj_matrix = nx.adjacency_matrix(G).todense()
    edge_index = torch.from_numpy(np.array(adj_matrix.nonzero())).long()
    num_nodes = adj_matrix.shape[0]

    if num_nodes <= 50:
        one_hot = torch.zeros((num_nodes, 50))
        one_hot[:num_nodes, :num_nodes] = torch.eye(num_nodes)
    else:
        one_hot = torch.eye(num_nodes)[:, :50]

    pos_features = create_node_features(num_nodes)
    x = torch.cat([one_hot, pos_features], dim=1)

    return Data(x=x, edge_index=edge_index)


def visualize_layouts(G, pred_coords, size, graph_type):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    edges = np.array(G.edges())
    # Predicted layout
    for edge in edges:
        ax1.plot(pred_coords[edge, 0], pred_coords[edge, 1], 'gray', alpha=0.5)
    ax1.scatter(pred_coords[:, 0], pred_coords[:, 1], c='blue')
    ax1.set_title(f'Model Prediction\n({size} nodes)')
    ax1.axis('equal')

    # NetworkX circular layout
    nx_circular = nx.circular_layout(G)
    nx_coords = np.array([coord for coord in nx_circular.values()])
    for edge in edges:
        ax2.plot(nx_coords[edge, 0], nx_coords[edge, 1], 'gray', alpha=0.5)
    ax2.scatter(nx_coords[:, 0], nx_coords[:, 1], c='green')
    ax2.set_title('NetworkX Circular')
    ax2.axis('equal')

    plt.tight_layout()
    # Save the figure
    filename = f'layout_comparison_{graph_type}_{size}nodes.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Saved visualization to: {filepath}")

    plt.show()


def test_model_generalization():
    model = GNN_Model_1()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Test different node sizes
    node_sizes = [50, 60, 70, 80, 90]

    for size in node_sizes:
        print(f"\nTesting graph with {size} nodes:")

        graphs = {
            'BA': nx.barabasi_albert_graph(n=size, m=2),
            'ER': nx.erdos_renyi_graph(n=size, p=0.1),
            'WS': nx.watts_strogatz_graph(n=size, k=4, p=0.1)
        }

        for graph_type, G in graphs.items():
            print(f"\nTesting {graph_type} graph:")

            try:
                test_data = prepare_variable_size_graph(G)

                with torch.no_grad():
                    pred_coords = model(test_data.x, test_data.edge_index)
                    pred_coords = pred_coords.numpy()

                print(f"Successfully generated layout for {graph_type} graph with {size} nodes")
                visualize_layouts(G, pred_coords, size, graph_type)

            except Exception as e:
                print(f"Error processing {graph_type} graph with {size} nodes: {str(e)}")

#
# if __name__ == "__main__":
#     test_model_generalization()