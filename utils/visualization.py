import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# from models.GCN import GCN
from models.GIN import GNN_Model_GIN
# from models.GAT import GAT
# from models.ChebConv import GNN_ChebConv
from torch_geometric.data import Data
import matplotlib.gridspec as gridspec

def nx_to_pyg_data(G, expected_feature_size):
    """
    Converts a NetworkX graph into a PyTorch Geometric Data object.
    Node features: identity matrix (one-hot) concatenated with positional encoding.
    """
    num_nodes = G.number_of_nodes()
    adj_matrix = nx.to_numpy_array(G)
    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create one-hot encoding
    one_hot = torch.eye(num_nodes, dtype=torch.float32)
    
    # Create positional encoding (normalize to [0,1] range)
    pos_encoding = torch.arange(num_nodes, dtype=torch.float32) / num_nodes
    pos_encoding = pos_encoding.unsqueeze(1)  # Make it a column vector
    
    # Concatenate one-hot and positional encoding
    x = torch.cat([one_hot, pos_encoding], dim=1)

    if x.shape[1] < expected_feature_size:
        # Pad with zeros
        padding = torch.zeros((num_nodes, expected_feature_size - x.shape[1]))
        x = torch.cat([x, padding], dim=1)
    elif x.shape[1] > expected_feature_size:
        # Trim extra dimensions
        x = x[:, :expected_feature_size]

    data = Data(x=x, edge_index=edge_index)
    return data

def generate_distinct_colors(n):
    """
    Generate n visually distinct colors using HSV color space.
    Returns a list of RGB colors.
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + 0.3 * (i % 2)
        value = 0.7 + 0.3 * ((i + 1) % 2)
        colors.append(plt.cm.hsv(hue))
    return colors

def draw_graph_layout(G, coords, edge_list, ax, title, node_colors=None):
    """Draw a graph layout on the given axes with the given coordinates"""
    num_nodes = len(coords)
    
    # If no colors provided, generate distinct colors for each node
    if node_colors is None:
        node_colors = generate_distinct_colors(num_nodes)
    
    # Plot nodes with their unique colors
    for i in range(num_nodes):
        ax.scatter(coords[i, 0], coords[i, 1], color=node_colors[i], s=100, alpha=0.7)
    
    # Plot edges in light gray to not interfere with node colors
    for edge in edge_list:
        i, j = edge
        ax.plot([coords[i, 0], coords[j, 0]], 
                [coords[i, 1], coords[j, 1]], 
                color='gray', alpha=0.2, linewidth=0.5)
    
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal')

def visualize_model_predictions(model_path, test_nodes=40):
    """
    Visualize model predictions using more nodes than training.
    Args:
        model_path: Path to the trained model weights
        test_nodes: Number of nodes to test (e.g., 60)
    """
    # Extract batch size from model path
    if 'batch' in model_path:
        batch_size = model_path.split('batch')[-1].split('.')[0]
    else:
        batch_size = model_path.split('_')[-1].split('.')[0]

    mode = "unnormalized" if "unnormalized" in model_path else "normalized"

    # Set up the figure
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 2)
    
    # Initialize model with original training size
    # model = GCN(max_nodes=40)  # Original model trained on 40 nodes
    model = GNN_Model_GIN(max_nodes=40)  # Now expects max_nodes+1 as input dim
    # model = GAT(max_nodes=40)
    # model = GNN_ChebConv(max_nodes=40)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Graph types to test
    graph_types = [
        ("ER (p=0.3)", nx.erdos_renyi_graph(n=test_nodes, p=0.3)),
        ("ER (p=0.5)", nx.erdos_renyi_graph(n=test_nodes, p=0.5)),
        ("BA (m=3)", nx.barabasi_albert_graph(n=test_nodes, m=3)),
        ("BA (m=5)", nx.barabasi_albert_graph(n=test_nodes, m=5)),
        ("WS (k=4, p=0.3)", nx.watts_strogatz_graph(n=test_nodes, k=4, p=0.3)),
        ("WS (k=6, p=0.4)", nx.watts_strogatz_graph(n=test_nodes, k=6, p=0.4))
    ]

    for idx, (graph_type, G) in enumerate(graph_types):
        # Convert to PyG data (using existing function)
        data = nx_to_pyg_data(G, 41)  # 40 + 1 for positional encoding

        # Get ground truth circular layout
        pos = nx.circular_layout(G)
        true_coords = np.array(list(pos.values()))
        
        # Get model prediction
        with torch.no_grad():
            pred_coords = model(data.x, data.edge_index).cpu().numpy()

        # Generate colors for nodes
        node_colors = generate_distinct_colors(test_nodes)

        # Create subplots
        ax_pred = fig.add_subplot(gs[idx, 0])
        ax_true = fig.add_subplot(gs[idx, 1])

        # Draw layouts
        edge_list = data.edge_index.t().cpu().numpy()
        draw_graph_layout(G, pred_coords, edge_list, ax_pred, 
                         f"{graph_type} Prediction\n(Testing on {test_nodes} nodes)", 
                         node_colors)
        draw_graph_layout(G, true_coords, edge_list, ax_true, 
                         f"{graph_type} Ground Truth\n({test_nodes} nodes)", 
                         node_colors)

    # Adjust layout and save
    plt.tight_layout()
    
    # Save visualization
    save_dir = os.path.join('results', 'figures', 'GNN_Model_GIN')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'GNN_Model_GIN_withPositionalFeature_{mode}_{test_nodes}nodes_batch{batch_size}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved node scaling visualization to {save_path}")

def usage():
    if len(sys.argv) != 2:
        print("Usage: python visualization.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        sys.exit(1)

    print(f"Testing model on 60 nodes using weights from {model_path}")
    visualize_model_predictions(model_path, test_nodes=41)

# Call usage function directly
usage()
