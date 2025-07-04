import pickle
import torch
from torch_geometric.data import Data

# Load the community dataset
print("Loading community dataset...") 
with open('data/raw/graph_dataset/community_graphs_dataset1024.pkl', 'rb') as f:
    community_data = pickle.load(f)

print("\nDataset Overview:")
print(f"Type of dataset: {type(community_data)}")
print(f"Number of graphs: {len(community_data)}")
print(f"Type of first item: {type(community_data[0])}")

# Inspect first graph in detail
first_graph = community_data[0]
print("\nFirst Graph Details:")
if isinstance(first_graph, Data):
    print("PyTorch Geometric Data object with attributes:")
    # Get all available attributes
    for key in first_graph.keys():
        attr = getattr(first_graph, key)
        if hasattr(attr, 'shape'):
            print(f"- {key}: shape={attr.shape}")
        else:
            print(f"- {key}: {attr}")
    print(f"Number of nodes: {first_graph.x}")

    # Check if edge weights exist
    if hasattr(first_graph, 'edge_attr'):
        print("\nEdge Attributes:")
        print(f"Shape: {first_graph.edge_attr.shape}")
        print(f"Sample values: {first_graph.edge_attr[:5]}")
    
    # Check if community information exists
    if hasattr(first_graph, 'community'):
        print("\nCommunity Information:")
        print(f"Shape: {first_graph.community.shape}")
        print(f"Unique communities: {torch.unique(first_graph.community)}")
        
    # Additional useful information
    print("\nGraph Statistics:")
    print(f"Number of nodes: {first_graph.num_nodes}")
    print(f"Number of edges: {first_graph.edge_index.shape[1]}")
    if hasattr(first_graph, 'batch'):
        print(f"Batch info: {first_graph.batch}")
    if hasattr(first_graph, 'graph_id'):
        print(f"Graph ID: {first_graph.graph_id}") 