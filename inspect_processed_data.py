import torch
import os
import numpy as np

def inspect_processed_data(file_path):
    print(f"\nInspecting processed data from: {file_path}")
    print("-" * 50)
    
    data = torch.load(file_path)
    print(f"Type of loaded data: {type(data)}")
    
    # Print dictionary keys
    print("\nKeys in the dataset:")
    for key in data.keys():
        print(f"- {key}: {type(data[key])}")
    
    # Get dataset info
    dataset = data['dataset']
    print(f"\nNumber of graphs in dataset: {len(dataset)}")
    # print(f"Maximum nodes: {data['max_nodes']}")
    # print(f"Layout type: {data['layout_type']}")
    
    # Inspect first graph
    first_graph = dataset[0]
    print("\nFirst graph structure:")
    print(f"Number of nodes: {first_graph.num_nodes}")
    print(f"Number of edges: {first_graph.edge_index.size(1)}")
    # print(f"Graph type: {first_graph.graph_type}")
    print(f"Graph ID: {first_graph.graph_id}")
    
    # Inspect node features
    print("\nNode features (x) structure:")
    print(f"Shape: {first_graph.x.shape}")
    print("\nFeature breakdown:")
    print("1. Positional encoding (first 3 nodes):")
    print(first_graph.x[:10, 0])  # First feature
    print("\n2. Node degrees (first 3 nodes):")
    print(first_graph.x[:3, 1])  # Second feature
    print("\n3. Initial positions (x,y) (first 3 nodes):")
    print(first_graph.x[:3, 2:4])  # Third and fourth features
    
    # Inspect coordinates
    print("\nTarget coordinates structure:")
    print("Normalized coordinates (y) shape:", first_graph.y.shape)
    print("First 3 nodes normalized coordinates:")
    print(first_graph.y[:3])
    print("\nOriginal coordinates (original_y) shape:", first_graph.original_y.shape)
    print("First 3 nodes original coordinates:")
    print(first_graph.original_y[:3])
    
    # Edge structure
    print("\nEdge structure:")
    print("Edge index shape:", first_graph.edge_index.shape)
    print("First 5 edges (node pairs):")
    print(first_graph.edge_index[:, :5].t())

# Path to the processed file
processed_file = "data/processed/processed_forcedirected_onehot_positional.pt"

# Inspect the data
inspect_processed_data(processed_file) 