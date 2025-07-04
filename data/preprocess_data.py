import argparse
import pickle
import torch
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree

def load_data(args):
    """Load layout data and adjacency matrices"""
    print(f"Loading data from {args.input}...")
    
    # Load layouts
    layouts = torch.load(args.input)
    
    # Load adjacency matrices
    adjacency_matrices = torch.load(args.adj_matrices)
    print(f"Loaded adjacency matrices from {args.adj_matrices}")
    
    # For force-directed, also load initial positions
    initial_positions = None
    if args.layout_type == 'forcedirected':
        if args.init_positions:
            initial_positions = torch.load(args.init_positions)
            print(f"Loaded initial positions from {args.init_positions}")
    
    return layouts, adjacency_matrices, initial_positions

def create_node_features(num_nodes, feature_types):
    """Create node features based on specified types"""
    feature_list = []
    
    if 'onehot' in feature_types:
        one_hot = torch.eye(num_nodes)  # Shape: [N, N]
        feature_list.append(one_hot)
    
    if 'positional' in feature_types:
        pos_encoding = torch.linspace(0, 1, num_nodes).unsqueeze(1)  # Shape: [N, 1]
        feature_list.append(pos_encoding)
    
    if 'degree' in feature_types:
        # Degree will be added later when we have edge_index
        pass
    
    if not feature_list and 'degree' not in feature_types:
        raise ValueError("At least one feature type must be specified (onehot, positional, or degree)")
    
    return torch.cat(feature_list, dim=1) if feature_list else None

def process_circular_layout(adj_data, layout_data, feature_types):
    """Process circular layout following original data_preprocessing.py approach"""
    # Get adjacency matrix
    adj_matrix = adj_data['matrix']
    num_nodes = adj_matrix.shape[0]
    
    # Process coordinates in order of node indices
    coords = np.zeros((num_nodes, 2))
    layout_dict = layout_data['layout']
    for node_idx in range(num_nodes):
        if node_idx in layout_dict:
            coords[node_idx] = layout_dict[node_idx]
    
    # Create edge index from adjacency matrix
    rows, cols = adj_matrix.nonzero()
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    
    # Create node features
    x = create_node_features(num_nodes, feature_types)
    
    # Add degree features if requested
    if 'degree' in feature_types:
        row, _ = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)
        x = torch.cat([x, deg], dim=1) if x is not None else deg
    
    # Normalize coordinates
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-5)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.from_numpy(coords).float(),
        graph_id=adj_data['graph_id']
    )
    
    return data

def process_force_directed_layout(adj_data, layout_data, feature_types, initial_pos=None):
    """Process force-directed layout following Preprocessing_ForceDirected.py approach"""
    # Get adjacency matrix and mapping
    adj_matrix = adj_data['matrix']
    node_mapping = adj_data['node_mapping']
    graph_id = adj_data['graph_id']
    num_nodes = adj_matrix.shape[0]
    
    # Process coordinates using node mapping
    coords = np.zeros((num_nodes, 2))
    layout_dict = layout_data['layout']
    for orig_id, mapped_idx in node_mapping.items():
        if orig_id in layout_dict:
            coords[mapped_idx] = layout_dict[orig_id]
    
    # Create edge index and edge weights from adjacency matrix
    rows, cols = adj_matrix.nonzero()
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weights = adj_matrix[rows, cols]
    edge_weights = edge_weights.astype(np.float32)
    
    # Normalize edge weights to [0, 1]
    # min_val = edge_weights.min()
    # max_val = edge_weights.max()
    # if max_val > min_val:
    #     edge_weights = (edge_weights - min_val) / (max_val - min_val)
    # else:
    #     edge_weights = np.zeros_like(edge_weights)
    
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)
    
    # Create node features
    x = create_node_features(num_nodes, feature_types)
    
    # Add degree features if requested
    if 'degree' in feature_types:
        row, _ = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)
        x = torch.cat([x, deg], dim=1) if x is not None else deg
    
    # Normalize coordinates
    original_coords = coords.copy()
    normalized_coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-5)
    
    # Create PyG Data object with force-directed specific attributes
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.from_numpy(normalized_coords).float(),
        original_y=torch.from_numpy(original_coords).float(),
        graph_id=graph_id
    )
    
    # Add initial coordinates if available
    if initial_pos is not None:
        init_coords = np.zeros((num_nodes, 2))
        init_dict = initial_pos['layout']
        for orig_id, mapped_idx in node_mapping.items():
            if orig_id in init_dict:
                init_coords[mapped_idx] = init_dict[orig_id]
        data.init_coords = torch.from_numpy(init_coords).float()
    
    return data

def prepare_dataset(layouts, adjacency_matrices, layout_type, feature_types, initial_positions=None):
    """Prepare the complete dataset"""
    dataset = []
    
    # For circular layouts
    if layout_type == 'circular':
        for layout_data in layouts:
            graph_id = layout_data['graph_id']
            # Find matching adjacency data
            adj_data = next((adj for adj in adjacency_matrices if adj['graph_id'] == graph_id), None)
            if adj_data:
                data = process_circular_layout(adj_data, layout_data, feature_types)
                dataset.append(data)
    
    # For force-directed layouts
    else:
        # For force-directed, layouts is a list of layout data
        for layout_data in layouts:
            graph_id = layout_data['graph_id']
            # Find matching adjacency and initial position data
            adj_data = next((adj for adj in adjacency_matrices if adj['graph_id'] == graph_id), None)
            init_pos = next((init for init in initial_positions if init['graph_id'] == graph_id), None) if initial_positions else None
            
            if adj_data:
                data = process_force_directed_layout(
                    adj_data, layout_data, feature_types, init_pos
                )
                dataset.append(data)
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Preprocess graph layouts for GNN training')
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                      help='Path to layout pickle file')
    parser.add_argument('--adj-matrices', type=str, required=True,
                      help='Path to adjacency matrices pickle file')
    parser.add_argument('--layout-type', type=str, required=True,
                      choices=['circular', 'forcedirected'],
                      help='Type of layout to process')
    parser.add_argument('--features', type=str, required=True,
                      help='Comma-separated list of feature types (onehot,positional,degree)')
    
    # Optional arguments
    parser.add_argument('--init-positions', type=str,
                      help='Path to initial positions file (required for force-directed)')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                      help='Output directory for processed data')
    parser.add_argument('--output-name', type=str,
                      help='Custom name for output file (default: processed_[layout-type]_[features].pt)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.layout_type == 'forcedirected' and not args.init_positions:
        parser.error("--init-positions is required for force-directed layout processing")
    
    # Parse and validate feature types
    feature_types = args.features.split(',')
    valid_features = {'onehot', 'positional', 'degree'}
    if not feature_types:
        parser.error("At least one feature type must be specified")
    if not all(f in valid_features for f in feature_types):
        parser.error(f"Invalid feature type(s). Valid options are: {', '.join(valid_features)}")
    
    # Load data
    layouts, adjacency_matrices, initial_positions = load_data(args)
    
    # Process data
    print(f"\nProcessing {args.layout_type} layouts with features: {', '.join(feature_types)}")
    dataset = prepare_dataset(
        layouts,
        adjacency_matrices,
        args.layout_type,
        feature_types,
        initial_positions
    )
    
    # Save processed data
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename
    if args.output_name:
        output_file = os.path.join(args.output_dir, args.output_name)
    else:
        output_file = os.path.join(
            args.output_dir,
            f"processed_{args.layout_type}_{'_'.join(feature_types)}.pt"
        )
    
    data_dict = {
        'dataset': dataset,
        'feature_types': feature_types,
        'layout_type': args.layout_type
    }
    
    print(f"\nSaving processed data to {output_file}")
    torch.save(data_dict, output_file)
    print("Done!")
    
    # Print dataset statistics
    print(f"\nDataset statistics:")
    print(f"Number of graphs: {len(dataset)}")
    if len(dataset) > 0:
        print(f"Node features shape: {dataset[0].x.shape}")
        print(f"Edge index shape: {dataset[0].edge_index.shape}")
        print(f"Layout coordinates shape: {dataset[0].y.shape}")
        if hasattr(dataset[0], 'edge_attr'):
            print(f"Edge attributes shape: {dataset[0].edge_attr.shape}")
        if hasattr(dataset[0], 'original_y'):
            print(f"Original coordinates shape: {dataset[0].original_y.shape}")
        if hasattr(dataset[0], 'init_coords'):
            print(f"Initial coordinates shape: {dataset[0].init_coords.shape}")

if __name__ == "__main__":
    main() 