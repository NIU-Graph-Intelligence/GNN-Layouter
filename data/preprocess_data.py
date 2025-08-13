import argparse
import pickle
import torch
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_utils.config_manager import ConfigManager

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
    nodes_with_coords = 0
    for node_idx in range(num_nodes):
        if node_idx in layout_dict:
            coords[node_idx] = layout_dict[node_idx]
            nodes_with_coords += 1
    
    # Check if we have enough valid coordinates
    if nodes_with_coords == 0:
        print(f"Warning: No coordinates found for circular layout graph {adj_data['graph_id']}")
        return None
    
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
    
    # Normalize coordinates with better handling of edge cases
    coord_mean = coords.mean(axis=0)
    coord_std = coords.std(axis=0)
    
    # Handle case where std is very small or zero
    coord_std = np.where(coord_std < 1e-6, 1.0, coord_std)
    
    # Normalize coordinates
    coords = (coords - coord_mean) / coord_std
    
    # Check for NaN or inf values after normalization
    if np.isnan(coords).any() or np.isinf(coords).any():
        print(f"Warning: NaN/Inf detected after normalization for circular layout graph {adj_data['graph_id']}")
        return None
    
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
    nodes_with_coords = 0
    for orig_id, mapped_idx in node_mapping.items():
        if orig_id in layout_dict:
            coords[mapped_idx] = layout_dict[orig_id]
            nodes_with_coords += 1
    
    # Check if we have enough valid coordinates
    if nodes_with_coords == 0:
        print(f"Warning: No coordinates found for graph {graph_id}")
        return None
    
    if nodes_with_coords < num_nodes:
        print(f"Warning: Only {nodes_with_coords}/{num_nodes} nodes have coordinates for graph {graph_id}")
    
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
    
    # Normalize coordinates with better handling of edge cases
    original_coords = coords.copy()
    
    # Check for valid coordinates (not all zeros)
    non_zero_mask = np.any(coords != 0, axis=1)
    if not np.any(non_zero_mask):
        print(f"Warning: All coordinates are zero for graph {graph_id}")
        return None
    
    # Calculate mean and std, handling edge cases
    coord_mean = coords.mean(axis=0)
    coord_std = coords.std(axis=0)
    
    # Handle case where std is very small or zero
    coord_std = np.where(coord_std < 1e-6, 1.0, coord_std)
    
    # Use milder normalization to preserve relative structure
    # Center the coordinates but use a global scale factor instead of per-dimension std
    centered_coords = coords - coord_mean
    global_scale = max(centered_coords.std(), 1e-6)  # Use overall standard deviation
    normalized_coords = centered_coords / global_scale
    
    # Alternative: Use range-based normalization to preserve aspect ratio
    # coord_range = np.ptp(coords, axis=0)  # peak-to-peak (max - min) per dimension
    # coord_range = np.where(coord_range < 1e-6, 1.0, coord_range)
    # normalized_coords = (coords - coord_mean) / coord_range.max()  # Use max range for isotropic scaling
    
    # Check for NaN or inf values after normalization
    if np.isnan(normalized_coords).any() or np.isinf(normalized_coords).any():
        print(f"Warning: NaN/Inf detected after normalization for graph {graph_id}")
        print(f"  Coordinate mean: {coord_mean}")
        print(f"  Coordinate std: {coord_std}")
        print(f"  Original coord range: [{coords.min():.6f}, {coords.max():.6f}]")
        return None
    
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
    
    # Add community information if available (for force-directed layouts)
    if 'community' in layout_data and 'num_communities' in layout_data:
        community_dict = layout_data['community']
        num_communities = layout_data['num_communities']
        
        # Convert community dictionary to tensor using node mapping
        community_tensor = torch.zeros(num_nodes, dtype=torch.long)
        for orig_id, mapped_idx in node_mapping.items():
            if orig_id in community_dict:
                community_tensor[mapped_idx] = community_dict[orig_id]
        
        data.community = community_tensor
        data.num_communities = torch.tensor(num_communities)
    
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
                # Only add to dataset if processing was successful (not None)
                if data is not None:
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
                # Only add to dataset if processing was successful (not None)
                if data is not None:
                    dataset.append(data)
    
    return dataset

def main():
    config = ConfigManager()
    
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
    parser.add_argument('--output-dir', type=str, default=config.get_data_path('processed'),
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
            f"processed5000_{args.layout_type}_{'_'.join(feature_types)}.pt"
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
    
    # Community statistics
    community_graphs = sum(1 for data in dataset if hasattr(data, 'community'))
    non_community_graphs = len(dataset) - community_graphs
    print(f"Graphs with community info: {community_graphs}")
    print(f"Graphs without community info: {non_community_graphs}")
    
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
        if hasattr(dataset[0], 'community'):
            print(f"Community tensor shape: {dataset[0].community.shape}")
            print(f"Number of communities: {dataset[0].num_communities.item()}")
            
        # Show community distribution example
        if community_graphs > 0:
            example_graph = next(data for data in dataset if hasattr(data, 'community'))
            unique_comms, counts = torch.unique(example_graph.community, return_counts=True)
            print(f"Example community distribution: {dict(zip(unique_comms.tolist(), counts.tolist()))}")

if __name__ == "__main__":
    main() 