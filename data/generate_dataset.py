# data/generate_dataset.py
import argparse
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
import numpy as np
import networkx as nx
import pickle
import os
import sys
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path for importing config_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_utils import ConfigManager, load_experiment_config, find_files_by_patterns

def load_graphs(filepath: str) -> List[nx.Graph]:
    """Load graphs from file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'graphs' in data:
        return data['graphs']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown file format in {filepath}")

def load_layouts(filepath: str) -> Tuple[List[Dict], Optional[List[Dict]]]:
    """Load layouts from file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        return data['layouts'], data.get('initial_positions', None)
    else:
        return data, None

def compute_graph_level_features(graphs: List[nx.Graph]) -> Dict[str, torch.Tensor]:
    """Compute graph-level features for all graphs and return normalized versions"""
    
    # Compute raw features for all graphs
    graph_sizes = []
    graph_densities = []
    clustering_coeffs = []
    
    for G in graphs:
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        # Graph size (number of nodes)
        graph_sizes.append(num_nodes)
        
        # Graph density
        max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0.0
        graph_densities.append(density)
        
        # Average clustering coefficient
        clustering_coeff = nx.average_clustering(G)
        clustering_coeffs.append(clustering_coeff)
    
    # Convert to tensors
    graph_sizes = torch.tensor(graph_sizes, dtype=torch.float32)
    graph_densities = torch.tensor(graph_densities, dtype=torch.float32)
    clustering_coeffs = torch.tensor(clustering_coeffs, dtype=torch.float32)
    
    # Compute normalized versions (z-score normalization)
    def normalize_feature(feature_tensor):
        mean = feature_tensor.mean()
        std = feature_tensor.std()
        if std < 1e-6:  # Handle case where all values are the same
            return torch.zeros_like(feature_tensor)
        return (feature_tensor - mean) / std
    
    graph_densities_norm = normalize_feature(graph_densities)
    clustering_coeffs_norm = normalize_feature(clustering_coeffs)
    
    return {
        'graph_size': graph_sizes,
        'graph_density': graph_densities,
        'graph_density_norm': graph_densities_norm,
        'clustering_coefficient': clustering_coeffs,
        'clustering_coefficient_norm': clustering_coeffs_norm
    }

def create_node_features(graph_idx: int, num_nodes: int, edge_index: torch.Tensor, 
                        feature_config: Dict[str, bool],
                        initial_coords: Optional[np.ndarray] = None,
                        graph_level_features: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    """Create node features based on configuration"""
    features = []
    
    # Node-level features (excluding initial_position, which goes at the end)
    if feature_config.get('degree', False):
        row, _ = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)
        features.append(deg)
    
    if feature_config.get('positional', False):
        pos_feat = torch.linspace(0, 1, num_nodes).unsqueeze(1)
        features.append(pos_feat)
    
    if feature_config.get('onehot', False):
        # Check if all graphs have same number of nodes would be done at dataset level
        features.append(torch.eye(num_nodes))
    
    if feature_config.get('random', False):
        random_feat = torch.randn(num_nodes, 2)
        features.append(random_feat)
    
    # Graph-level features (same for all nodes in the graph)
    if graph_level_features is not None:
        if feature_config.get('graph_size', False):
            graph_size_feat = graph_level_features['graph_size'][graph_idx].unsqueeze(0).repeat(num_nodes, 1)
            features.append(graph_size_feat)
        
        if feature_config.get('graph_density', False):
            density_feat = graph_level_features['graph_density'][graph_idx].unsqueeze(0).repeat(num_nodes, 1)
            features.append(density_feat)
            
            if feature_config.get('graph_density_norm', False):
                density_norm_feat = graph_level_features['graph_density_norm'][graph_idx].unsqueeze(0).repeat(num_nodes, 1)
                features.append(density_norm_feat)
        
        if feature_config.get('clustering_coefficient', False):
            clustering_feat = graph_level_features['clustering_coefficient'][graph_idx].unsqueeze(0).repeat(num_nodes, 1)
            features.append(clustering_feat)
            
            if feature_config.get('clustering_coefficient_norm', False):
                clustering_norm_feat = graph_level_features['clustering_coefficient_norm'][graph_idx].unsqueeze(0).repeat(num_nodes, 1)
                features.append(clustering_norm_feat)
    
    # Always add initial_position at the end if available
    if feature_config.get('initial_position', False) and initial_coords is not None:
        init_pos_feat = torch.from_numpy(initial_coords).float()
        features.append(init_pos_feat)
    
    if not features:
        # Default to degree features if nothing specified
        row, _ = edge_index
        deg = degree(row, num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)
        features.append(deg)
    
    return torch.cat(features, dim=1)

def normalize_coordinates(coords: np.ndarray, method: str = 'standard') -> np.ndarray:
    """Normalize coordinates"""
    if method == 'standard':
        mean = coords.mean(axis=0)
        std = coords.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return (coords - mean) / std
    elif method == 'center':
        return coords - coords.mean(axis=0)
    else:
        return coords

def process_graph_layout_pair(graph_idx: int, graph: nx.Graph, layout_data: Dict, 
                            initial_positions_data: Optional[Dict],
                            feature_config: Dict[str, bool],
                            graph_level_features: Optional[Dict[str, torch.Tensor]] = None) -> Optional[Data]:
    """Process single graph-layout pair"""
    try:
        num_nodes = graph.number_of_nodes()
        graph_id = graph.graph['id']
        
        # Create edge index (bidirectional)
        edges = list(graph.edges())
        if not edges:
            return None
        
        edge_list = edges + [(v, u) for u, v in edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Extract final coordinates
        coord_dict = layout_data['coordinates']
        coordinates = np.array([coord_dict[node] for node in range(num_nodes)])
        
        # Normalize coordinates
        normalized_coords = normalize_coordinates(coordinates)
        
        # Check validity
        if np.isnan(normalized_coords).any() or np.isinf(normalized_coords).any():
            return None
        
        # Extract initial coordinates if available
        initial_coords = None
        if initial_positions_data is not None:
            initial_coord_dict = initial_positions_data['coordinates']
            initial_coords = np.array([initial_coord_dict[node] for node in range(num_nodes)])
            # Normalize initial coordinates with same method
            initial_coords = normalize_coordinates(initial_coords)
            
            # Check validity
            if np.isnan(initial_coords).any() or np.isinf(initial_coords).any():
                initial_coords = None
        
        # Check onehot feature compatibility
        if feature_config.get('onehot', False):
            # This check would ideally be done at dataset level, but we can warn here
            print(f"Warning: onehot feature enabled for graph with {num_nodes} nodes. "
                  f"Ensure all graphs have the same number of nodes.")
        
        # Check initial_position feature availability
        if feature_config.get('initial_position', False) and initial_coords is None:
            raise ValueError(f"initial_position feature requested but not available for graph {graph_id}")
        
        # Create features (initial_position will be at the end if included)
        x = create_node_features(graph_idx, num_nodes, edge_index, feature_config, 
                               initial_coords, graph_level_features)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.from_numpy(normalized_coords).float(),
            graph_id=graph_id,
            graph_type=graph.graph.get('type', 'unknown'),
            layout_type=layout_data.get('layout_type', 'unknown')
        )
        
        # Add original coordinates
        data.original_y = torch.from_numpy(coordinates).float()
        
        # Store feature configuration for reference
        data.has_initial_position = feature_config.get('initial_position', False)
        if data.has_initial_position:
            data.initial_position_dims = (x.size(1) - 2, x.size(1))  # Last 2 dimensions
        
        # Add community information for visualization (but not as features)
        if 'community' in graph.nodes[0]:  # Check if community info exists
            community_tensor = torch.zeros(num_nodes, dtype=torch.long)
            for node in range(num_nodes):
                if 'community' in graph.nodes[node]:
                    community_tensor[node] = graph.nodes[node]['community']
            data.community = community_tensor
            data.num_communities = torch.tensor(len(set(community_tensor.tolist())))
        
        return data
        
    except Exception as e:
        print(f"Error processing graph {graph.graph.get('id', 'unknown')}: {e}")
        return None

def find_matching_layout_file(graph_file: str, layout_type: str, 
                            config_manager: ConfigManager) -> Optional[str]:
    """Find matching layout file for graph file"""
    graph_basename = os.path.splitext(os.path.basename(graph_file))[0]
    layout_pattern = f"{graph_basename}_{layout_type}_layout.pkl"
    
    layouts_dir = config_manager.get_path('layouts')
    layout_path = os.path.join(layouts_dir, layout_pattern)
    
    if os.path.exists(layout_path):
        return layout_path
    else:
        return None

def validate_feature_config(feature_config: Dict[str, bool], graphs: List[nx.Graph], 
                          has_initial_positions: bool):
    """Validate feature configuration against available data"""
    
    # Check onehot compatibility
    if feature_config.get('onehot', False):
        node_counts = [G.number_of_nodes() for G in graphs]
        if len(set(node_counts)) > 1:
            raise ValueError(f"onehot feature requires all graphs to have same number of nodes. "
                           f"Found node counts: {set(node_counts)}")
    
    # Check initial_position availability
    if feature_config.get('initial_position', False) and not has_initial_positions:
        raise ValueError("initial_position feature requested but initial positions not available in layout data")

def create_dataset_from_config(dataset_config: Dict[str, Any], 
                             config_manager: ConfigManager) -> List[Data]:
    """Create dataset from configuration"""
    dataset_name = dataset_config['name']
    graph_patterns = dataset_config['graph_patterns']
    layout_type = dataset_config['layout_type']
    feature_config = dataset_config.get('features', {'degree': True})
    
    print(f"Creating dataset '{dataset_name}' with {layout_type} layout")
    print(f"Enabled features: {[k for k, v in feature_config.items() if v]}")
    
    # Special note about initial_position feature placement
    if feature_config.get('initial_position', False):
        print("Note: initial_position feature will be placed at the end of feature matrix (last 2 dimensions)")
    
    # Find matching graph files
    graphs_dir = config_manager.get_path('graphs')
    graph_files = find_files_by_patterns(graphs_dir, graph_patterns)
    
    if not graph_files:
        print(f"Warning: No graph files found for patterns: {graph_patterns}")
        return []
    
    print(f"Found {len(graph_files)} graph files")
    
    # Collect all graphs for graph-level feature computation
    all_graphs = []
    all_layouts = []
    all_initial_positions = []
    graph_to_file_mapping = []
    
    for graph_file in graph_files:
        print(f"Loading {os.path.basename(graph_file)}")
        
        # Find corresponding layout file
        layout_file = find_matching_layout_file(graph_file, layout_type, config_manager)
        if not layout_file:
            raise ValueError(f"No {layout_type} layout file found for {os.path.basename(graph_file)}")
        
        # Load data
        graphs = load_graphs(graph_file)
        layouts, initial_positions = load_layouts(layout_file)
        
        # Create layout lookup dictionary
        layout_dict = {layout['graph_id']: layout for layout in layouts}
        initial_dict = {}
        if initial_positions:
            initial_dict = {pos['graph_id']: pos for pos in initial_positions}
        
        # Add graphs with their corresponding data
        for graph in graphs:
            graph_id = graph.graph['id']
            if graph_id in layout_dict:
                all_graphs.append(graph)
                all_layouts.append(layout_dict[graph_id])
                all_initial_positions.append(initial_dict.get(graph_id, None))
                graph_to_file_mapping.append(graph_file)
            else:
                print(f"Warning: No layout found for graph {graph_id}")
    
    if not all_graphs:
        print("No valid graph-layout pairs found")
        return []
    
    # Validate feature configuration
    has_initial_positions = any(pos is not None for pos in all_initial_positions)
    validate_feature_config(feature_config, all_graphs, has_initial_positions)
    
    # Compute graph-level features if needed
    graph_level_features = None
    if any(feature_config.get(f, False) for f in ['graph_size', 'graph_density', 'graph_density_norm', 
                                                  'clustering_coefficient', 'clustering_coefficient_norm']):
        print("Computing graph-level features...")
        graph_level_features = compute_graph_level_features(all_graphs)
    
    # Process each graph
    dataset = []
    processed_count = 0
    
    for idx, (graph, layout_data, initial_pos_data) in enumerate(zip(all_graphs, all_layouts, all_initial_positions)):
        data = process_graph_layout_pair(idx, graph, layout_data, initial_pos_data, 
                                       feature_config, graph_level_features)
        if data is not None:
            dataset.append(data)
            processed_count += 1
    
    print(f"Successfully processed {processed_count} graphs for dataset '{dataset_name}'")
    return dataset

def save_dataset(dataset: List[Data], filename: str, config_manager: ConfigManager):
    """Save dataset"""
    processed_dir = config_manager.get_path('processed')
    filepath = os.path.join(processed_dir, filename)
    
    torch.save(dataset, filepath)
    print(f"Saved dataset with {len(dataset)} graphs to {filepath}")
    
    # Print statistics
    _print_dataset_statistics(dataset)
    
    return filepath

def _print_dataset_statistics(dataset: List[Data]):
    """Print dataset statistics"""
    if not dataset:
        print("Empty dataset")
        return
    
    print(f"\nDataset Statistics:")
    print(f"  Total graphs: {len(dataset)}")
    
    # Statistics by type
    graph_types = {}
    layout_types = {}
    community_count = 0
    initial_position_count = 0
    
    for data in dataset:
        gt = getattr(data, 'graph_type', 'unknown')
        lt = getattr(data, 'layout_type', 'unknown')
        graph_types[gt] = graph_types.get(gt, 0) + 1
        layout_types[lt] = layout_types.get(lt, 0) + 1
        if hasattr(data, 'community'):
            community_count += 1
        if getattr(data, 'has_initial_position', False):
            initial_position_count += 1
    
    print(f"  Graph types: {dict(graph_types)}")
    print(f"  Layout types: {dict(layout_types)}")
    print(f"  Graphs with community info: {community_count}")
    print(f"  Graphs with initial_position feature: {initial_position_count}")
    print(f"  Feature dimensions: {dataset[0].x.shape[1]}")
    
    if initial_position_count > 0:
        sample_data = next(data for data in dataset if getattr(data, 'has_initial_position', False))
        print(f"  Initial position dimensions: {sample_data.initial_position_dims}")
    
    # Graph size statistics
    node_counts = [data.num_nodes for data in dataset]
    edge_counts = [data.edge_index.shape[1] // 2 for data in dataset]  # Undirected graphs
    
    print(f"  Average nodes: {np.mean(node_counts):.1f} (range: {min(node_counts)}-{max(node_counts)})")
    print(f"  Average edges: {np.mean(edge_counts):.1f} (range: {min(edge_counts)}-{max(edge_counts)})")

def main():
    parser = argparse.ArgumentParser(description='Generate ML datasets from graphs and layouts')
    parser.add_argument('--config', required=True,
                       help='Experiment configuration file name (without .yaml)')
    parser.add_argument('--global-config', default='../config.yaml',
                       help='Global configuration file path')
    
    args = parser.parse_args()
    
    # Initialize configuration manager
    config_manager = ConfigManager(args.global_config)
    config_manager.ensure_directories()
    
    # Load experiment configuration
    try:
        experiment_config = load_experiment_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"Running dataset generation for experiment: {experiment_config.get('experiment_name', 'unnamed')}")
    
    # Get dataset configurations
    dataset_configs = experiment_config.get('datasets', [])
    if not dataset_configs:
        print("No dataset configurations found in experiment config")
        return
    
    generated_files = []
    
    for dataset_config in dataset_configs:
        try:
            dataset = create_dataset_from_config(dataset_config, config_manager)
            
            if dataset:
                filename = f"{dataset_config['name']}.pt"
                filepath = save_dataset(dataset, filename, config_manager)
                generated_files.append(filepath)
            else:
                print(f"Warning: Empty dataset for '{dataset_config['name']}'")
                
        except Exception as e:
            print(f"Error creating dataset '{dataset_config['name']}': {e}")
            raise  # Re-raise to see the full traceback
    
    print(f"\nGenerated {len(generated_files)} dataset files:")
    for filepath in generated_files:
        print(f"  {filepath}")

if __name__ == "__main__":
    main()