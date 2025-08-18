# data/generate_graphs.py
import argparse
import networkx as nx
import numpy as np
import pickle
import os
import sys
import random
from datetime import datetime
from typing import List, Dict, Any, Union

# Add parent directory to path for importing config_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_utils import ConfigManager, load_experiment_config

def parse_node_range(num_nodes_config: Union[int, str, Dict[str, int]]) -> tuple:
    """
    Parse node range configuration
    
    Args:
        num_nodes_config: Can be:
            - int: fixed number of nodes
            - str: range like "50-100" 
            - dict: {"min": 50, "max": 100}
    
    Returns:
        tuple: (min_nodes, max_nodes)
    """
    if isinstance(num_nodes_config, int):
        return num_nodes_config, num_nodes_config
    elif isinstance(num_nodes_config, str):
        if "-" in num_nodes_config:
            min_val, max_val = map(int, num_nodes_config.split("-"))
            return min_val, max_val
        else:
            val = int(num_nodes_config)
            return val, val
    elif isinstance(num_nodes_config, dict):
        return num_nodes_config["min"], num_nodes_config["max"]
    else:
        raise ValueError(f"Invalid num_nodes format: {num_nodes_config}")

def parse_range_config(config: Union[int, str, Dict[str, int]]) -> tuple:
    """
    Generic range parsing function for any parameter
    
    Args:
        config: Can be int, string range, or dict with min/max
        
    Returns:
        tuple: (min_val, max_val)
    """
    if isinstance(config, int):
        return config, config
    elif isinstance(config, str):
        if "-" in config:
            min_val, max_val = map(int, config.split("-"))
            return min_val, max_val
        else:
            val = int(config)
            return val, val
    elif isinstance(config, dict):
        return config["min"], config["max"]
    else:
        raise ValueError(f"Invalid range format: {config}")

def sample_num_nodes(min_nodes: int, max_nodes: int, seed: int = None) -> int:
    """Sample number of nodes from range"""
    if seed is not None:
        random.seed(seed)
    return random.randint(min_nodes, max_nodes)

def generate_er_graphs(num_graphs: int, num_nodes_config: Union[int, str, Dict], 
                      p: float = 0.1, seed: int = 42) -> List[nx.Graph]:
    """Generate Erdős-Rényi graphs with variable node counts"""
    graphs = []
    min_nodes, max_nodes = parse_node_range(num_nodes_config)
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    for i in range(num_graphs):
        # Sample number of nodes for this graph
        num_nodes = sample_num_nodes(min_nodes, max_nodes, seed + i if seed else None)
        
        current_seed = seed + i if seed else None
        # Ensure connectivity by regenerating if disconnected
        while True:
            G = nx.erdos_renyi_graph(num_nodes, p, seed=current_seed)
            if nx.is_connected(G):
                break
            if current_seed:
                current_seed += 1
        
        G.graph['id'] = f"ER_{i:04d}"
        G.graph['type'] = 'ER'
        G.graph['params'] = {'p': p, 'seed': current_seed, 'num_nodes': num_nodes}
        G.graph['has_communities'] = False
        graphs.append(G)
    
    return graphs

def generate_ba_graphs(num_graphs: int, num_nodes_config: Union[int, str, Dict], 
                      m: int = 3, seed: int = 42) -> List[nx.Graph]:
    """Generate Barabási-Albert graphs with variable node counts"""
    graphs = []
    min_nodes, max_nodes = parse_node_range(num_nodes_config)
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    for i in range(num_graphs):
        # Sample number of nodes for this graph
        num_nodes = sample_num_nodes(min_nodes, max_nodes, seed + i if seed else None)
        
        # Ensure m is valid for the number of nodes
        effective_m = min(m, num_nodes - 1)
        
        current_seed = seed + i if seed else None
        G = nx.barabasi_albert_graph(num_nodes, effective_m, seed=current_seed)
        
        G.graph['id'] = f"BA_{i:04d}"
        G.graph['type'] = 'BA'
        G.graph['params'] = {'m': effective_m, 'seed': current_seed, 'num_nodes': num_nodes}
        G.graph['has_communities'] = False
        graphs.append(G)
    
    return graphs

def generate_ws_graphs(num_graphs: int, num_nodes_config: Union[int, str, Dict], 
                      k: int = 6, p: float = 0.2, seed: int = 42) -> List[nx.Graph]:
    """Generate Watts-Strogatz graphs with variable node counts"""
    graphs = []
    min_nodes, max_nodes = parse_node_range(num_nodes_config)
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    for i in range(num_graphs):
        # Sample number of nodes for this graph
        num_nodes = sample_num_nodes(min_nodes, max_nodes, seed + i if seed else None)
        
        # Ensure k is valid for the number of nodes
        effective_k = min(k, num_nodes - 1)
        # k must be even for WS graphs
        if effective_k % 2 != 0:
            effective_k -= 1
        effective_k = max(2, effective_k)  # Minimum k = 2
        
        current_seed = seed + i if seed else None
        G = nx.watts_strogatz_graph(num_nodes, effective_k, p, seed=current_seed)
        
        G.graph['id'] = f"WS_{i:04d}"
        G.graph['type'] = 'WS'
        G.graph['params'] = {'k': effective_k, 'p': p, 'seed': current_seed, 'num_nodes': num_nodes}
        G.graph['has_communities'] = False
        graphs.append(G)
    
    return graphs

def generate_community_graphs_integrated(num_graphs: int, num_nodes_config: Union[int, str, Dict], 
                                       **params) -> List[nx.Graph]:
    """
    Generate community graphs using LFR benchmark
    
    Args:
        num_graphs: Number of graphs to generate
        num_nodes_config: Node count configuration 
        **params: LFR parameters (all standard LFR parameters supported)
    
    Returns:
        List of NetworkX graphs with community information
    """
    from community_graph_utils import generate_community_graphs_variable
    
    # LFR defaults
    default_params = {
        'avg_degree': 10,
        'max_degree': 50,
        'mixing_parameter': 0.1,
        'weight_mixing': 0.0,
        'degree_exponent': 2.0,
        'community_exponent': 1.0,
        'weight_exponent': 1.5,
        'min_community_size': 10,
        'max_community_size': 50,
        'timeout': 30,
        'seed': 42,
    }
    
    final_params = default_params.copy()
    final_params.update(params)
    
    return generate_community_graphs_variable(
        num_graphs=num_graphs,
        num_nodes_config=num_nodes_config,
        **final_params
    )

def save_graphs(graphs: List[nx.Graph], filepath: str):
    """Save graphs to file with metadata"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Add generation metadata
    metadata = {
        'generation_time': datetime.now().isoformat(),
        'num_graphs': len(graphs),
        'graph_types': list(set(g.graph.get('type', 'unknown') for g in graphs))
    }
    
    save_data = {
        'graphs': graphs,
        'metadata': metadata
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"Saved {len(graphs)} graphs to {filepath}")

def load_graphs_with_metadata(filepath: str) -> tuple:
    """
    Load graphs with metadata, handling both old and new formats
    
    Returns:
        tuple: (graphs, metadata)
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        # Old format: just graphs
        return data, {}
    elif isinstance(data, dict) and 'graphs' in data:
        # New format: graphs with metadata
        return data['graphs'], data.get('metadata', {})
    else:
        raise ValueError(f"Unknown file format in {filepath}")

def generate_graphs_from_config(graph_config: Dict[str, Any], 
                              config_manager: ConfigManager) -> str:
    """Generate graphs according to configuration"""
    graph_type = graph_config['type']
    num_graphs = graph_config['num_graphs']
    
    # Handle both old and new num_nodes format
    num_nodes_config = graph_config.get('num_nodes', config_manager.get_default('num_nodes', 50))
    params = graph_config.get('params', {})
    output_prefix = graph_config.get('output_prefix', f"{graph_type.lower()}")
    
    # Add default seed if not specified
    if 'seed' not in params:
        params['seed'] = config_manager.get_default('seed', 42)
    
    # Parse node range for logging
    min_nodes, max_nodes = parse_node_range(num_nodes_config)
    if min_nodes == max_nodes:
        print(f"Generating {num_graphs} {graph_type} graphs with {min_nodes} nodes...")
    else:
        print(f"Generating {num_graphs} {graph_type} graphs with {min_nodes}-{max_nodes} nodes...")
    
    # Generate graphs based on type
    try:
        if graph_type == 'ER':
            graphs = generate_er_graphs(num_graphs, num_nodes_config, **params)
        elif graph_type == 'BA':
            graphs = generate_ba_graphs(num_graphs, num_nodes_config, **params)
        elif graph_type == 'WS':
            graphs = generate_ws_graphs(num_graphs, num_nodes_config, **params)
        elif graph_type == 'Community':
            graphs = generate_community_graphs_integrated(num_graphs, num_nodes_config, **params)
            if not graphs:
                print(f"Warning: No {graph_type} graphs were successfully generated")
                return ""
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
    except Exception as e:
        print(f"Error generating {graph_type} graphs: {e}")
        return ""
    
    if not graphs:
        print(f"No graphs generated for type {graph_type}")
        return ""
    
    # Generate descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if min_nodes == max_nodes:
        filename = f"{output_prefix}_{num_graphs}graphs_{min_nodes}nodes_{timestamp}.pkl"
    else:
        filename = f"{output_prefix}_{num_graphs}graphs_{min_nodes}-{max_nodes}nodes_{timestamp}.pkl"
    
    # Save graphs
    graphs_dir = config_manager.get_path('graphs')
    filepath = os.path.join(graphs_dir, filename)
    save_graphs(graphs, filepath)
    
    # Print statistics
    print_graph_statistics(graphs)
    
    return filepath

def print_graph_statistics(graphs: List[nx.Graph]):
    """Print comprehensive statistics about generated graphs"""
    if not graphs:
        print("No graphs to analyze")
        return
    
    # Basic statistics
    node_counts = [G.number_of_nodes() for G in graphs]
    edge_counts = [G.number_of_edges() for G in graphs]
    avg_degrees = [2 * G.number_of_edges() / G.number_of_nodes() for G in graphs if G.number_of_nodes() > 0]
    
    print(f"\n=== Graph Statistics ===")
    print(f"Total graphs: {len(graphs)}")
    print(f"Node count: {min(node_counts)}-{max(node_counts)} (avg: {np.mean(node_counts):.1f})")
    print(f"Edge count: {min(edge_counts)}-{max(edge_counts)} (avg: {np.mean(edge_counts):.1f})")
    print(f"Average degree: {min(avg_degrees):.1f}-{max(avg_degrees):.1f} (avg: {np.mean(avg_degrees):.1f})")
    
    # Type-specific statistics
    graph_types = {}
    for G in graphs:
        gtype = G.graph.get('type', 'unknown')
        graph_types[gtype] = graph_types.get(gtype, 0) + 1
    
    print(f"Graph types: {dict(graph_types)}")
    
    # Community-specific statistics
    community_graphs = [G for G in graphs if G.graph.get('has_communities', False)]
    if community_graphs:
        num_comms = [G.graph.get('num_communities', 0) for G in community_graphs]
        mix_weights = [G.graph.get('params', {}).get('mix_weight', 0) for G in community_graphs]
        
        print(f"Community graphs: {len(community_graphs)}")
        print(f"Communities per graph: {min(num_comms)}-{max(num_comms)} (avg: {np.mean(num_comms):.1f})")
        if any(mw > 0 for mw in mix_weights):
            print(f"Mix weights: {min(mix_weights):.3f}-{max(mix_weights):.3f} (avg: {np.mean(mix_weights):.3f})")
    
    # Connectivity check
    connected_graphs = sum(1 for G in graphs if nx.is_connected(G))
    print(f"Connected graphs: {connected_graphs}/{len(graphs)} ({100*connected_graphs/len(graphs):.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Generate graph datasets')
    parser.add_argument('--config', required=True, 
                       help='Experiment configuration file name (without .yaml)')
    parser.add_argument('--global-config', default='../config.yaml',
                       help='Global configuration file path')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
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
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    experiment_name = experiment_config.get('experiment_name', 'unnamed')
    print(f"Running experiment: {experiment_name}")
    
    # Generate graphs according to configuration
    graph_configs = experiment_config.get('graphs', [])
    if not graph_configs:
        print("No graph configurations found in experiment config")
        return
    
    generated_files = []
    total_graphs = 0
    
    for i, graph_config in enumerate(graph_configs):
        try:
            print(f"\n--- Processing graph configuration {i+1}/{len(graph_configs)} ---")
            if args.verbose:
                print(f"Config: {graph_config}")
                
            filepath = generate_graphs_from_config(graph_config, config_manager)
            if filepath:
                generated_files.append(filepath)
                # Count graphs in this file
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and 'graphs' in data:
                        total_graphs += len(data['graphs'])
                    elif isinstance(data, list):
                        total_graphs += len(data)
            else:
                print(f"Failed to generate graphs for configuration {i+1}")
                
        except Exception as e:
            print(f"Error generating graphs for configuration {i+1}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Summary
    print(f"\n=== Generation Summary ===")
    print(f"Generated {len(generated_files)} graph files with {total_graphs} total graphs")
    print(f"Experiment: {experiment_name}")
    print(f"Output files:")
    for filepath in generated_files:
        print(f"  {filepath}")

if __name__ == "__main__":
    main()