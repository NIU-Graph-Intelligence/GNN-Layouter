# data/generate_layouts.py
import argparse
import networkx as nx
import numpy as np
import pickle
import os
import glob
import sys
from typing import List, Dict, Any, Tuple, Optional

# Add parent directory to path for importing config_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_utils import ConfigManager, load_experiment_config, find_files_by_patterns

def load_graphs(filepath: str) -> List[nx.Graph]:
    """Load graphs from file"""
    with open(filepath, 'rb') as f:
        graphs = pickle.load(f)
    return graphs['graphs']

def compute_circular_layout(graphs: List[nx.Graph]) -> List[Dict]:
    """Compute circular layouts"""
    layouts = []
    
    for G in graphs:
        pos = nx.circular_layout(G, scale=1.0)
        
        layout_data = {
            'graph_id': G.graph['id'],
            'layout_type': 'circular',
            'coordinates': {node: np.array(coord, dtype=np.float32) 
                          for node, coord in pos.items()}
        }
        
        layouts.append(layout_data)
    
    return layouts

def compute_spring_layout(graphs: List[nx.Graph], iterations: int = 50, 
                         seed: int = 42, scale: Optional[float] = None) -> Tuple[List[Dict], List[Dict]]:
    """Compute spring layouts, returns (layouts, initial_positions)"""
    layouts = []
    initial_positions = []
    
    for G in graphs:
        # Generate initial positions
        init_pos = _generate_grid_positions(G)
        
        # Compute spring layout
        pos = nx.spring_layout(G, pos=init_pos, iterations=iterations, seed=seed, scale=scale)
        
        layout_data = {
            'graph_id': G.graph['id'],
            'layout_type': 'spring',
            'coordinates': {node: np.array(coord, dtype=np.float32) 
                          for node, coord in pos.items()}
        }
        
        init_data = {
            'graph_id': G.graph['id'],
            'coordinates': init_pos
        }
        
        layouts.append(layout_data)
        initial_positions.append(init_data)
    
    return layouts, initial_positions

def _generate_grid_positions(G: nx.Graph) -> Dict[int, np.ndarray]:
    """Generate grid initial positions"""
    nodes = list(G.nodes())
    n = len(nodes)
    grid_size = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / grid_size))
    
    pos = {}
    for idx, node in enumerate(nodes):
        u = (idx % grid_size) / max(grid_size - 1, 1)
        v = (idx // grid_size) / max(rows - 1, 1)
        x = 2 * u - 1  # Map to [-1, 1]
        y = 2 * v - 1
        pos[node] = np.array([x, y], dtype=np.float32)
    
    return pos

def save_layouts(layouts: List[Dict], filepath: str,
                initial_positions: Optional[List[Dict]] = None):
    """Save layouts to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data = {
        'layouts': layouts,
        'initial_positions': initial_positions
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved layouts for {len(layouts)} graphs to {filepath}")

def generate_layout_filename(graph_filepath: str, layout_type: str) -> str:
    """Generate layout filename based on graph file"""
    base_name = os.path.splitext(os.path.basename(graph_filepath))[0]
    return f"{base_name}_{layout_type}_layout.pkl"

def generate_layouts_for_graph_file(graph_filepath: str, layout_config: Dict[str, Any],
                                  config_manager: ConfigManager) -> str:
    """Generate layouts for single graph file"""
    layout_type = layout_config['type']
    params = layout_config.get('params', {})
    save_initial = layout_config.get('save_initial', False)
    
    # Add default parameters
    if 'seed' not in params:
        params['seed'] = config_manager.get_default('seed', 42)
    
    print(f"Computing {layout_type} layout for {os.path.basename(graph_filepath)}")
    
    # Load graphs
    graphs = load_graphs(graph_filepath)
    
    # Compute layouts
    if layout_type == 'circular':
        layouts = compute_circular_layout(graphs)
        initial_positions = None
    elif layout_type == 'spring':
        layouts, initial_positions = compute_spring_layout(graphs, **params)
        if not save_initial:
            initial_positions = None
    else:
        raise ValueError(f"Unknown layout type: {layout_type}")
    
    # Generate output filename
    filename = generate_layout_filename(graph_filepath, layout_type)
    layouts_dir = config_manager.get_path('layouts')
    filepath = os.path.join(layouts_dir, filename)
    
    # Save layouts
    save_layouts(layouts, filepath, initial_positions)
    
    return filepath

def find_graph_files_for_prefix(prefix: str, config_manager: ConfigManager) -> List[str]:
    """Find graph files by prefix"""
    graphs_dir = config_manager.get_path('graphs')
    pattern = os.path.join(graphs_dir, f"{prefix}*.pkl")
    return glob.glob(pattern)

def main():
    parser = argparse.ArgumentParser(description='Generate graph layouts')
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
    
    print(f"Running layout generation for experiment: {experiment_config.get('experiment_name', 'unnamed')}")
    
    # Get layout configurations
    layout_configs = experiment_config.get('layouts', [])
    print(layout_configs)
    if not layout_configs:
        print("No layout configurations found in experiment config")
        return
    
    generated_files = []
    
    for layout_config in layout_configs:
        apply_to = layout_config.get('apply_to', [])
        if not apply_to:
            print(f"Warning: No 'apply_to' specified for layout {layout_config['type']}")
            continue
        
        # Generate layouts for each specified prefix
        for prefix in apply_to:
            graph_files = find_graph_files_for_prefix(prefix, config_manager)
            
            if not graph_files:
                print(f"Warning: No graph files found for prefix '{prefix}'")
                continue
            
            for graph_file in graph_files:
                try:
                    layout_file = generate_layouts_for_graph_file(
                        graph_file, layout_config, config_manager
                    )
                    generated_files.append(layout_file)
                except Exception as e:
                    print(f"Error generating layout for {graph_file}: {e}")
    
    print(f"\nGenerated {len(generated_files)} layout files:")
    for filepath in generated_files:
        print(f"  {filepath}")

if __name__ == "__main__":
    main()
