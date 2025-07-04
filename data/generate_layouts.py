import argparse
import os
import pickle
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx, to_networkx

def assign_initial_positions(G, dim=2):
    """Grid-based initial positioning"""
    nodes = list(G.nodes())
    n = len(nodes)
    grid_size = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / grid_size))
    
    pos = {}
    for idx, node in enumerate(nodes):
        u = (idx % grid_size) / max(grid_size - 1, 1)  # in [0,1]
        v = (idx // grid_size) / max(rows - 1, 1)
        # remap to [-1,1]
        x = 2 * u - 1
        y = 2 * v - 1
        pos[node] = np.array([x, y], dtype=float)
    
    return pos

def get_adjacency_matrix(data):
    """Get binary adjacency matrix and node mapping from PyG Data"""
    G = to_networkx(data, to_undirected=True)
    
    nodes = list(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(nodes)}

    n = len(nodes)
    adjacency_matrix = np.zeros((n, n), dtype=float)

    # Fill adjacency matrix with 1.0 for edges (ignoring weights)
    for u, v in G.edges():
        i, j = node_mapping[u], node_mapping[v]
        adjacency_matrix[i, j] = 1.0
        adjacency_matrix[j, i] = 1.0  # symmetric

    return adjacency_matrix, node_mapping, G

def compute_adjacency_matrices(graphs_data):
    """Compute adjacency matrices for all graphs"""
    adjacency_matrices = []
    for data in graphs_data:
        adj_matrix, node_map, _ = get_adjacency_matrix(data)
        adjacency_matrices.append({
            'matrix': adj_matrix,
            'node_mapping': node_map,
            'graph_id': data.graph_id
        })
    return adjacency_matrices

def circular_layout(data):
    """Generate circular layout"""
    G = to_networkx(data, to_undirected=True)
    pos = nx.circular_layout(G)
    return pos

def fruchterman_reingold_layout(data, iterations=50, scale=1.0):
    """Generate Fruchterman-Reingold layout"""
    G = to_networkx(data, to_undirected=True)
    
    # Set edge weights if they exist
    has_weights = hasattr(data, 'edge_attr')
    if has_weights:
        edge_weights = data.edge_attr.squeeze().tolist()
        for i, (u, v) in enumerate(data.edge_index.t().tolist()):
            G[u][v]['weight'] = edge_weights[i]
    
    initial_pos = assign_initial_positions(G)
    pos = nx.spring_layout(
        G,
        iterations=iterations,
        scale=scale,
        seed=42,
        pos=initial_pos,
        weight='weight' if has_weights else None  # Only use weights if they exist
    )
    
    return pos, initial_pos

def forceatlas2_layout(data, iterations=100, scale=1.0):
    """Generate ForceAtlas2 layout"""
    G = to_networkx(data, to_undirected=True)
    
    # Set edge weights if they exist
    has_weights = hasattr(data, 'edge_attr')
    if has_weights:
        edge_weights = data.edge_attr.squeeze().tolist()
        for i, (u, v) in enumerate(data.edge_index.t().tolist()):
            G[u][v]['weight'] = edge_weights[i]
    
    initial_pos = assign_initial_positions(G)
    positions = nx.forceatlas2_layout(
        G,
        pos=initial_pos,
        max_iter=iterations,
        scaling_ratio=scale,
        seed=42,
        gravity=1.0,
        strong_gravity=True,
        jitter_tolerance=1.0,
        weight='weight' if has_weights else None  # Only use weights if they exist
    )
    
    return positions

def generate_circular_layouts(graphs_data):
    """Generate circular layouts for all graphs"""
    layouts = []
    for data in graphs_data:
        pos = circular_layout(data)
        layouts.append({
            'layout': pos,
            'graph_id': data.graph_id
        })
    return layouts

def generate_force_directed_layouts(graphs_data, algorithms):
    """Generate requested force-directed layouts"""
    layouts = {alg: [] for alg in algorithms}
    initial_positions = []
    
    for data in graphs_data:
        # Generate FR layout if requested
        if 'FR' in algorithms:
            fr_pos, init_pos = fruchterman_reingold_layout(data)
            layouts['FR'].append({
                'layout': fr_pos,
                'graph_id': data.graph_id
            })
            # Only add initial positions once per graph
            initial_positions.append({
                'layout': init_pos,
                'graph_id': data.graph_id
            })
        
        # Generate FA2 layout if requested
        if 'FA2' in algorithms:
            fa2_pos = forceatlas2_layout(data)
            layouts['FA2'].append({
                'layout': fa2_pos,
                'graph_id': data.graph_id
            })
    
    return layouts, initial_positions

def save_results(layouts, adjacency_matrices, output_dir, layout_type, initial_positions=None):
    """Save all results using torch.save"""
    # Create directories
    layout_dir = os.path.join('data', 'raw', 'layouts')
    adj_dir = os.path.join('data', 'raw', 'adjacency_matrices')
    os.makedirs(layout_dir, exist_ok=True)
    os.makedirs(adj_dir, exist_ok=True)
    
    # Save adjacency matrices
    adj_path = os.path.join(adj_dir, "combined_adjacency_matrices.pkl")
    torch.save(adjacency_matrices, adj_path)
    print(f"Saved adjacency matrices to {adj_path}")
    
    # Save layouts based on type
    if layout_type == 'circular':
        layout_path = os.path.join(layout_dir, "combined_circular_layouts.pkl")
        torch.save(layouts, layout_path)
        print(f"Saved circular layouts to {layout_path}")
    else:  # force-directed
        for alg, alg_layouts in layouts.items():
            layout_path = os.path.join(layout_dir, f"combined_{alg.lower()}_layouts.pkl")
            torch.save(alg_layouts, layout_path)
            print(f"Saved {alg} layouts to {layout_path}")
        
        # Save initial positions for force-directed layouts
        if initial_positions:
            init_path = os.path.join(layout_dir, "combined_initial_positions.pkl")
            torch.save(initial_positions, init_path)
            print(f"Saved initial positions to {init_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate layouts for graphs')
    parser.add_argument('--input', type=str, required=True,
                        help='Input pickle file containing PyG graphs')
    parser.add_argument('--layout-type', type=str, required=True, choices=['circular', 'force-directed'],
                        help='Type of layout to generate')
    parser.add_argument('--algorithms', type=str, default='FR',
                        help='Comma-separated list of force-directed algorithms (FR,FA2)')
    parser.add_argument('--output-dir', type=str, default='data/raw/layouts',
                        help='Directory to save the layouts')
    
    args = parser.parse_args()
    
    # Load input graphs
    print(f"Loading graphs from {args.input}...")
    with open(args.input, 'rb') as f:
        graphs_data = pickle.load(f)
    print(f"Loaded {len(graphs_data)} graphs")
    
    # Compute adjacency matrices
    print("Computing adjacency matrices...")
    adjacency_matrices = compute_adjacency_matrices(graphs_data)
    
    # Generate layouts based on type
    if args.layout_type == 'circular':
        print("Generating circular layouts...")
        layouts = generate_circular_layouts(graphs_data)
        initial_positions = None
    else:  # force-directed
        # Parse algorithms
        algorithms = [alg.strip().upper() for alg in args.algorithms.split(',')]
        valid_algorithms = {'FR', 'FA2'}
        if not all(alg in valid_algorithms for alg in algorithms):
            parser.error(f"Invalid algorithm(s). Valid options are: {', '.join(valid_algorithms)}")
        
        print(f"Generating layouts using algorithms: {', '.join(algorithms)}")
        layouts, initial_positions = generate_force_directed_layouts(graphs_data, algorithms)
    
    # Save results
    save_results(layouts, adjacency_matrices, args.output_dir, args.layout_type, initial_positions)
    print("Done!")

if __name__ == "__main__":
    main() 