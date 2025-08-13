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
    """Generate Fruchterman-Reingold layout with robust error handling"""
    G = to_networkx(data, to_undirected=True)
    
    # Check for disconnected graph
    if not nx.is_connected(G):
        print(f"Warning: Graph {data.graph_id} is disconnected for FR layout. Using largest connected component.")
    
    # Set edge weights if they exist
    has_weights = hasattr(data, 'edge_attr')
    if has_weights:
        edge_weights = data.edge_attr.squeeze().tolist()
        for i, (u, v) in enumerate(data.edge_index.t().tolist()):
            G[u][v]['weight'] = edge_weights[i]
    
    initial_pos = assign_initial_positions(G)
    
    try:
        pos = nx.spring_layout(
            G,
            iterations=iterations,
            scale=scale,
            seed=42,
            pos=initial_pos,
            weight='weight' if has_weights else None
        )
        
        # Check for NaN/inf in positions
        for node, position in pos.items():
            if np.isnan(position).any() or np.isinf(position).any():
                print(f"Warning: FR layout produced NaN/Inf for graph {data.graph_id}. Using initial positions.")
                pos = initial_pos
                break
                
    except Exception as e:
        print(f"FR layout failed for graph {data.graph_id}: {e}. Using initial positions.")
        pos = initial_pos
    
    return pos, initial_pos

def forceatlas2_layout(data, iterations=100, scale=1.0):
    """Generate ForceAtlas2 layout with robust fallback handling"""
    G = to_networkx(data, to_undirected=True)
    original_nodes = set(range(data.x.shape[0]))
    
    # Always use all nodes, even if graph is disconnected
    # NetworkX can handle disconnected graphs
    
    # Set edge weights if they exist
    has_weights = hasattr(data, 'edge_attr')
    if has_weights:
        edge_weights = data.edge_attr.squeeze().tolist()
        for i, (u, v) in enumerate(data.edge_index.t().tolist()):
            if G.has_edge(u, v):
                G[u][v]['weight'] = edge_weights[i]
    
    # Create initial positions for ALL nodes
    initial_pos = assign_initial_positions(G)
    
    # Ensure all original nodes have initial positions
    for node in original_nodes:
        if node not in initial_pos:
            initial_pos[node] = np.random.uniform(-1, 1, 2)
    
    try:
        # Try ForceAtlas2 with conservative parameters
        positions = nx.forceatlas2_layout(
            G,
            pos=initial_pos,
            max_iter=min(iterations, 50),  # Limit iterations to prevent instability
            scaling_ratio=1.0,  # Use default scaling
            seed=42,
            gravity=0.01,  # Very low gravity
            strong_gravity=False,
            jitter_tolerance=0.01,  # Very low jitter tolerance
            weight='weight' if has_weights else None
        )
        
        # Validate all positions
        valid_positions = True
        for node, pos in positions.items():
            if np.isnan(pos).any() or np.isinf(pos).any():
                valid_positions = False
                break
        
        if not valid_positions:
            raise ValueError("NaN/Inf detected in ForceAtlas2 output")
            
        # Ensure we have positions for all original nodes
        for node in original_nodes:
            if node not in positions:
                positions[node] = initial_pos.get(node, np.array([0.0, 0.0]))
                
    except Exception as e:
        print(f"ForceAtlas2 failed for graph {data.graph_id}: {e}")
        print("Falling back to spring layout...")
        
        try:
            # Fallback to spring layout (more stable)
            positions = nx.spring_layout(
                G,
                pos=initial_pos,
                iterations=50,
                seed=42,
                weight='weight' if has_weights else None
            )
            
            # Validate spring layout positions
            for node, pos in positions.items():
                if np.isnan(pos).any() or np.isinf(pos).any():
                    raise ValueError("NaN/Inf in spring layout")
                    
        except Exception as e2:
            print(f"Spring layout also failed for graph {data.graph_id}: {e2}")
            print("Using initial positions as final fallback...")
            positions = initial_pos.copy()
    
    # Final validation: ensure all original nodes have valid positions
    for node in original_nodes:
        if node not in positions:
            positions[node] = np.array([0.0, 0.0])
        elif np.isnan(positions[node]).any() or np.isinf(positions[node]).any():
            positions[node] = initial_pos.get(node, np.array([0.0, 0.0]))
    
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
        # Prepare community information if available (for force-directed layouts)
        community_info = {}
        if hasattr(data, 'community') and hasattr(data, 'num_communities'):
            # Convert community tensor to dictionary format for compatibility
            community_dict = {}
            for node_idx, comm_id in enumerate(data.community):
                community_dict[node_idx] = comm_id.item()
            community_info = {
                'community': community_dict,
                'num_communities': data.num_communities.item()
            }
        
        # Generate FR layout if requested
        if 'FR' in algorithms:
            fr_pos, init_pos = fruchterman_reingold_layout(data)
            fr_layout_data = {
                'layout': fr_pos,
                'graph_id': data.graph_id
            }
            # Add community info to FR layout
            fr_layout_data.update(community_info)
            layouts['FR'].append(fr_layout_data)
            
            # Only add initial positions once per graph
            init_data = {
                'layout': init_pos,
                'graph_id': data.graph_id
            }
            # Add community info to initial positions too
            init_data.update(community_info)
            initial_positions.append(init_data)
        
        # Generate FA2 layout if requested
        if 'FA2' in algorithms:
            fa2_pos = forceatlas2_layout(data)
            fa2_layout_data = {
                'layout': fa2_pos,
                'graph_id': data.graph_id
            }
            # Add community info to FA2 layout
            fa2_layout_data.update(community_info)
            layouts['FA2'].append(fa2_layout_data)
    
    return layouts, initial_positions

def save_results(layouts, adjacency_matrices, output_dir, layout_type, initial_positions=None):
    """Save all results using torch.save"""
    # Create directories
    layout_dir = os.path.join('data', 'raw', 'layouts')
    adj_dir = os.path.join('data', 'raw', 'adjacency_matrices')
    os.makedirs(layout_dir, exist_ok=True)
    os.makedirs(adj_dir, exist_ok=True)
    
    # Save adjacency matrices
    adj_path = os.path.join(adj_dir, "LFR_adjacency_matrices5000.pkl")
    torch.save(adjacency_matrices, adj_path)
    print(f"Saved adjacency matrices to {adj_path}")
    
    # Save layouts based on type
    if layout_type == 'circular':
        layout_path = os.path.join(layout_dir, "combined_circular_layouts.pkl")
        torch.save(layouts, layout_path)
        print(f"Saved circular layouts to {layout_path}")
    else:  # force-directed
        for alg, alg_layouts in layouts.items():
            layout_path = os.path.join(layout_dir, f"LFR_{alg.lower()}_layouts5000.pkl")
            torch.save(alg_layouts, layout_path)
            print(f"Saved {alg} layouts to {layout_path}")
        
        # Save initial positions for force-directed layouts
        if initial_positions:
            init_path = os.path.join(layout_dir, "LFR_initial_positions5000.pkl")
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
    
    # Print community statistics
    community_graphs = sum(1 for g in graphs_data if hasattr(g, 'community'))
    non_community_graphs = len(graphs_data) - community_graphs
    print(f"Input graphs: {len(graphs_data)} total ({community_graphs} with communities, {non_community_graphs} without)")
    
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
        
        print(f"Generating force-directed layouts using algorithms: {', '.join(algorithms)}")
        print("Community information will be preserved in force-directed layouts")
        layouts, initial_positions = generate_force_directed_layouts(graphs_data, algorithms)
        
        # Print community preservation statistics
        for alg in algorithms:
            alg_layouts = layouts[alg]
            community_layouts = sum(1 for layout in alg_layouts if 'community' in layout)
            print(f"  {alg} layouts: {len(alg_layouts)} total ({community_layouts} with community info)")
    
    # Save results
    save_results(layouts, adjacency_matrices, args.output_dir, args.layout_type, initial_positions)
    print("Done!")

if __name__ == "__main__":
    main() 