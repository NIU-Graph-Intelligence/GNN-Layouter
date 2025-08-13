import argparse
import os
import pickle
import random
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_utils.config_manager import ConfigManager

def assign_communities(G, num_communities):
    """Assign community labels to nodes using network structure"""
    # Use NetworkX's community detection
    try:
        from community import community_louvain
        communities = community_louvain.best_partition(G)
        # Remap community IDs to be 1-based like in the dataset
        unique_comms = sorted(set(communities.values()))
        comm_map = {old: new + 1 for new, old in enumerate(unique_comms)}
        communities = {node: comm_map[comm] for node, comm in communities.items()}
    except ImportError:
        # Fallback: random assignment if community detection package not available
        communities = {node: random.randint(1, num_communities) for node in G.nodes()}
    return communities

def convert_to_pyg_data(G, graph_id, add_communities=False):
    """Convert NetworkX graph to PyG Data object with same attributes as community dataset"""
    # Create node features (node indices as features)
    num_nodes = G.number_of_nodes()
    x = torch.arange(num_nodes, dtype=torch.float).view(-1, 1)  # [num_nodes, 1] feature with indices
    
    # Create edge list and edge weights
    edges = list(G.edges())
    
    # Create edge_index with both directions [2, num_edges*2]
    edge_index = torch.tensor([[u, v] for u, v in edges] + [[v, u] for u, v in edges], dtype=torch.long).t()
    
    # Create edge weights (1.0) for both directions [num_edges*2, 1]
    edge_attr = torch.ones(len(edges) * 2, dtype=torch.float).view(-1, 1)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
        graph_id=graph_id
    )
    
    # Add community information if requested
    if add_communities:
        communities_dict = assign_communities(G, num_communities=3)  # Default to 3 communities
        
        # Convert to tensor format (node index -> community id)
        community_tensor = torch.zeros(num_nodes, dtype=torch.long)
        for node, comm in communities_dict.items():
            community_tensor[node] = comm
        
        data.community = community_tensor
        data.num_communities = torch.tensor(len(set(communities_dict.values())))
    
    return data

def generate_graphs(num_samples, num_nodes, graph_types, save_path, community_dataset_path=None, add_communities_to_generated=False):
    """Generate a mix of specified graph types (BA, ER, WS) and combine with community dataset
    
    Args:
        num_samples: Total number of graphs to generate
        num_nodes: Number of nodes in each graph
        graph_types: List of graph types to generate ('BA', 'ER', 'WS')
        save_path: Path to save the generated graphs
        community_dataset_path: Path to community dataset (if None, uses config)
        add_communities_to_generated: Whether to add community detection to generated graphs
    """
    config = ConfigManager()
    
    # Use config path if not provided
    if community_dataset_path is None:
        community_dataset_path = os.path.join(
            config.get_data_path('graphs'), 
            'deepdrawingReproducecommunity_graphs_dataset.pkl'
        )
        save_path: Path to save the generated graphs
        community_dataset_path: Path to the existing community dataset
    """
    graphs = []
    samples_per_type = num_samples // len(graph_types)  # Equal distribution among types
    
    # First generate our new graphs
    for graph_type in graph_types:
        print(f"Generating {samples_per_type} {graph_type} graphs...")
        
        for i in range(samples_per_type):
            graph_id = f"graph_R{len(graphs)}"  # R prefix for random/generated graphs
            
            if graph_type == 'BA':
                # Barabási-Albert graphs - always connected by design
                m = max(2, num_nodes // 10)  # Increase minimum edges for better connectivity
                G = nx.barabasi_albert_graph(n=num_nodes, m=m)
                
            elif graph_type == 'ER':
                # Erdős-Rényi graphs - ensure higher edge probability
                # p > ln(n)/n ensures connectivity with high probability
                p = max(0.3, np.log(num_nodes) / num_nodes)
                while True:
                    G = nx.erdos_renyi_graph(num_nodes, p)
                    if nx.is_connected(G):
                        break
                
            elif graph_type == 'WS':
                # Watts-Strogatz graphs - ensure higher k and lower rewiring probability
                k = max(4, num_nodes // 5)  # Higher k ensures better connectivity
                p = 0.2  # Lower rewiring probability to maintain structure
                G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p)
            
            # Convert to PyG Data object
            data = convert_to_pyg_data(G, graph_id, add_communities=add_communities_to_generated)
            graphs.append(data)

    # Now load and add the community dataset graphs
    print(f"\nLoading community dataset from {community_dataset_path}")
    try:
        with open(community_dataset_path, 'rb') as f:
            community_graphs = pickle.load(f)
        print(f"Loaded {len(community_graphs)} community graphs")
        
        # Process community graphs to ensure they have community attributes
        processed_community_graphs = []
        for data in community_graphs:
            # Check if community information already exists
            if hasattr(data, 'community') and hasattr(data, 'num_communities'):
                processed_community_graphs.append(data)
            else:
                # Extract community information if it exists in the original format
                if hasattr(data, 'communities'):  # Old format
                    communities_dict = data.communities
                    community_tensor = torch.zeros(data.num_nodes, dtype=torch.long)
                    for node, comm in communities_dict.items():
                        community_tensor[node] = comm
                    
                    data.community = community_tensor
                    data.num_communities = torch.tensor(len(set(communities_dict.values())))
                
                processed_community_graphs.append(data)
        
        # Add all processed community graphs to our collection
        graphs.extend(processed_community_graphs)
        print(f"Combined dataset now has {len(graphs)} graphs total")
        
        # Print community statistics
        community_graphs_count = sum(1 for g in graphs if hasattr(g, 'community'))
        non_community_graphs_count = len(graphs) - community_graphs_count
        print(f"  - Graphs with community info: {community_graphs_count}")
        print(f"  - Graphs without community info: {non_community_graphs_count}")
        
    except FileNotFoundError:
        print(f"Warning: Community dataset not found at {community_dataset_path}")
        print("Proceeding with only generated graphs")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the combined graphs
    with open(save_path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"Saved combined dataset to {save_path}")
    
    return graphs

if __name__ == "__main__":
    config = ConfigManager()
    
    parser = argparse.ArgumentParser(description='Generate graph dataset with specified types')
    parser.add_argument('--graph-types', type=str, default='ER,WS,BA',
                        help='Comma-separated list of graph types to generate (ER,WS,BA)')
    parser.add_argument('--num-samples', type=int, default=config.get('data.generation.num_samples', 3),
                        help='Total number of graphs to generate')
    parser.add_argument('--num-nodes', type=int, default=config.get('data.generation.num_nodes', 40),
                        help='Number of nodes in each graph')
    parser.add_argument('--output', type=str, 
                        default=os.path.join(config.get_data_path('graphs'), 'deepdrawingReproduceFinal_graphs.pkl'),
                        help='Path to save the generated graphs')
    parser.add_argument('--seed', type=int, default=config.get('training.seed', 42),
                        help='Random seed for reproducibility')
    parser.add_argument('--add-communities-to-generated', action='store_true',
                        help='Add detected communities to generated graphs (ER, WS, BA)')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Parse and validate graph types
    graph_types = [t.strip().upper() for t in args.graph_types.split(',')]
    valid_types = {'ER', 'WS', 'BA'}
    if not all(t in valid_types for t in graph_types):
        invalid = [t for t in graph_types if t not in valid_types]
        print(f"Error: Invalid graph type(s): {invalid}")
        print(f"Valid types are: {', '.join(valid_types)}")
        exit(1)
        
    # Generate graphs
    generate_graphs(args.num_samples, args.num_nodes, graph_types, args.output, 
                   add_communities_to_generated=args.add_communities_to_generated) 