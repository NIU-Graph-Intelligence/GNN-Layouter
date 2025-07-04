import argparse
import os
import pickle
import random
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import numpy as np

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

def convert_to_pyg_data(G, graph_id):
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
    
    return data

def generate_graphs(num_samples, num_nodes, graph_types, save_path, community_dataset_path="data/raw/graph_dataset/community_graphs_dataset40Nodes.pkl"):
    """Generate a mix of specified graph types (BA, ER, WS) and combine with community dataset
    
    Args:
        num_samples: Total number of graphs to generate
        num_nodes: Number of nodes in each graph
        graph_types: List of graph types to generate ('BA', 'ER', 'WS')
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
            data = convert_to_pyg_data(G, graph_id)
            graphs.append(data)

    # Now load and add the community dataset graphs
    print(f"\nLoading community dataset from {community_dataset_path}")
    try:
        with open(community_dataset_path, 'rb') as f:
            community_graphs = pickle.load(f)
        print(f"Loaded {len(community_graphs)} community graphs")
        
        # Add all community graphs to our collection
        graphs.extend(community_graphs)
        print(f"Combined dataset now has {len(graphs)} graphs total")
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
    parser = argparse.ArgumentParser(description='Generate graph dataset with specified types')
    parser.add_argument('--graph-types', type=str, default='ER,WS,BA',
                        help='Comma-separated list of graph types to generate (ER,WS,BA)')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Total number of graphs to generate')
    parser.add_argument('--num-nodes', type=int, default=40,
                        help='Number of nodes in each graph')
    parser.add_argument('--output', type=str, default='data/raw/graph_dataset/CodeRefactorgenerated_graphs.pkl',
                        help='Path to save the generated graphs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
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
    generate_graphs(args.num_samples, args.num_nodes, graph_types, args.output) 