import argparse
import os
import pickle
import random
import networkx as nx
import sys

graph_dataset = {}

def BA_graph_dataset(num_samples, num_nodes, num_edges):
    graph_dataset["BA_graphs"] = []
    for _ in range(num_samples):
        G = nx.barabasi_albert_graph(n=num_nodes, m=random.randint(*num_edges))
        graph_dataset["BA_graphs"].append(G)

def ER_graph_dataset(num_samples, nodes):
    graph_dataset["ER_graphs"] = []
    samples_per_prob = num_samples // 3
    for p in [0.4, 0.5, 0.6]:
        for _ in range(samples_per_prob):
            G = nx.erdos_renyi_graph(nodes, p)
            graph_dataset["ER_graphs"].append(G)

def WS_graph_dataset(num_samples, num_nodes, nearest_neighbors, rewiring_probability):
    graph_dataset["WS_graphs"] = []
    for _ in range(num_samples):
        G = nx.watts_strogatz_graph(
            n=num_nodes,
            k=random.randint(*nearest_neighbors),
            p=random.uniform(*rewiring_probability)
        )
        graph_dataset["WS_graphs"].append(G)

def save_graph_dataset(save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    with open(file_path, "wb") as f:
        pickle.dump(graph_dataset, f)
    print(f"Graph dataset saved to {file_path}")

def usage():
    le = len(sys.argv)
    if le != 5:
        print("Usage: python generate_graph_data.py data/raw/graph_dataset 500 50 all")
        sys.exit(1)
    
    # Get arguments
    output_dir = sys.argv[1]
    num_samples = int(sys.argv[2])
    num_nodes = int(sys.argv[3])
    graph_type = sys.argv[4]
    
    # Validate graph type
    if graph_type not in ['ws', 'er', 'ba', 'all']:
        print("Usage: python generate_graph_data.py data/raw/graph_dataset 500 50 all")
        sys.exit(1)

    # Generate graphs based on type
    if graph_type in ["er", "all"]:
        print("Generating Erdős-Rényi (ER) graphs...")
        ER_graph_dataset(num_samples=num_samples, nodes=num_nodes)

    if graph_type in ["ws", "all"]:
        print("Generating Watts-Strogatz (WS) graphs...")
        WS_graph_dataset(num_samples=num_samples, num_nodes=num_nodes,
                         nearest_neighbors=(2, 10), rewiring_probability=(0.4, 0.6))

    if graph_type in ["ba", "all"]:
        print("Generating Barabási-Albert (BA) graphs...")
        BA_graph_dataset(num_samples=num_samples, num_nodes=num_nodes, num_edges=(2, 25))

    save_graph_dataset(save_dir=output_dir, filename="Kamada15000graphs_dataset.pkl")

# Call usage function directly
usage()
