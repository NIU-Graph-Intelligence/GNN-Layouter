import networkx as nx
import random
import pickle
import os
import numpy as np

# Global dictionary to store all graph datasets in a single structure
merged_graph_dataset = {}


def BA_graph_dataset(num_samples, num_nodes, num_edges):
    merged_graph_dataset["BA_graphs"] = []
    for _ in range(num_samples):
        G = nx.barabasi_albert_graph(
            n= num_nodes,
            m=random.randint(*num_edges)
        )
        merged_graph_dataset["BA_graphs"].append(G)

def ER_graph_dataset(num_samples, nodes):
    merged_graph_dataset["ER_graphs"] = []

    for _ in range(num_samples):
        # Create three graphs for each probability
        for p in [0.6, 0.5, 0.4]:
            for _ in range(3):
                G = nx.erdos_renyi_graph(nodes, p, seed=42)
                merged_graph_dataset["ER_graphs"].append(G)

def WS_graph_dataset(num_samples, num_nodes, nearest_neighbors, rewiring_probability):
    merged_graph_dataset["WS_graphs"] = []
    for _ in range(num_samples):
        G = nx.watts_strogatz_graph(
            n=num_nodes,
            k=random.randint(*nearest_neighbors),
            p=random.uniform(*rewiring_probability)
        )
        merged_graph_dataset["WS_graphs"].append(G)

def save_graph_dataset(save_dir="data/raw/graph_dataset", filename="merged_graph_dataset.pkl"):
    """
    Save the merged_graph_dataset dictionary to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    with open(file_path, "wb") as f:
        pickle.dump(merged_graph_dataset, f)
    print(f"Graph dataset saved to {file_path}")



if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate graph datasets.")
    parser.add_argument("--output", type=str, default="data/raw/graph_dataset", help="Output directory.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples per graph type.")
    parser.add_argument("--num_nodes", type=int, default=50, help="Number of nodes in the graph.")
    args = parser.parse_args()

    # Generate graphs
    ER_graph_dataset(num_samples=args.num_samples, nodes=args.num_nodes)
    WS_graph_dataset(num_samples=args.num_samples, num_nodes=args.num_nodes,
                     nearest_neighbors=(2, 10), rewiring_probability=(0.4, 0.6))
    BA_graph_dataset(num_samples=args.num_samples, num_nodes=args.num_nodes, num_edges=(2, 25))

    # Save the dataset
    save_graph_dataset(save_dir=args.output)