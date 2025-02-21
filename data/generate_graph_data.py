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

def save_graph_dataset(save_dir="data/raw/graph_dataset"):
    """
    Save the merged_graph_dataset dictionary to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "merged_graph_dataset.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(merged_graph_dataset, f)
    print(f"Graph dataset saved to {save_dir}")


