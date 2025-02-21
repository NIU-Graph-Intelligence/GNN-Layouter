import networkx as nx
import random
import pickle
import os
import numpy as np

def load_graph_dataset(file_path="data/raw/graph_dataset/merged_graph_dataset.pkl"):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def convert_to_adjacency_matrices(merged_graph_dataset):
    adjacency_matrice = {}
    for key, graphs in merged_graph_dataset.items():
        adjacency_matrice[key] = []
        for graph in graphs:
            adj_matrix = nx.adjacency_matrix(graph).todense()
            adjacency_matrice[key].append(adj_matrix)
    return adjacency_matrice

def save_adjacency_matrices(adjacency_matrices, save_dir="data/raw/adjacency_matrices"):
    os.makedirs(save_dir, exist_ok=True)
    for graph_type, matrices in adjacency_matrices.items():
        file_path = os.path.join(save_dir, f"{graph_type}.npy")
        np.save(file_path, matrices)  # Save as a NumPy array
    print(f"Adjacency matrices saved to {save_dir}")

def draw_circular_layouts(merged_graph_dataset):
    circular_layouts = {}
    for graph_type, graphs in merged_graph_dataset.items():
        circular_layouts[graph_type] = []
        for graph in graphs:
            layout = nx.circular_layout(graph)
            circular_layouts[graph_type].append(layout)
    return circular_layouts

def draw_shell_layouts(merged_graph_dataset):
    # Dictionary to store the circular layouts
    shell_layouts = {}
    for graph_type, graphs in merged_graph_dataset.items():
        shell_layouts[graph_type] = []

        for graph in graphs:
            # Generate 2D coordinates for the graph nodes
            # Check if the graph is valid
            if not isinstance(graph, nx.Graph):
                raise ValueError("Invalid graph passed to the function.")

            # Generate 2D coordinates for the graph nodes
            pos = nx.shell_layout(graph)

            # Ensure pos is populated
            if not pos:
                raise ValueError(f"Shell layout failed for graph {graph}")

            # Store the layout (node positions)
            shell_layouts[graph_type].append(pos)

    return shell_layouts

def draw_kamada_kawai(merged_graph_dataset):
    # Dictionary to store the circular layouts
    kamada_kawai_layouts = {}
    for graph_type, graphs in merged_graph_dataset.items():
        kamada_kawai_layouts[graph_type] = []
        for graph in graphs:
            # Generate 2D coordinates for the graph nodes
            # Check if the graph is valid
            if not isinstance(graph, nx.Graph):
                raise ValueError("Invalid graph passed to the function.")

            # Generate 2D coordinates for the graph nodes
            pos = nx.kamada_kawai_layout(graph)

            # Ensure pos is populated
            if not pos:
                raise ValueError(f"Kamada-Kawai layout failed for graph {graph}")

            # Store the layout (node positions)
            kamada_kawai_layouts[graph_type].append(pos)

    return kamada_kawai_layouts

def save_layouts(layouts, save_dir="data/raw/layouts"):
    os.makedirs(save_dir, exist_ok=True)
    for graph_type, layout_list in layouts.items():
        file_path = os.path.join(save_dir, f"{graph_type}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(layout_list, f)
    print(f"Layouts saved to {save_dir}")


