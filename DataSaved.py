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

def save_graph_dataset(save_dir="data/graph_dataset"):
    """
    Save the merged_graph_dataset dictionary to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "merged_graph_dataset.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(merged_graph_dataset, f)
    print(f"Graph dataset saved to {save_dir}")

def convert_to_adjacency_matrices():
    adjacency_matrice = {}
    for key, graphs in merged_graph_dataset.items():
        adjacency_matrice[key] = []
        for graph in graphs:
            adj_matrix = nx.adjacency_matrix(graph).todense()
            adjacency_matrice[key].append(adj_matrix)
    return adjacency_matrice

def save_adjacency_matrices(adjacency_matrices, save_dir="data/adjacency_matrices"):
    os.makedirs(save_dir, exist_ok=True)
    for graph_type, matrices in adjacency_matrices.items():
        file_path = os.path.join(save_dir, f"{graph_type}.npy")
        np.save(file_path, matrices)  # Save as a NumPy array
    print(f"Adjacency matrices saved to {save_dir}")

def draw_circular_layouts():
    circular_layouts = {}
    for graph_type, graphs in merged_graph_dataset.items():
        circular_layouts[graph_type] = []
        for graph in graphs:
            layout = nx.circular_layout(graph)
            circular_layouts[graph_type].append(layout)
    return circular_layouts

def draw_shell_layouts():
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

def draw_kamada_kawai():
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

def save_layouts(layouts, save_dir="data/layouts"):
    os.makedirs(save_dir, exist_ok=True)
    for graph_type, layout_list in layouts.items():
        file_path = os.path.join(save_dir, f"{graph_type}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(layout_list, f)
    print(f"Layouts saved to {save_dir}")


# Generate datasets
ER_graph_dataset(num_samples=500, nodes=50)
WS_graph_dataset(num_samples=4500, num_nodes=50, nearest_neighbors=(2, 10),
                 rewiring_probability=(0.3999999, 0.59999999))
BA_graph_dataset(num_samples=3000, num_nodes= 50, num_edges=(2, 25))

save_graph_dataset()

# Process and convert data
adjacency_matrices = convert_to_adjacency_matrices()
save_adjacency_matrices(adjacency_matrices)

circular_layouts = draw_circular_layouts()
save_layouts(circular_layouts)

shell_layouts =draw_shell_layouts()
save_layouts(shell_layouts)

kamada_kawai = draw_kamada_kawai()
save_layouts(kamada_kawai)
