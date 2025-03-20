import os
import pickle
import networkx as nx


def convert_to_sparse_adjacency(dataset_path):
    file_path = dataset_path
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # Initialize dictionary to store adjacency matrices
    adjacency_matrices = {'WS': [], 'ER': [], 'BA': []}

    # Map from your data keys to the shorter keys used in adjacency_matrices
    key_mapping = {
        'ER_graphs': 'ER',
        'WS_graphs': 'WS',
        'BA_graphs': 'BA'
    }

    for graph_type in data:  # Iterate through 'ER_graphs', 'WS_graphs', 'BA_graphs'
        print(f"Processing {graph_type} graphs...")
        short_key = key_mapping.get(graph_type)
        if short_key is None:
            continue  # Skip if no mapping exists

        for graph in data[graph_type]:
            adj_matrix = nx.adjacency_matrix(graph).todense()  # Use .todense() for numpy array
            adjacency_matrices[short_key].append(adj_matrix)

    return adjacency_matrices


def save_adjacency_data(adjacency_matrices, dataset_path):
    file_path = dataset_path
    with open(file_path, "wb") as f:
        pickle.dump(adjacency_matrices, f)
    print(f"Adjacency data saved to {file_path}")


def draw_circular_layouts(dataset_path):
    # Load the graph dataset from file
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    print(f"Data contains {len(data)} graph types")

    # Map from your data keys to the shorter keys
    key_mapping = {
        'ER_graphs': 'ER',
        'WS_graphs': 'WS',
        'BA_graphs': 'BA'
    }

    circular_layouts = {}
    # Process each graph type
    for graph_type in data:  # Iterate through 'ER_graphs', 'WS_graphs', 'BA_graphs'
        print(f"Processing {graph_type} graphs...")
        short_key = key_mapping.get(graph_type, graph_type)  # Use mapping or original key if no mapping
        circular_layouts[short_key] = []

        for i, graph in enumerate(data[graph_type]):
            print(f"  Processing graph {i}, nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
            layout = nx.circular_layout(graph)
            circular_layouts[short_key].append(layout)

    return circular_layouts


def draw_shell_layouts(dataset_path):
    # Load the graph dataset from file
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    # Map from your data keys to the shorter keys
    key_mapping = {
        'ER_graphs': 'ER',
        'WS_graphs': 'WS',
        'BA_graphs': 'BA'
    }

    shell_layouts = {}
    # Process each graph type
    for graph_type in data:  # Iterate through 'ER_graphs', 'WS_graphs', 'BA_graphs'
        print(f"Processing {graph_type} graphs...")
        short_key = key_mapping.get(graph_type, graph_type)
        shell_layouts[short_key] = []

        for i, graph in enumerate(data[graph_type]):
            print(f"  Processing graph {i}, nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
            # Validate the graph instance
            if not isinstance(graph, nx.Graph):
                raise ValueError(f"Invalid graph passed to the function: {type(graph)}")
            pos = nx.shell_layout(graph)
            if not pos:
                raise ValueError(f"Shell layout failed for graph {i} of type {graph_type}")
            shell_layouts[short_key].append(pos)

    return shell_layouts


def draw_kamada_kawai(dataset_path):
    # Load the graph dataset from file
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    # Map from your data keys to the shorter keys
    key_mapping = {
        'ER_graphs': 'ER',
        'WS_graphs': 'WS',
        'BA_graphs': 'BA'
    }

    kamada_kawai_layouts = {}
    # Process each graph type
    for graph_type in data:  # Iterate through 'ER_graphs', 'WS_graphs', 'BA_graphs'
        print(f"Processing {graph_type} graphs...")
        short_key = key_mapping.get(graph_type, graph_type)
        kamada_kawai_layouts[short_key] = []

        for i, graph in enumerate(data[graph_type]):
            print(f"  Processing graph {i}, nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
            # Validate the graph instance
            if not isinstance(graph, nx.Graph):
                raise ValueError(f"Invalid graph passed to the function: {type(graph)}")
            pos = nx.kamada_kawai_layout(graph)
            if not pos:
                raise ValueError(f"Kamada-Kawai layout failed for graph {i} of type {graph_type}")
            kamada_kawai_layouts[short_key].append(pos)

    return kamada_kawai_layouts


def save_layouts(layouts, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for graph_type, layout_list in layouts.items():
        file_path = os.path.join(save_dir, f"{graph_type}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(layout_list, f)
    print(f"Layouts saved to {save_dir}")

def save_layouts(layouts, save_dir, filename="circular_layouts.pkl"):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    with open(file_path, "wb") as f:
        pickle.dump(layouts, f)
    print(f"All layouts saved to {file_path}")


# import os
# import pickle
# import networkx as nx
#
#
# def convert_to_sparse_adjacency(dataset_path):
#
#     file_path = dataset_path
#     with open(file_path, "rb") as f:
#         data = pickle.load(f)
#     adjacency_matrices = {'WS': [], 'ER': [], 'BA': []}
#
#     for graph_type in data['graphs']:
#         print(f"Processing {graph_type} graphs...")
#         for graph in data['graphs'][graph_type]:
#             adj_matrix = nx.adjacency_matrix(graph).todense()  # Use .todense() for numpy array
#             adjacency_matrices[graph_type].append(adj_matrix)
#
#     return adjacency_matrices
#
# def save_adjacency_data(adjacency_matrices, dataset_path):
#     file_path = dataset_path
#     with open(file_path, "wb") as f:
#         pickle.dump(adjacency_matrices, f)
#     print(f"Adjacency data saved to {file_path}")
#
# def draw_circular_layouts(dataset_path):
#     # Load the graph dataset from file
#     with open(dataset_path, "rb") as f:
#         data = pickle.load(f)
#
#     print(data)
#     print("Nodes:", data.nodes())
#     print("Edges:", data.edges())
#
#     circular_layouts = {}
#     # Process each graph type (e.g., 'WS', 'ER', 'BA')
#     for graph_type in data['graphs']:
#         print(f"Processing {graph_type} graphs...")
#         circular_layouts[graph_type] = []
#         for graph in data['graphs'][graph_type]:
#             layout = nx.circular_layout(graph)
#             circular_layouts[graph_type].append(layout)
#     return circular_layouts
#
# def draw_shell_layouts(dataset_path):
#     # Load the graph dataset from file
#     with open(dataset_path, "rb") as f:
#         data = pickle.load(f)
#
#     shell_layouts = {}
#     # Process each graph type
#     for graph_type in data['graphs']:
#         print(f"Processing {graph_type} graphs...")
#         shell_layouts[graph_type] = []
#         for graph in data['graphs'][graph_type]:
#             # Validate the graph instance
#             if not isinstance(graph, nx.Graph):
#                 raise ValueError("Invalid graph passed to the function.")
#             pos = nx.shell_layout(graph)
#             if not pos:
#                 raise ValueError(f"Shell layout failed for graph {graph}")
#             shell_layouts[graph_type].append(pos)
#     return shell_layouts
#
# def draw_kamada_kawai(dataset_path):
#     # Load the graph dataset from file
#     with open(dataset_path, "rb") as f:
#         data = pickle.load(f)
#
#     kamada_kawai_layouts = {}
#     # Process each graph type
#     for graph_type in data['graphs']:
#         print(f"Processing {graph_type} graphs...")
#         kamada_kawai_layouts[graph_type] = []
#         for graph in data['graphs'][graph_type]:
#             # Validate the graph instance
#             if not isinstance(graph, nx.Graph):
#                 raise ValueError("Invalid graph passed to the function.")
#             pos = nx.kamada_kawai_layout(graph)
#             if not pos:
#                 raise ValueError(f"Kamada-Kawai layout failed for graph {graph}")
#             kamada_kawai_layouts[graph_type].append(pos)
#     return kamada_kawai_layouts
#
# def save_layouts(layouts, save_dir):
#     os.makedirs(save_dir, exist_ok=True)
#     for graph_type, layout_list in layouts.items():
#         file_path = os.path.join(save_dir, f"{graph_type}.pkl")
#         with open(file_path, "wb") as f:
#             pickle.dump(layout_list, f)
#     print(f"Layouts saved to {save_dir}")
