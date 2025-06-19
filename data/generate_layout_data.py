import os
import pickle
import networkx as nx
import sys

def convert_to_sparse_adjacency(dataset_path):
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    adjacency_matrices = {}
    key_mapping = {
        'ER_graphs': 'ER',
        'WS_graphs': 'WS',
        'BA_graphs': 'BA'
    }

    for graph_type in data:
        short_key = key_mapping.get(graph_type)
        if short_key is None:
            continue
        
        adjacency_matrices[short_key] = []
        for graph in data[graph_type]:
            adj_matrix = nx.adjacency_matrix(graph).todense()
            adjacency_matrices[short_key].append(adj_matrix)

    return adjacency_matrices


def save_adjacency_data(adjacency_matrices, dataset_path):
    file_path = dataset_path
    with open(file_path, "wb") as f:
        pickle.dump(adjacency_matrices, f)
    print(f"Adjacency data saved to {dataset_path}")


def draw_circular_layouts(dataset_path):
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    key_mapping = {
        'ER_graphs': 'ER',
        'WS_graphs': 'WS',
        'BA_graphs': 'BA'
    }

    circular_layouts = {}
    for graph_type in data:
        short_key = key_mapping.get(graph_type, graph_type)
        circular_layouts[short_key] = []
        for graph in data[graph_type]:
            layout = nx.circular_layout(graph)
            circular_layouts[short_key].append(layout)

    return circular_layouts


def draw_shell_layouts(dataset_path):
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    key_mapping = {
        'ER_graphs': 'ER',
        'WS_graphs': 'WS',
        'BA_graphs': 'BA'
    }

    shell_layouts = {}
    for graph_type in data:
        short_key = key_mapping.get(graph_type, graph_type)
        shell_layouts[short_key] = []
        for graph in data[graph_type]:
            pos = nx.shell_layout(graph)
            shell_layouts[short_key].append(pos)

    return shell_layouts


def draw_kamada_kawai(dataset_path):
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    key_mapping = {
        'ER_graphs': 'ER',
        'WS_graphs': 'WS',
        'BA_graphs': 'BA'
    }

    kamada_kawai_layouts = {}
    for graph_type in data:
        short_key = key_mapping.get(graph_type, graph_type)
        kamada_kawai_layouts[short_key] = []
        for graph in data[graph_type]:
            pos = nx.kamada_kawai_layout(graph)
            kamada_kawai_layouts[short_key].append(pos)

    return kamada_kawai_layouts

def save_layouts(layouts, save_dir, layout_type):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{layout_type}_layouts.pkl"
    file_path = os.path.join(save_dir, filename)
    with open(file_path, "wb") as f:
        pickle.dump(layouts, f)
    print(f"Layouts saved to {file_path}")

def usage():
    le = len(sys.argv)
    if le != 5:
        print("Usage: python generate_layout_data.py data/raw/graph_dataset.pkl data/raw/layouts data/raw/adjacency layout_type")
        sys.exit(1)
    
    # Get arguments
    dataset = sys.argv[1]
    layout_output = sys.argv[2]
    adjacency_output = sys.argv[3]
    layout = sys.argv[4]
    
    # Validate layout type
    if layout not in ['circular', 'shell', 'kamada_kawai']:
        print("Usage: python generate_layout_data.py data/raw/graph_dataset.pkl data/raw/layouts data/raw/adjacency layout_type")
        sys.exit(1)

    print(f"Loading dataset from {dataset}")
    if layout == "circular":
        layouts = draw_circular_layouts(dataset)
    elif layout == "shell":
        layouts = draw_shell_layouts(dataset)
    elif layout == "kamada_kawai":
        layouts = draw_kamada_kawai(dataset)

    save_layouts(layouts, save_dir=layout_output, layout_type=layout)

    print("Generating adjacency matrices...")
    adjacency_matrices = convert_to_sparse_adjacency(dataset)
    adjacency_output_path = os.path.join(adjacency_output, "kamada15000graphs_adjacency_matrices.pkl")
    save_adjacency_data(adjacency_matrices, dataset_path=adjacency_output_path)

# Call usage function directly
if __name__ == "__main__":
       usage()
