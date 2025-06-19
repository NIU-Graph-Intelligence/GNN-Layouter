import pickle
import torch
import os
import sys
import numpy as np
from torch_geometric.data import Data
import networkx as nx

def process_dictionary_data(adj_dict, coord_dict):
    processed_adj_matrices = []
    processed_coordinates = []

    # Only process the keys that exist in both dictionaries
    common_keys = set(adj_dict.keys()) & set(coord_dict.keys())

    for struct_type in common_keys:
        adj_matrices = adj_dict[struct_type]
        layouts = coord_dict[struct_type]

        for adj_matrix, layout in zip(adj_matrices, layouts):
            num_nodes = adj_matrix.shape[0]
            coords = np.zeros((num_nodes, 2))

            for node_idx, coord in layout.items():
                coords[node_idx] = coord

            processed_adj_matrices.append(np.array(adj_matrix))
            processed_coordinates.append(coords)

    return processed_adj_matrices, processed_coordinates

def create_node_features(num_nodes):
    indices = torch.arange(num_nodes, dtype=torch.float32)
    features = torch.zeros((num_nodes, 1))
    features[:, 0] = indices
    return features

def prepare_data(adj_matrices, coordinates, normalize=True):
    dataset = []
    for adj_mat, coords in zip(adj_matrices, coordinates):
        edge_index = torch.from_numpy(np.array(adj_mat.nonzero())).long()
        num_nodes = adj_mat.shape[0]

        # Create node features
        one_hot = torch.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            one_hot[i, i] = 1.0
        pos_features = create_node_features(num_nodes)
        # x = torch.cat([one_hot, pos_features], dim=1)  # with positional feature
        x = torch.cat([one_hot], dim=1)    # without positional feature

        # Normalize coordinates if requested
        coords = np.array(coords)
        if normalize:
            coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-5)
        y = torch.from_numpy(coords).float()

        # For Circular layout
        data = Data(x=x, edge_index=edge_index, y=y)


        # Calculate shortest paths for Kamada-Kawai
        # G = nx.from_numpy_array(adj_mat)
        # shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
        # # Convert shortest paths dict to tensor
        # sp_tensor = torch.zeros((num_nodes, num_nodes))
        # for i in range(num_nodes):
        #     for j in range(num_nodes):
        #         sp_tensor[i, j] = shortest_paths.get(i, {}).get(j, num_nodes)  # Use num_nodes as infinity
        # connected_mask = sp_tensor < num_nodes
        #
        # # Create PyG data object with shortest paths
        # data = Data(
        #     x=x,
        #     edge_index=edge_index,
        #     y=y,
        #     shortest_paths=sp_tensor,
        #     connected_mask=connected_mask,
        # )

        dataset.append(data)

    return dataset

def usage():
    le = len(sys.argv)
    if le != 6:
        print("Usage: python data_preprocessing.py data/raw/adjacency_matrices/adjacency_matrices.pkl data/raw/layouts/kamada_kawai_layouts.pkl data/processed -n layout_type")
        sys.exit(1)

    # Get arguments
    adj_path = sys.argv[1]
    layout_path = sys.argv[2]
    output_dir = sys.argv[3]
    mode_flag = sys.argv[4]
    layout_type = sys.argv[5]

    # Validate mode flag
    if mode_flag not in ['-n', '-u']:
        print("Usage: python data_preprocessing.py data/raw/adjacency_matrices/adjacency_matrices.pkl data/raw/layouts/kamada_kawai_layouts.pkl data/processed -n layout_type")
        sys.exit(1)

    # Set mode based on flag
    normalize = (mode_flag == '-n')
    mode = "normalized" if normalize else "unnormalized"
    print(f"Running in {mode} mode...")

    print(f"Loading adjacency matrices from {adj_path}")
    with open(adj_path, "rb") as f:
        adjacency_matrices = pickle.load(f)

    if adjacency_matrices is None:
        print('ERROR: Failed to load adjacency data')
        sys.exit(1)

    print(f"Loading layouts from {layout_path}")
    with open(layout_path, "rb") as f:
        layouts = pickle.load(f)

    print("Processing data...")
    adj_matrices, coordinates = process_dictionary_data(adjacency_matrices, layouts)

    # Find the maximum number of nodes in the dataset
    max_nodes = max(adj_mat.shape[0] for adj_mat in adj_matrices)
    print(f"Maximum number of nodes found: {max_nodes}")

    dataset = prepare_data(adj_matrices, coordinates, normalize=normalize)

    # Create a dictionary containing both the dataset and metadata
    data_dict = {
        'dataset': dataset,
        'max_nodes': max_nodes
    }

    # Set output filename based on mode and layout type
    output_filename = f"modelInput_{layout_type}.pt"
    if mode == "normalized":
        output_filename = "normalized_WithoutPositionalFeature" + output_filename
    elif mode == "unnormalized":
        output_filename = "unnormalized_" + output_filename

    output_file = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving processed data to {output_file}")
    torch.save(data_dict, output_file)
    print('Data preprocessing completed successfully!')

# Call usage function directly
usage()
