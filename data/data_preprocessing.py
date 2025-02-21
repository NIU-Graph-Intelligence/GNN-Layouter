import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import math

def process_dictionary_data(adj_dict, coord_dict):
    processed_adj_matrices = []
    processed_coordinates = []

    for struct_type in adj_dict.keys():
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
    "Create node features including normalized indices and positional encoding"
    indices = torch.arange(num_nodes, dtype=torch.float32)
    normalized_indices = 2 * math.pi * indices / num_nodes # Convert to angles

    # Create Features: [normalized_index, sin, cos]
    features = torch.zeros((num_nodes, 3))
    features[:, 0] = indices / num_nodes # Normalized index
    features[:, 1] = torch.sin(normalized_indices) # Sin component
    features[:, 2] = torch.cos(normalized_indices) # Cos component
    return features

def prepare_data(adj_matrices, coordinates):
    "Modified data preparation with both one-hot encoding and enhanced node features"

    dataset = []
    for adj_mat, coords in zip(adj_matrices, coordinates):
        # Convert adjacency matrix to edge indices
        edge_index = torch.from_numpy(np.array(adj_mat.nonzero())).long()

        # Get number of nodes
        num_nodes = adj_mat.shape[0]

        # Create one-hot encoding
        one_hot = torch.eye(50)

        # Create circular position features
        pos_features = create_node_features(num_nodes)

        # Concatenate features [50+3 = 53 features]
        x = torch.cat([one_hot, pos_features], dim=1) # Combined features

        # Normalize coordinates
        coords = np.array(coords)
        coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-5)

        # Convert coordinates to tensor
        y = torch.from_numpy(coords).float()

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    return dataset

def normalized_coordinates(coords):
    # Center the coordinates
    coords = coords - coords.mean(dim=0, keepdim=True)
    # scale to [-1,1] range
    max_dist = torch.max(torch.norm(coords, dim=1))
    return coords / ( max_dist + 1e-5)
