import pickle
import torch
import os
import sys
import numpy as np
from torch_geometric.data import Data
import argparse
from torch_geometric.utils import degree

def process_community_data(adj_dict, layout_dict):

    processed_adj_matrices = []
    processed_coordinates  = []
    processed_mappings     = []
    graph_ids             = []

    common_keys = sorted(set(adj_dict.keys()) & set(layout_dict.keys()))
    print(f"Found {len(common_keys)} graphs in both adjacency and layout data")

    for graph_id in common_keys:
        adj_data = adj_dict[graph_id]
        layout   = layout_dict[graph_id]

        # Extract adjacency and original node-to-index mapping
        A = adj_data["matrix"]
        mapping = adj_data["node_mapping"]
        coords  = layout.numpy()

        processed_adj_matrices.append(A)
        processed_coordinates.append(coords)
        processed_mappings.append(mapping)
        graph_ids.append(graph_id)

    return processed_adj_matrices, processed_coordinates, processed_mappings, graph_ids

def prepare_data(
    adj_matrices,
    coordinates,
    mappings,
    graph_ids,
    community_dataset,
    raw_layout_dict,
    community_dict,
    init_coord_dict,
    use_degree=False,
    use_positional_encoding=False,
    use_one_hot=False
):
    dataset = []
    
    # Initialize features_used list based on arguments
    features_used = []
    if use_degree:
        features_used.append("degree")
    if use_positional_encoding:
        features_used.append("positional_encoding")
    if use_one_hot:
        features_used.append("one_hot")

    print(f"\nFeatures that will be used: {', '.join(features_used)}")

    id_to_index = { data.graph_id: i for i, data in enumerate(community_dataset) }

    for A, coords, mapping, graph_id in zip(adj_matrices, coordinates, mappings, graph_ids):
        if graph_id not in id_to_index:
            continue
        orig = community_dataset[id_to_index[graph_id]]
        community_tensor = orig.community

        # BUILD PyG STRUCTURES

        # 1) edges
        rows, cols = A.nonzero()
        edge_index = torch.tensor([rows, cols], dtype=torch.long)

        # Adjacency matrix connection through edge_weights
        edge_weights = A[rows, cols]
        edge_weights = edge_weights.astype(np.float32)

        # Normalize to [0, 1]
        min_val = edge_weights.min()
        max_val = edge_weights.max()
        if max_val > min_val:
            edge_weights = (edge_weights - min_val) / (max_val - min_val)
        else:
            edge_weights = np.zeros_like(edge_weights)  # All weights are the same

        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)

        N = A.shape[0]

        # 2) community features
        unique_comms = sorted(set(community_tensor.tolist()))
        comm_to_idx = {c: i for i, c in enumerate(unique_comms)}
        C = len(unique_comms)
        
        comm_feat = torch.zeros((N, C))
        for idx in range(N):
            print(idx)
            comm = int(community_tensor[idx].item())
            comm_feat[idx, comm_to_idx[comm]] = 1.0
        
        # Collect features based on arguments
        feature_list = []
        
        # 1. Node degrees (if requested)
        if use_degree:
            row, _ = edge_index
            deg = degree(row, num_nodes=N, dtype=torch.float32).unsqueeze(1)  # Shape: [N, 1]
            feature_list.append(deg)

        # 2. Positional Encoding (if requested)
        if use_positional_encoding:
            pos_encoding = torch.linspace(0, 1, N).unsqueeze(1)  # Shape: [N, 1]
            feature_list.append(pos_encoding)

        # 3. One-hot encoding (if requested)
        if use_one_hot:
            one_hot = torch.eye(N)  # Shape: [N, N]
            feature_list.append(one_hot)

        # Combine all requested features (we know there's at least one)
        x = torch.cat(feature_list, dim=1)  # Final shape: [N, num_features]

        # 3) coords
        original_coords   = coords.copy()
        normalized_coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-5)

        # For Visual
        orig_ids = [None] * N
        for orig_id, idx in mapping.items():
            orig_ids[idx] = orig_id
        orig_ids_tensor = torch.tensor(orig_ids, dtype=torch.long)

        # Initial coordinates
        init_tensor = init_coord_dict[graph_id]

        data = Data(
            x=x,                                              # Node features
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.from_numpy(normalized_coords).float(),
            original_y=torch.from_numpy(original_coords).float(),
            num_communities=torch.tensor(C),
            graph_id=graph_id,
            orig_node_ids=orig_ids_tensor,
        )
        data.init_coords = init_tensor

        dataset.append(data)

        # Print feature dimensions for the first graph
        if len(dataset) == 1:
            print(f"\nFeature dimensions for first graph:")
            print(f"Node features (x) shape: {data.x.shape}")
            
            # Calculate feature dimensions
            feature_start_idx = 0
            feature_dims = {}
            
            if use_degree:
                feature_dims["Degree feature"] = (feature_start_idx, feature_start_idx + 1)
                feature_start_idx += 1
                
            if use_positional_encoding:
                feature_dims["Positional encoding"] = (feature_start_idx, feature_start_idx + 1)
                feature_start_idx += 1
                
            if use_one_hot:
                feature_dims["One-hot encoding"] = (feature_start_idx, feature_start_idx + N)
            
            # Print feature dimensions
            for feature_name, (start, end) in feature_dims.items():
                if end - start == 1:
                    print(f"  - {feature_name}: {data.x[:, start].shape} (column {start})")
                else:
                    print(f"  - {feature_name}: {data.x[:, start:end].shape} (columns {start}-{end-1})")
            
            print(f"Edge attributes shape: {data.edge_attr.shape}")
            print(f"Initial coordinates shape: {data.init_coords.shape}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Process community FR/FA2 layouts for GNN training')
    parser.add_argument('--adj-path', type=str, required=True,
                        help='Path to adjacency matrices pickle (data/raw/adj_matrix/adj_dic)')
    parser.add_argument('--layout-path', type=str, required=True,
                        help='Path to FA2 layouts .pt (e.g. data/raw/layouts/FT2_1024.pt)')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to original community graph dataset pickle')
    parser.add_argument('--init-coord-path', type=str, required=True,
                        help='Path to initial coordinates .pt file')
    parser.add_argument('--community-path', type=str, required=True,
                        help='Path to community labels .pt file')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Directory to save processed .pt')
    parser.add_argument('--layout-type', type=str, required=True,
                        help='Layout type name (e.g. FR, FT2)')
    
    # Feature selection arguments
    feature_group = parser.add_argument_group('Feature Selection (at least one required)')
    feature_group.add_argument('--use-degree', action='store_true',
                        help='Use node degree as a feature')
    feature_group.add_argument('--use-positional-encoding', action='store_true',
                        help='Use positional encoding as a feature')
    feature_group.add_argument('--use-one-hot', action='store_true',
                        help='Use one-hot encoding as a feature')
    
    args = parser.parse_args()

    # Validate that at least one feature is selected
    if not any([args.use_degree, args.use_positional_encoding, args.use_one_hot]):
        parser.error("At least one feature must be selected (--use-degree, --use-positional-encoding, or --use-one-hot)")

    # Count selected features and warn if more than 3
    selected_features = sum([args.use_degree, args.use_positional_encoding, args.use_one_hot])
    if selected_features > 3:
        print("Warning: More than 3 features selected. This might impact model performance.")

    print("Selected features:")
    if args.use_degree:
        print("- Node degree")
    if args.use_positional_encoding:
        print("- Positional encoding")
    if args.use_one_hot:
        print("- One-hot encoding")

    print(f"Loading community dataset from {args.dataset_path}")
    with open(args.dataset_path, 'rb') as f:
        community_dataset = pickle.load(f)

    print(f"Loading adjacency dict from {args.adj_path}")
    with open(args.adj_path, 'rb') as f:
        adj_dict = pickle.load(f)

    print(f"Loading layouts from {args.layout_path}")
    layout_dict = torch.load(args.layout_path)

    print(f"Loading community labels from {args.community_path}")
    community_dict = torch.load(args.community_path)

    print(f"Loading initial coordinates from {args.init_coord_path}")
    init_dict = torch.load(args.init_coord_path)

    adj_ms, coords, maps, gids = process_community_data(adj_dict, layout_dict)
    print(f"Max nodes: {max(A.shape[0] for A in adj_ms)}")

    dataset = prepare_data(adj_ms, coords, maps, gids, community_dataset, layout_dict, 
                         community_dict, init_dict, args.use_degree, 
                         args.use_positional_encoding, args.use_one_hot)
    print(f"Created {len(dataset)} PyG Data objects")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create filename suffix based on features used
    feature_parts = []
    if args.use_degree:
        feature_parts.append("deg")
    if args.use_positional_encoding:
        feature_parts.append("pe")
    if args.use_one_hot:
        feature_parts.append("onehot")
        
    feature_suffix = "_" + "_".join(feature_parts)
        
    out_file = os.path.join(
        args.output_dir,
        f"modelInput_{args.layout_type}graphs1024_40Nodes{feature_suffix}.pt"
    )

    print(f"Saving processed data to {out_file}")
    torch.save({
        'dataset': dataset,
        'max_nodes': max(A.shape[0] for A in adj_ms),
        'layout_type': args.layout_type
    }, out_file)
    print("Done.")

if __name__ == '__main__':
    main()
