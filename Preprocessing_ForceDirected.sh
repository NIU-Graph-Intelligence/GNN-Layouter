#!/bin/bash

#!/bin/bash

python3 data/Preprocessing_ForceDirected.py --adj-path data/raw/adjacency_matrices/Finaladjacency_matrix_1024.pkl --layout-path data/raw/layouts/FinalFR_1024.pt --dataset-path data/raw/graph_dataset/community_graphs_dataset1024.pkl --init-coord-path data/raw/layouts/FinalInit_1024.pt --community-path data/raw/layouts/Community_1024.pt --output-dir data/processed/ --layout-type FR --use-degree --use-positional-encoding --use-one-hot


#python3 data/Preprocessing_ForceDirected.py --adj-path data/raw/adjacency_matrices/Finaladjacency_matrix_1024.pkl --layout-path data/raw/layouts/FinalFR_1024.pt --dataset-path data/raw/graph_dataset/community_graphs_dataset1024.pkl --init-coord-path data/raw/layouts/FinalInit_1024.pt --community-path data/raw/layouts/Community_1024.pt --output-dir data/processed/ --layout-type FR
#python3 data/Preprocessing_ForceDirected.py --adj-path data/raw/adjacency_matrices/Finaladjacency_matrix_1024.pkl --layout-path data/raw/layouts/FinalFT2_1024.pt --dataset-path data/raw/graph_dataset/community_graphs_dataset1024.pkl --init-coord-path data/raw/layouts/FinalInit_1024.pt --community-path data/raw/layouts/Community_1024.pt --output-dir data/processed/ --layout-type FT2

# Generalization on more nodes
#python3 data/Preprocessing_ForceDirected.py --adj-path data/raw/adjacency_matrices/Finaladjacency_matrix_50Nodes.pkl --layout-path data/raw/layouts/FinalFR_50Nodes.pt --dataset-path data/raw/graph_dataset/community_graphs_dataset50Nodes.pkl --init-coord-path data/raw/layouts/FinalInit_50Nodes.pt --community-path data/raw/layouts/Community_1024.pt --output-dir data/processed/ --layout-type FR
# python3 data/Preprocessing_ForceDirected.py --adj-path data/raw/adjacency_matrices/Finaladjacency_matrix_1024.pkl --layout-path data/raw/layouts/FinalFR_1024.pt --dataset-path data/raw/graph_dataset/community_graphs_dataset1024.pkl --init-coord-path data/raw/layouts/FinalInit_1024.pt --community-path data/raw/layouts/Community_1024.pt --output-dir data/processed/ --layout-type FR


#python3 data/Preprocessing_ForceDirected.py --adj-path data/raw/adjacency_matrices/Finaladjacency_matrix_1024.pkl --layout-path data/raw/layouts/FinalFR_1024.pt --dataset-path data/raw/graph_dataset/community_graphs_dataset1024.pkl --init-coord-path data/raw/layouts/FinalInit_1024.pt --community-path data/raw/layouts/Community_1024.pt --output-dir data/processed/ --layout-type FR
#python3 data/Preprocessing_ForceDirected.py --adj-path data/raw/adjacency_matrices/Finaladjacency_matrix_1024.pkl --layout-path data/raw/layouts/FinalFT2_1024.pt --dataset-path data/raw/graph_dataset/community_graphs_dataset1024.pkl --init-coord-path data/raw/layouts/FinalInit_1024.pt --community-path data/raw/layouts/Community_1024.pt --output-dir data/processed/ --layout-type FT2

# Generalization on more nodes
#python3 data/Preprocessing_ForceDirected.py --adj-path data/raw/adjacency_matrices/Finaladjacency_matrix_50Nodes.pkl --layout-path data/raw/layouts/FinalFR_50Nodes.pt --dataset-path data/raw/graph_dataset/community_graphs_dataset50Nodes.pkl --init-coord-path data/raw/layouts/FinalInit_50Nodes.pt --community-path data/raw/layouts/Community_1024.pt --output-dir data/processed/ --layout-type FR
