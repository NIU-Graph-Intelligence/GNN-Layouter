# GNN-Layouter

A Graph Neural Network (GNN) based approach for generating and optimizing graph layouts, supporting multiple layout algorithms including Force-Directed and Circular layouts.

## Features

- **Graph Dataset Generation:** Generate synthetic graph data with different graph models (ER, WS, BA)
- **Multiple Layout Algorithms:** 
  - Force-Directed Layouts (FR and FA2)
  - Circular Layouts (from previous implementation)
- **Graph Processing:**
  - Adjacency Matrix Generation
  - Community Detection
  - Layout Generation and Optimization
- **Data Preprocessing:** Preprocess generated graph data for model training
- **Model Training:** Train GNN models on preprocessed graph data

## Project Structure

```
GNN-Layouter/
├── data/
│   ├── ForceDirectedLayouts.py   # Force-directed layout generation (FR and FA2)
│   └── raw/
│       ├── graph_dataset/        # Contains graph datasets
│       ├── layouts/             # Generated layouts
│       └── adjacency_matrices/  # Generated adjacency matrices
└── utils/                      # Utility functions and visualization tools
```

## Layout Algorithms

### Force-Directed Layouts
The project implements two popular force-directed layout algorithms:

1. **Fruchterman-Reingold (FR)** Layout
   - Physics-based layout algorithm
   - Balances attractive and repulsive forces between nodes
   - Suitable for medium-sized graphs
   - Generated using `ForceDirectedLayouts.py`

2. **ForceAtlas2 (FA2)** Layout
   - Continuous force-directed layout algorithm
   - Better for preserving community structures
   - Handles larger graphs efficiently
   - Generated using `ForceDirectedLayouts.py`

### Previous Layout Implementations
- Circular layouts

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/NIU-Graph-Intelligence/GNN-Layouter
cd GNN-Layouter
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Create Virtual Environment
python3 -m venv venv

# Activate Virtual Environment
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generating Force-Directed Layouts

The `ForceDirectedLayouts.py` script supports generating different types of layouts:

```bash
# Generate all layout types (adjacency matrices, FR, and FA2)
python ForceDirectedLayouts.py --generate-adj --generate-fr --generate-fa2

# Generate only Fruchterman-Reingold layouts
python ForceDirectedLayouts.py --generate-fr

# Generate only ForceAtlas2 layouts
python ForceDirectedLayouts.py --generate-fa2

# Generate layouts with visualizations
python ForceDirectedLayouts.py --generate-fr --generate-fa2 --visualize
```

### 2. Dataset
The current project uses a dataset of community graphs (`community_graphs_dataset1024.pkl`) containing:
- 1020 graphs
- 40 nodes per graph
- Community information for each node
- Edge attributes and weights

### 3. Output Files

The script generates several types of output files:

1. **Adjacency Matrices**: `data/raw/adjacency_matrices/Finaladjacency_matrix_1024.pkl`
2. **Layout Files**:
   - Initial Layouts: `data/raw/layouts/FinalInit_1024.pt`
   - FR Layouts: `data/raw/layouts/FinalFR_1024.pt`
   - FA2 Layouts: `data/raw/layouts/FinalFT2_1024.pt`

## Requirements

- Python 3.x
- PyTorch
- PyTorch Geometric
- NetworkX
- Matplotlib
- NumPy

## Note

This repository currently focuses on Force-Directed layouts as an alternative to previously implemented layouts. The Force-Directed approach often provides more natural and aesthetically pleasing layouts, especially for graphs with community structures.

# Machine Learning Based Graph Visualization

This project generates synthetic graph datasets using various models like Erdős-Rényi (ER), Watts-Strogatz (WS), and Barabási-Albert (BA).

## Features

- **Graph Dataset Generation:** Generate synthetic graph data with different graph models.
- **Graph Layout Generation:** Create various layouts such as circular, shell, and Kamada-Kawai for graph visualization.
- **Adjacency Matrix Generation:** Generate adjacency matrices for graph data.
- **Data Preprocessing:** Preprocess generated graph data for further analysis and model training.
- **Model Training:** Train machine learning models on preprocessed graph data.

## Prerequisites

- Python 3.x
- Bash shell (Linux/macOS)
- pip (Python package installer)

## Setup and Installation

### 1. Clone the Repository
Before running the project, clone the repository using the following command:

```bash
git clone https://github.com/NIU-Graph-Intelligence/GNN-Layouter
cd GNN-Layouter
```
### 2. Create and Activate a Virtual Environment ( Recommended )
It is recommended to use a virtual environment to avoid conflicts with system packages.

```bash
# Create Virtual Environment:
python3 -m venv venv

# Activate Virtual Environment:
venv\Scripts\activate (Windows)
source venv/bin/activate (macOS/Linux)

```

### 3. Install Dependencies
All required dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## 1. Data Generation

To generate the dataset, use the `generate_graph_data.sh` script. This script allows you to create different types of graph datasets with configurable parameters.

### Usage
Run the following command to generate graph data:

```bash
./generate_graph_data.sh --output <output_dir> --num_samples <num_samples> [--graph_type <graph_type>] [--num_nodes <num_nodes>] [--output_filename <output_filename>]
```

### Arguments:
- `--output` (required): Directory where the generated dataset will be saved.
- `--num_samples` (required): Number of samples to generate for each graph type.
- `--graph_type` (optional): Type of graph to generate. Choose from:
  - `ws` (Watts-Strogatz)
  - `er` (Erdős-Rényi)
  - `ba` (Barabási-Albert)  
  If not provided, all graph types will be generated.
- `--num_nodes` (optional): Number of nodes in the graph (default: 50).
- `--output_filename` (optional): Name of the output file (default: `graph_dataset.pkl`).
- `--help`: Displays the usage instructions.

## 2. Layout Generation for Graph Data

This script is designed to generate layouts and adjacency matrices for a given graph dataset. The layout types include **circular**, **shell**, and **kamada_kawai**. You can customize the layout generation process by specifying the dataset path, layout type, and output directories.

### Usage

To generate graph layouts and adjacency matrices from a given dataset, use the following command:

```bash
./generate_layout_data.sh --dataset <dataset_path> --layout_output <layout_output_dir> --adjacency_output <adjacency_output_dir> --layout <layout_type>
```

### Arguments:

- `--dataset` (required): Path to the input graph dataset (.pkl file).
- `--layout_output` (required): Directory where the generated layout files will be saved.
- `--adjacency_output` (required): Directory where the generated adjacency matrices will be saved.
- `--layout` (required): Type of layout to generate. Choose from:
  - `circular`: Generates a circular layout.
  - `shell`: Generates a shell layout.
  - `kamada_kawai`: Generates a Kamada-Kawai layout.

## 3. Data Preprocessing

### Usage
```bash
./data_preprocessing.sh [options]
```