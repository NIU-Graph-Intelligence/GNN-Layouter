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
bash generate_graph_data.sh --output <output_dir> --num_samples <num_samples> [--graph_type <graph_type>] [--num_nodes <num_nodes>] [--output_filename <output_filename>]
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
bash generate_layout.sh --dataset <dataset_path> --layout_output <layout_output_dir> --adjacency_output <adjacency_output_dir> --layout <layout_type>
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

### Arguments:
- `--batch_id ID` : Specify a batch identifier (default: 'default')
- `--adj_path PATH` : Path to adjacency matrices pickle file
- `--layout_path PATH` : Path to layout pickle file
- `--output_dir DIR` : Directory to save processed data (default: `data/processed`)
- `--help` : Display help message and exit

## 4. Model Training

### Usage
```bash
./model_training.sh --batch_id <id> [options]
```

### Arguments:
- `--model <name>` : Specify the model type (default: `GNN_Model_1`)
- `--data_path <path>` : Path to input dataset (default: `data/processed/modelInput.pt`)
- `--batch_size <num>` : Training batch size (default: 1)
- `--help` : Display help message and exit

