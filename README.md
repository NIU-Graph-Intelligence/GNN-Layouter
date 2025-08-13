# GNN-Layouter

A comprehensive Graph Neural Network (GNN) based framework for generating and optimizing graph layouts, supporting multiple layout algorithms including Force-Directed and Circular layouts with advanced deep learning architectures.

## 🚀 Features

### **Graph Neural Network Models**
- **ForceGNN**: Specialized GNN for force-directed layout generation with physics-inspired loss functions
- **ChebConv**: Chebyshev spectral graph convolution networks for efficient graph processing
- **GAT (Graph Attention Networks)**: Attention-based models for layout generation with node importance
- **GIN (Graph Isomorphism Networks)**: Powerful graph neural networks with theoretical guarantees
- **GCN (Graph Convolutional Networks)**: Classic graph convolution for baseline comparisons

### **Layout Algorithms**
- **Force-Directed Layouts**: 
  - Fruchterman-Reingold (FR) algorithm
  - ForceAtlas2 (FA2) algorithm
- **Circular Layouts**: Traditional circular arrangement for comparison

### **Advanced Features**
- **Centralized Configuration Management**: Single `config.json` for all parameters
- **Modular Architecture**: Flexible and extensible codebase design
- **Multiple Optimizers**: Adam, SGD, and custom optimization strategies
- **Comprehensive Evaluation**: MSE, visualization, and layout quality metrics
- **Batch Processing**: Efficient handling of large graph datasets

## 📁 Project Structure

```
GNN-Layouter/
├── config.json                          # Centralized configuration file
├── config_utils/                        # Configuration management
│   ├── config_manager.py                # Configuration loading and access
│   └── visualization.py                 # Unified visualization tools
├── data/                                # Data handling and preprocessing
│   ├── dataset.py                       # Dataset loaders and utilities
│   ├── generate_graphs.py               # Graph generation scripts
│   ├── generate_layouts.py              # Layout generation utilities
│   └── preprocess_data.py               # Data preprocessing pipelines
├── models/                              # Neural network architectures
│   ├── ChebConv.py                      # Chebyshev convolution models
│   ├── GAT.py                           # Graph Attention Networks
│   ├── GIN.py                           # Graph Isomorphism Networks
│   ├── GCN.py                           # Graph Convolutional Networks
│   ├── GCNFR.py                         # Force-directed GNN (ForceGNN)
│   ├── mlp_layers.py                    # Modular MLP components
│   └── coordinate_layers.py             # Coordinate transformation layers
├── training/                            # Training and evaluation
│   ├── train.py                         # Main training script
│   ├── trainer.py                       # Training pipeline classes
│   └── Eval_MSE.py                      # Model evaluation and metrics
|   └── evaluation.py                    # Loss Function and loader
├── results/                             # Training results and outputs
├── requirements.txt                     # Python dependencies
└── *.sh                                 # Shell scripts for automation
```

## 🛠️ Setup and Installation

### **Prerequisites**
- Python 3.8+ (recommended: Python 3.9-3.11)
- CUDA-compatible GPU (optional, for faster training)


### **1. Clone the Repository**
```bash
git clone https://github.com/NIU-Graph-Intelligence/GNN-Layouter.git
cd GNN-Layouter
```

### **2. Create Virtual Environment (Recommended)**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

```


## ⚙️ Configuration

The project uses a centralized configuration system through `config.json`. All model parameters, training settings, and paths are managed from this single file.


## 🚀 Usage

### **1. Data Generation**

Generate synthetic graph datasets with various properties:

```bash
# Generate graphs with different models (ER, WS, BA)
python data/generate_graphs.py --num_samples 1000 --num_nodes 40

# Generate layouts for existing graphs
python data/generate_layouts.py --dataset data/processed/graphs.pkl
```

### **2. Training Models**

#### **Basic Training**
```bash
# Train ForceGNN model on force-directed layouts
python training/train.py --model ForceGNN --layout_type force_directed

# Train GAT model on circular layouts  
python training/train.py --model GAT --layout_type circular

# Train with custom parameters (overrides config defaults)
python training/train.py --model ChebConv --batch_size 128 --learning_rate 0.001
```

### **3. Model Evaluation**

```bash
# Evaluate all trained models
python training/Eval_MSE.py

# Evaluate specific model
python training/Eval_MSE.py --model ForceGNN --layout_type force_directed

# Generate evaluation visualizations
python training/Eval_MSE.py --visualize --output_dir results/eval/
```

### **4. Visualization**

```bash
# Generate layout visualizations
python config_utils/visualization.py --model ForceGNN --num_samples 10

# Compare multiple models
python config_utils/visualization.py --compare --models ForceGNN GAT ChebConv

# Create training progress plots
python config_utils/visualization.py --plot_training --model_path results/training/
```

