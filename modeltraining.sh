#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 --batch_id <id> [options]"
    echo "Options:"
    echo "  --model <name>        Model type (default: GNN_Model_1)"
    echo "  --data_path <path>    Path to input dataset (default: data/processed/modelInput.pt)"
    echo "  --batch_size <num>    Training batch size (default: 1)"
    echo "  --help                Show this help message"
    exit 1
}

# Default values
MODEL="GNN_Model_1"
DATA_PATH=""
BATCH_SIZE=1
TEST_GEN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --data_path) DATA_PATH="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Set default data path if not provided
if [ -z "$DATA_PATH" ]; then
    DATA_PATH="data/processed/modelInput.pt"
fi

# Create output directory
OUTPUT_DIR="results/metrics"
mkdir -p "$OUTPUT_DIR"

# Run training
python3 - <<END

import torch
import os
import sys
sys.path.append('.')

# Import necessary modules dynamically
from data.dataset import data_loader
from training.train import train_model
from utils.visualization import *

# Import the specified model
model_name = "$MODEL"
if model_name == "GNN_Model_1":
    from models.gnn_model_1 import GNN_Model_1 as ModelClass
elif model_name == "GNN_Model_2":
    from models.gnn_model_2 import GNN_Model_2 as ModelClass
elif model_name == "GNN_Model_3":
    from models.gnn_model_3 import GNN_Model_3 as ModelClass
else:
    print(f"Error: Unknown model {model_name}")
    sys.exit(1)

# Load dataset
try:
    dataset = torch.load("$DATA_PATH")
    print(f"Dataset loaded from $DATA_PATH")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Create data loaders
train_loader, val_loader = data_loader(dataset)

# Initialize and train model
model_instance = ModelClass()
trained_model = train_model(model_instance, train_loader, val_loader, batch_size=$BATCH_SIZE)

# Save the trained model
save_path = os.path.join("$OUTPUT_DIR", f"best_modelTK.pt")
torch.save(trained_model.state_dict(), save_path)
print(f"Model saved to {save_path}")

END

if [ $? -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training failed"
    exit 1
fi