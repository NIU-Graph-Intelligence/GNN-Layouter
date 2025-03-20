#!/bin/bash

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --batch_id ID       Specify a batch identifier (default: 'default')"
    echo "  --adj_path PATH     Path to adjacency matrices pickle file"
    echo "  --layout_path PATH  Path to layout pickle file"
    echo "  --output_dir DIR    Directory to save processed data (default: data/processed)"
    echo "  --help              Display this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 --batch_id experiment1 --output_dir ./results/data"
    exit 1
}

# Default parameters
BATCH_ID="default"
ADJ_PATH="data/raw/adjacency_matrices/adjacency_matrices.pkl"
LAYOUT_PATH="data/raw/layouts/circular_layouts.pkl"
OUTPUT_DIR="data/processed"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --batch_id)
            BATCH_ID="$2"
            shift 2
            ;;
        --adj_path)
            ADJ_PATH="$2"
            shift 2
            ;;
        --layout_path)
            LAYOUT_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            echo ""
            usage
            ;;
    esac
done

# Print information
echo "==== Data Preprocessing ===="
echo "Batch ID: $BATCH_ID"
echo "Adjacency Matrix Path: $ADJ_PATH"
echo "Layout Path: $LAYOUT_PATH"
echo "Output Directory: $OUTPUT_DIR"

# Check if input files exist
if [ ! -f "$ADJ_PATH" ]; then
    echo "Error: Adjacency matrix file not found at $ADJ_PATH"
    exit 1
fi

if [ ! -f "$LAYOUT_PATH" ]; then
    echo "Error: Layout file not found at $LAYOUT_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script for data preprocessing
python3 - <<END

import pickle
import torch
import os
import sys
sys.path.append('.')  # Add current directory to path

from data.data_preprocessing import process_dictionary_data, prepare_data

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

print('Loading adjacency matrices from $ADJ_PATH')
adjacency_matrices = load_pickle('$ADJ_PATH')

if adjacency_matrices is None:
    print('ERROR: Failed to load adjacency data')
    sys.exit(1)

print('Loading layouts from $LAYOUT_PATH')
circular_layouts = load_pickle('$LAYOUT_PATH')

print('Processing data...')
adj_matrices, coordinates = process_dictionary_data(adjacency_matrices, circular_layouts)
dataset = prepare_data(adj_matrices, coordinates)

# Save processed data
output_file = os.path.join('$OUTPUT_DIR', 'modelInput_${BATCH_ID}.pt')
print(f'Saving processed data to {output_file}')
torch.save(dataset, output_file)
print('Data preprocessing completed successfully!')

END

# Check if the processing was successful
if [ $? -eq 0 ]; then
    echo "✓ Data preprocessing completed successfully"
    echo "Output saved to $OUTPUT_DIR/modelInput_${BATCH_ID}.pt"
else
    echo "✗ Data preprocessing failed"
    exit 1
fi