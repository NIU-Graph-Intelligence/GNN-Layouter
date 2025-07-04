#!/bin/bash

# Set paths
GRAPH_DIR="data/raw/custom_graph_dataset"
LAYOUT_DIR="data/raw/custom_layouts"
PROCESSED_DIR="data/processed"

# Step 1: Generate graphs
echo "Generating custom graphs..."
python3 data/generate_custom_graphs.py \
    --num-samples 1000 \
    --num-nodes 40 \
    --save-dir $GRAPH_DIR \
    --seed 42

# Step 2: Generate layouts
echo -e "\nGenerating FR layouts..."
python3 data/generate_custom_layouts.py \
    --graphs-path "$GRAPH_DIR/custom_graphs_dataset.pkl" \
    --save-dir $LAYOUT_DIR

# Step 3: Process data (normalized version)
echo -e "\nProcessing data (normalized)..."
python3 data/custom_preprocessing.py \
    --adj-path "$LAYOUT_DIR/custom_adjacency.pkl" \
    --layout-path "$LAYOUT_DIR/custom_fr_layouts.pkl" \
    --output-dir $PROCESSED_DIR \
    --normalize

echo -e "\nDataset generation complete!" 