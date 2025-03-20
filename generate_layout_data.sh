#!/bin/bash

# Function to display usage
usage() {
  echo "Usage: $0 --dataset <dataset_path> --layout_output <layout_output_dir> --adjacency_output <adjacency_output_dir> --layout <layout_type>"
  echo "  --dataset: Path to the input graph dataset (.pkl file)."
  echo "  --layout_output: Directory to save the generated layouts."
  echo "  --adjacency_output: Directory to save the generated adjacency matrices."
  echo "  --layout: Type of layout to generate (circular, shell, kamada_kawai)."
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    --layout_output)
      LAYOUT_OUTPUT="$2"
      shift 2
      ;;
    --adjacency_output)
      ADJACENCY_OUTPUT="$2"
      shift 2
      ;;
    --layout)
      LAYOUT_TYPE="$2"
      shift 2
      ;;
    --help)
      usage
      ;;
    *)
      usage
      ;;
  esac
done
# Validate arguments
if [ -z "$DATASET_PATH" ] || [ -z "$LAYOUT_OUTPUT" ] || [ -z "$ADJACENCY_OUTPUT" ] || [ -z "$LAYOUT_TYPE" ]; then
  echo "Error: Missing required arguments."
  usage
fi

# Validate layout type
if [[ ! "$LAYOUT_TYPE" =~ ^(circular|shell|kamada_kawai)$ ]]; then
  echo "Error: Invalid layout type. Use 'circular', 'shell', or 'kamada_kawai'."
  usage
fi

# Create output directory if it doesn't exist
mkdir -p "$LAYOUT_OUTPUT"
mkdir -p "$ADJACENCY_OUTPUT"

# Run the Python layout generation script
echo "Generating $LAYOUT_TYPE layouts and adjacency matrices for dataset: $DATASET_PATH"

python3 - <<END
import os
import pickle
import networkx as nx
from data.generate_layout_data import draw_circular_layouts, draw_shell_layouts, draw_kamada_kawai, save_layouts, convert_to_sparse_adjacency, save_adjacency_data

print("Loading dataset and generating layouts...")
if "$LAYOUT_TYPE" == "circular":
    layouts = draw_circular_layouts("$DATASET_PATH")
elif "$LAYOUT_TYPE" == "shell":
    layouts = draw_shell_layouts("$DATASET_PATH")
elif "$LAYOUT_TYPE" == "kamada_kawai":
    layouts = draw_kamada_kawai("$DATASET_PATH")
else:
    raise ValueError("Unsupported layout type.")

save_layouts(layouts, save_dir="$LAYOUT_OUTPUT")
print(f"Layouts saved to {os.path.abspath('$LAYOUT_OUTPUT')}")

print("Generating adjacency matrices...")
adjacency_matrices = convert_to_sparse_adjacency("$DATASET_PATH")
adjacency_output_path = os.path.join("$ADJACENCY_OUTPUT", "adjacency_matrices.pkl")
save_adjacency_data(adjacency_matrices, dataset_path=adjacency_output_path)
print(f"Adjacency matrices saved to {os.path.abspath(adjacency_output_path)}")

END

echo "Layout generation complete! Layouts saved to $OUTPUT_DIR."