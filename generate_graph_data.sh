#!/bin/bash

usage() {
  echo "Usage: $0 --output <output_dir> --num_samples <num_samples> [--graph_type <graph_type>] [--num_nodes <num_nodes>] [--output_filename <output_filename>] [--help]"
  echo "  --output: Directory to save the generated dataset."
  echo "  --num_samples: Number of samples to generate for each graph type."
  echo "  --graph_type: Type of graph to generate (ws, er, ba). If not provided, all graph types are generated."
  echo "  --num_nodes: Number of nodes in the graph (default: 50)."
  echo "  --output_filename: Name of the output file (default: merged_graph_dataset.pkl)."
  echo "  --help: Display this help message."
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --graph_type)
      GRAPH_TYPE="$2"
      shift 2
      ;;
    --num_nodes)
      NUM_NODES="$2"
      shift 2
      ;;
    --output_filename)
      OUTPUT_FILENAME="$2"
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

if [ -z "$OUTPUT_DIR" ] || [ -z "$NUM_SAMPLES" ]; then
  echo "Error: Missing arguments."
  usage
fi

# Set default values
NUM_NODES=${NUM_NODES:-50}
OUTPUT_FILENAME=${OUTPUT_FILENAME:-"graph_dataset.pkl"}

export NUM_NODES

if [ -n "$GRAPH_TYPE" ] && [[ ! "$GRAPH_TYPE" =~ ^(ws|er|ba)$ ]]; then
  echo "Error: Invalid graph type. Use 'ws', 'er', or 'ba'."
  usage
fi

mkdir -p "$OUTPUT_DIR"


echo "Generating graph dataset with $NUM_SAMPLES samples per graph type..."
python3 - <<END


import os
from data.generate_graph_data import ER_graph_dataset, WS_graph_dataset, BA_graph_dataset, save_graph_dataset

if '${GRAPH_TYPE}' == 'er' or '${GRAPH_TYPE}' == '':
    print("Generating Erdős-Rényi (ER) graphs...")
    ER_graph_dataset(num_samples=${NUM_SAMPLES}, nodes=${NUM_NODES})

if '${GRAPH_TYPE}' == 'ws' or '${GRAPH_TYPE}' == '':
    print('Generating Watts-Strogatz (WS) graphs...')
    WS_graph_dataset(num_samples=${NUM_SAMPLES}, num_nodes=${NUM_NODES},
                    nearest_neighbors=(2, 10), rewiring_probability=(0.4, 0.6))

if '${GRAPH_TYPE}' == 'ba' or '${GRAPH_TYPE}' == '':
    print('Generating Barabási-Albert (BA) graphs...')
    BA_graph_dataset(num_samples=${NUM_SAMPLES}, num_nodes=${NUM_NODES}, num_edges=(2, 25))


save_graph_dataset(save_dir='${OUTPUT_DIR}', filename='${OUTPUT_FILENAME}')
print(f"Dataset saved to ${OUTPUT_DIR}/${OUTPUT_FILENAME}")

END

echo "Data generation complete! Dataset saved to $OUTPUT_DIR/${OUTPUT_FILENAME}."