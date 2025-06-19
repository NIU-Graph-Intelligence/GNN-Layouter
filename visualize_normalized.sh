#!/bin/bash

# Activate virtual environment if needed
# source venv/bin/activate

# Change to project root directory
cd "$(dirname "$0")"

# Run visualization for normalized model
python utils/visualization.py results/metrics/GIN/best_model_GIN_withPositionalFeature_normalized_batch128.pt


#python utils/kamadavisualization.py results/metrics/best_model_kamada_batch256.pt
python kamadavisualization.py results/metrics/KamadaKawai/best_model_KamadaKawai_withPositionalFeature_normalized_batch256.pt data/processed/kamada_normalized.pt --train --samples 9

#PaperApproach
#python utils/PaperApproachKamadaVisualization.py results/metrics/best_model_kamada_batch256.pt data/processed/AddingFeatureExperimentkamada_normalized.pt --num_examples 6

echo "Visualization completed. Check results/figures for the output."


