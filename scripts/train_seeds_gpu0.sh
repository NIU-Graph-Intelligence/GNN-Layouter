#!/bin/bash
# GPU 0 chain: phi1_seed7, gmpnn_seed7, phi1_seed2024
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== phi1_seed7 starting $(date) ===" | tee checkpoints/phi1_seed7_progress.log
python -m philayouter.executor.train \
    --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
    --output_dir checkpoints/phi1_seed7 \
    --num_epochs 50 --device cuda:0 --seed 7 --split_seed 42 \
    2>&1 | tee checkpoints/phi1_seed7_progress.log
echo "=== phi1_seed7 done $(date) ===" >> checkpoints/phi1_seed7_progress.log

echo "=== gmpnn_seed7 starting $(date) ===" | tee checkpoints/gmpnn_seed7_progress.log
python -m philayouter.executor.train_gmpnn \
    --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
    --output_dir checkpoints/gmpnn_seed7 \
    --num_epochs 50 --device cuda:0 --seed 7 --split_seed 42 \
    2>&1 | tee checkpoints/gmpnn_seed7_progress.log
echo "=== gmpnn_seed7 done $(date) ===" >> checkpoints/gmpnn_seed7_progress.log

echo "=== phi1_seed2024 starting $(date) ===" | tee checkpoints/phi1_seed2024_progress.log
python -m philayouter.executor.train \
    --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
    --output_dir checkpoints/phi1_seed2024 \
    --num_epochs 50 --device cuda:0 --seed 2024 --split_seed 42 \
    2>&1 | tee checkpoints/phi1_seed2024_progress.log
echo "=== phi1_seed2024 done $(date) ===" >> checkpoints/phi1_seed2024_progress.log


echo "=== GPU0 ALL DONE $(date) ===" >> checkpoints/gpu0_chain_done.log
