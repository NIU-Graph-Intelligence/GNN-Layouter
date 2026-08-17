#!/bin/bash
# GPU 1 chain: mpnn_seed7, forgetnet_seed7, mpnn_seed2024, forgetnet_seed2024
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== mpnn_seed7 starting $(date) ===" | tee checkpoints/mpnn_seed7_progress.log
python -m philayouter.executor.train_mpnn \
    --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
    --output_dir checkpoints/mpnn_seed7 \
    --num_epochs 50 --device cuda:1 --seed 7 --split_seed 42 \
    2>&1 | tee checkpoints/mpnn_seed7_progress.log
echo "=== mpnn_seed7 done $(date) ===" >> checkpoints/mpnn_seed7_progress.log

echo "=== forgetnet_seed7 starting $(date) ===" | tee checkpoints/forgetnet_seed7_progress.log
python -m philayouter.executor.train_forgetnet \
    --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
    --output_dir checkpoints/forgetnet_seed7 \
    --num_epochs 50 --device cuda:1 --seed 7 --split_seed 42 \
    2>&1 | tee checkpoints/forgetnet_seed7_progress.log
echo "=== forgetnet_seed7 done $(date) ===" >> checkpoints/forgetnet_seed7_progress.log

echo "=== mpnn_seed2024 starting $(date) ===" | tee checkpoints/mpnn_seed2024_progress.log
python -m philayouter.executor.train_mpnn \
    --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
    --output_dir checkpoints/mpnn_seed2024 \
    --num_epochs 50 --device cuda:1 --seed 2024 --split_seed 42 \
    2>&1 | tee checkpoints/mpnn_seed2024_progress.log
echo "=== mpnn_seed2024 done $(date) ===" >> checkpoints/mpnn_seed2024_progress.log

echo "=== forgetnet_seed2024 starting $(date) ===" | tee checkpoints/forgetnet_seed2024_progress.log
python -m philayouter.executor.train_forgetnet \
    --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
    --output_dir checkpoints/forgetnet_seed2024 \
    --num_epochs 50 --device cuda:1 --seed 2024 --split_seed 42 \
    2>&1 | tee checkpoints/forgetnet_seed2024_progress.log
echo "=== forgetnet_seed2024 done $(date) ===" >> checkpoints/forgetnet_seed2024_progress.log

echo "=== GPU1 ALL DONE $(date) ===" >> checkpoints/gpu1_chain_done.log
