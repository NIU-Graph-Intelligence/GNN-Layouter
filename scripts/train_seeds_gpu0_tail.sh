#!/bin/bash
# GPU0 tail: gmpnn_seed7 still running orphaned; wait for it via executor_log.csv,
# then run phi1_seed2024 and gmpnn_seed2024.
set -e
cd /home/lei/work/GNN-Layouter
source .venv/bin/activate

# Wait for gmpnn_seed7 to finish: executor_log.csv has header + 50 rows = 51 lines
echo "=== tail chain: waiting for gmpnn_seed7 executor_log (epoch 50) $(date) ===" | tee checkpoints/gpu0_tail_chain.log
until [ "$(wc -l < checkpoints/gmpnn_seed7/executor_log.csv 2>/dev/null)" -ge 51 ]; do
    sleep 60
done
sleep 120  # wait for final checkpoint save + cleanup
echo "=== gmpnn_seed7 confirmed done $(date) ===" | tee -a checkpoints/gpu0_tail_chain.log
# write done marker for check_and_eval.py
echo "=== gmpnn_seed7 done $(date) ===" >> checkpoints/gmpnn_seed7_progress.log

echo "=== phi1_seed2024 starting $(date) ===" | tee checkpoints/phi1_seed2024_progress.log | tee -a checkpoints/gpu0_tail_chain.log
python -m philayouter.executor.train \
    --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
    --output_dir checkpoints/phi1_seed2024 \
    --num_epochs 50 --device cuda:0 --seed 2024 --split_seed 42 \
    2>&1 | tee -a checkpoints/phi1_seed2024_progress.log | tee -a checkpoints/gpu0_tail_chain.log
echo "=== phi1_seed2024 done $(date) ===" | tee -a checkpoints/phi1_seed2024_progress.log | tee -a checkpoints/gpu0_tail_chain.log

echo "=== gmpnn_seed2024 starting $(date) ===" | tee checkpoints/gmpnn_seed2024_progress.log | tee -a checkpoints/gpu0_tail_chain.log
python -m philayouter.executor.train_gmpnn \
    --dataset_path data/processed/comm_5k_v2_with_encodings.pt \
    --output_dir checkpoints/gmpnn_seed2024 \
    --num_epochs 50 --device cuda:0 --seed 2024 --split_seed 42 \
    2>&1 | tee -a checkpoints/gmpnn_seed2024_progress.log | tee -a checkpoints/gpu0_tail_chain.log
echo "=== gmpnn_seed2024 done $(date) ===" | tee -a checkpoints/gmpnn_seed2024_progress.log | tee -a checkpoints/gpu0_tail_chain.log

echo "=== GPU0 tail ALL DONE $(date) ===" | tee -a checkpoints/gpu0_tail_chain.log
