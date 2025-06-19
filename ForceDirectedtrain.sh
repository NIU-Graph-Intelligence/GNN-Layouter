#!/bin/bash
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#python3 training/FDlayout_train.py --data data/processed/modelInput_FR1024.pt --layout FR --batch-size 1
python3 training/ForceDirectedTrain.py --data data/processed/modelInput_FR1024.pt --layout FR --batch-size 128
