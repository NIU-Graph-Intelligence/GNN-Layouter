#!/bin/bash

#python3 training/ForceDirectedTrain.py --data data/processed/modelInput_FRgraphs1024_40Nodes_withPE.pt --layout FR --batch-size 16
python3 training/ForceDirectedTrain.py --data data/processed/custom_modelInput_FR_unnormalized_onehot.pt --layout FR --batch-size 16


#python3 training/FDlayout_train.py --data data/processed/modelInput_FR1024.pt --layout FR --batch-size 1
#/home/tanish/.firstvm/bin/python training/ForceDirectedTrain.py --data data/processed/modelInput_FR1024.pt --layout FR --batch-size 1

