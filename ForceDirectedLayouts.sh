#!/bin/bash

export PYTHONPATH=".:$PYTHONPATH"     # ensure project root is on PYTHONPATH
python3 data/ForceDirectedLayouts.py --generate-adj --generate-fr --generate-fa2 --visualize