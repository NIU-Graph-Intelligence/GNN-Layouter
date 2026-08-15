#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "=== Creating virtual environment ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo "=== Installing PyTorch ${TORCH_VERSION} ==="
python -m pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"

echo "=== Installing PyTorch Geometric ==="
python -m pip install torch-geometric==2.7.0

echo "=== Installing GLIDE dependencies ==="
python -m pip install -r requirements.txt

echo ""
echo "=== Verifying installation ==="
python -c "
import torch
import torch_geometric
import transformers
import scipy
import networkx
print('torch:          ', torch.__version__)
print('torch_geometric:', torch_geometric.__version__)
print('transformers:   ', transformers.__version__)
print('CUDA available: ', torch.cuda.is_available())
print('GPU count:      ', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}:', torch.cuda.get_device_name(i))
"

echo ""
echo "Installation complete."
echo "Activate with: source $VENV_DIR/bin/activate"
