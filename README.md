# GNN‑Layouter

 Learn graph layouts (circular and force‑directed / spring) with Graph Neural Networks over synthetic and community graph dataset.




---

## Table of Contents
1. [Motivation](#motivation)
2. [Key Features](#key-features)
3. [Repository Layout](#repository-layout)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Models](#models)
7. [Training](#training)
8. [Visualization](#visualization)

## Motivation
This project explores whether a GNN can infer spatial embeddings directly from graph topology, reducing iterative physics cost and enabling layout generalization across graph families.

## Key Features
* Unified YAML‑driven data pipeline (graphs ➜ layouts ➜ datasets)
* Multiple synthetic graph generators: ER, BA, WS, plus LFR community graphs
* Rich, opt‑in feature engineering (node + graph features)
* Spring layout models supporting optional initial position conditioning
* Deterministic generation via seeded sampling
* Lightweight trainer with early stopping & metric export

Commands:
```
python data/generate_graphs.py  --config <exp>
python data/generate_layouts.py --config <exp>
python data/generate_dataset.py --config <exp>
python train.py --model GCN --layout_type circular --data_path data/processed/<file>.pt
```

## Repository Layout
```
configs/                Experiment YAMLs (graphs, layouts, datasets)
config.yaml             Global paths & defaults
training_config.yaml    Hyperparameters + per‑model settings
data/
  generate_graphs.py    Graph family generation (timestamped bundles)
  generate_layouts.py   Circular or spring (optional initial positions)
  generate_dataset.py   Assemble PyG Data list with selected features
  dataset.py            Load + split utilities
models/                 All architectures + registry
training/               Trainer, losses, evaluation helpers
visualize.py            Ground truth vs prediction comparison
results/                Checkpoints, metrics, plots
```

## Installation
```
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # adjust if needed
pip install -r requirements.txt
pip install pyyaml  # ensure present
```

## Quick Start
```bash
# 1. Generate community graphs
python data/generate_graphs.py --config community_only

# 2. Generate spring layouts
python data/generate_layouts.py --config community_only

# 3. Build dataset
python data/generate_dataset.py --config community_only

# 4. Train a spring (force) layout model
python train.py --model ForceGNN --layout_type force_directed --data_path data/processed/community_5k_force_directed.pt

# 5. Visualize
python visualize.py --model_path results/force_directed/ForceGNN/..._best.pt --data_path data/processed/community_5k_force_directed.pt
```



## Models
Registered (see `models/registry.py`):
* Baseline: `GCN`, `GAT`, `GIN`, `ChebNet`
* Spring‑aware: `SimpleSpringGNN`, `MultiScaleSpringGNN`, `AntiSmoothingSpringGNN`, `ForceGNN`

Add a model: implement class and register in `MODEL_REGISTRY`.

## Training
```bash
python train.py \
  --model GCN \
  --layout_type circular \
  --data_path data/processed/random_graphs_3k_circular.pt

python train.py \
  --model ForceGNN \
  --layout_type force_directed \
  --data_path data/processed/community_5k_force_directed.pt
```

## Visualization
```bash
python visualize.py \
  --model_path results/force_directed/ForceGNN/..._best.pt \
  --data_path data/processed/community_5k_force_directed.pt
```
Outputs side‑by‑side predicted vs ground truth.



