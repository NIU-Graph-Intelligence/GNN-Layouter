# GNN-Layouter (Φ-layouter)

Learning **force-directed graph layout** as a neural algorithmic-reasoning problem: instead of
predicting a final layout from a graph, the model learns the layout *algorithm itself* — one
learned step of the update `X_{t+1} = Φ(G, X_t, τ_t)` — and reproduces the teacher's trajectory
step by step.

The implementation is `philayouter/executor/`: an **equivariant executor** whose
architecture is derived term-by-term from the Fruchterman-Reingold (FR) update (attraction over
topological edges, repulsion over a per-step geometric kNN graph, decaying temperature
conditioning, equivariant displacement readout). It is rotation/translation-equivariant by
construction, not by training.

This is the code for the paper *Rethinking Graph Data Visualization: Learning the Operator,
Not the Layout*. [`PROTOCOL.md`](PROTOCOL.md) records the parameters behind every reported
number: the graph generators and their seeds, the three teachers' constants, the model and
optimizer configuration, and the rollout classification thresholds.

This README takes you from **nothing to a trained executor and a scored evaluation**.

---

## Installation

Linux + an NVIDIA GPU. The installer builds a virtual environment with pinned dependencies
(defaults to PyTorch with CUDA 12.8):

```bash
bash install.sh
```

Activate it in every new terminal:

```bash
source .venv/bin/activate
```

Once the dataset has been generated (below), this checks it against the SHA-256 sums the
reported runs used, hardcoded in the script rather than fetched from a registry:

```bash
.venv/bin/python scripts/verify_setup.py
```

---

## The pipeline (data → training → evaluation)

Run all commands from the **repository root**, with `.venv` activated. If you already have
`data/processed/comm_5k_v2_with_encodings.pt`, skip straight to
[training](#train-the-executor).

### 1 — Generate graphs

```bash
python data/generate_graphs.py --config community_only
```

Writes a timestamped bundle to `data/graphs/`, e.g.
`comm_5k_5000graphs_20-50nodes_<timestamp>.pkl`. (`--config` names a file in `configs/`; use the
one describing your graph family — `community_only` produces the N=20–50 community graphs used
here.)

### 2 — Run FR and record every iteration

```bash
python data/generate_fr_iterations.py --graphs_file "data/graphs/comm_5k_*20-50nodes_*.pkl" --output data/layouts/comm_5k_fr_iters.pkl --n_iter 50 --verify
```

Runs Fruchterman-Reingold step-by-step and saves the full 50-step trajectory — this is what
lets the executor supervise every step of the process, not just the endpoint. `--verify`
sanity-checks the force computation.

### 3 — Assemble the raw dataset

```bash
python data/generate_final_dataset.py --graphs_file "data/graphs/comm_5k_*20-50nodes_*.pkl" --iters_file data/layouts/comm_5k_fr_iters.pkl --output data/processed/comm_5k_final_v2.pt
```

Combines graphs + trajectories into a single `.pt` list of PyG `Data` objects.

### 4 — Attach structural encodings (one-time)

```bash
python -m philayouter.stage1.preprocess_encodings --input_path data/processed/comm_5k_final_v2.pt --output_path data/processed/comm_5k_v2_with_encodings.pt --lap_k 10 --rw_k 16
```

Precomputes Laplacian / random-walk / community encodings once. **Output:
`data/processed/comm_5k_v2_with_encodings.pt` — the dataset every training script reads.**

### Train the executor

```bash
python -m philayouter.executor.train --dataset_path data/processed/comm_5k_v2_with_encodings.pt --output_dir checkpoints/phi1_v1 --num_epochs 50 --hidden_dim 64 --device cuda:0
```

Fast — the executor is a small equivariant GNN. **Output:
`checkpoints/phi1_v1/executor_best.pt`** plus `executor_log.csv`. A clean run has a monotonically
decreasing val loss; always check `skipped_steps`-style guards in the log before trusting a run.

Relevant flags and their ablation controls (see `philayouter/executor/train.py`):

| flag | default | ablates |
|---|---|---|
| `--use_rrwp 0` | `1` | RRWP in structural encodings (LapPE-only) |
| `--use_tau 0` | `1` | temperature conditioning |
| `--use_geo_mp 0` | `1` | geometric kNN message passing |
| `--use_equiv_readout 0` | `1` | equivariant readout |
| `--supervision endpoint` | `steps` | per-step supervision vs endpoint-only control |

### Evaluate the executor

```bash
python -m philayouter.executor.evaluate --checkpoint checkpoints/phi1_v1/executor_best.pt --dataset_path data/processed/comm_5k_v2_with_encodings.pt --device cuda:0
```

Reports teacher fidelity on held-out test graphs and a step-extrapolation rollout (how far the
learned operator can be rolled out past the trained horizon).

### Score it

`eval/metrics.py` holds every metric definition, so a model and everything it is compared
against are scored by one implementation:

```bash
.venv/bin/python eval/validate_metrics.py --dataset_path data/processed/comm_5k_v2_with_encodings.pt
.venv/bin/python eval/dump_executor_predictions.py --checkpoint checkpoints/phi1_v1/executor_best.pt --name phi1_v1
.venv/bin/python eval/score_predictions.py --all
```

A model writes its layouts to `eval/predictions/<name>.npz` in one model-agnostic format
(documented in [`eval/predictions/README.md`](eval/predictions/README.md)), and
`score_predictions.py` scores whatever it finds there. Predictions are dumped rather than
computed in place because the published comparison methods run in mutually incompatible
environments — their `.npz` files are produced in their own repositories, not here. See
[`eval/README.md`](eval/README.md).

---

## Repository layout

```
config.yaml                 global paths/defaults for data generation
configs/                    per-experiment graph configs (e.g. community_only.yaml)
install.sh                  environment installer  →  .venv/
requirements.txt            pinned dependency versions

data/
  generate_graphs.py        step 1  — graph generation
  community_graph_utils.py  helper for graph generation
  generate_fr_iterations.py step 2  — FR run + trajectory recording
  generate_final_dataset.py step 3  — assemble the raw dataset

philayouter/                 Φ-layouter Python package
  executor/                 equivariant executor (Φ₁) + NAR baselines
    model.py                EquivariantExecutor, the learned FR step
    geometric.py            per-step kNN graph + edge-set merge
    structural.py           topological encodings (reuses stage1 gst modules)
    data.py                 dataset loading, k-unit normalization
    train.py                Φ₁ training entry point
    train_multi.py          multi-teacher conditioning (FR/FA2/KK)
    train_mpnn.py           MPNN-max neural-executor baseline
    train_gmpnn.py          Triplet-GMPNN baseline
    train_forgetnet.py      G-ForgetNet gated-history baseline
    train_compressed.py     compressed executor (Φ_k)
    evaluate.py             fidelity + step extrapolation
    benchmark*.py           timing / wall-clock harness
    check_*.py              numeric verification scripts
  stage1/                   structural encodings reused by the executor: Laplacian PE
                            through a sign-invariant network, RRWP (gst/), and
                            preprocess_encodings.py, which caches them onto a dataset

eval/                       evaluation harness (metrics, scoring, figures)
PROTOCOL.md                 the parameters behind the reported numbers
```

> **Scope.** Code, data-generation scripts, and the evaluation harness. The generated
> datasets, the recorded teacher trajectories, and the trained checkpoints are too large for
> git and are not tracked here — the pipeline above regenerates them, and `PROTOCOL.md` fixes
> the parameters they were produced with.
