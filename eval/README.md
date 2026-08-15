# Evaluation

Turns model layout predictions into results: **quantitative metrics** (one JSON per model plus a
combined comparison table) and a **qualitative figure** (models side by side vs the ground truth).
Every model — Φ-layouter and all baselines — is scored by the same code, so no comparison is confounded
by two slightly different metric definitions.

```
score_predictions.py        →  quantitative:  eval/results/eval_<name>.json + comparison table/CSV
make_qualitative_figure.py  →  qualitative:   a side-by-side layout figure
metrics.py                  →  every metric definition (shared by both scripts)
```

Both scripts **discover** what's available in `eval/predictions/` and let you choose what to run —
interactively from a menu, or with flags for scripting. Nothing is hardcoded to a fixed set of
models, and both resolve their paths relative to the repo, so they work from any directory.

---

## Prerequisites

**1. Environment** — `torch`, `torch_geometric`, `numpy`, `scipy`, `matplotlib`. The repo's
root `install.sh` provides them:

```bash
bash install.sh && source .venv/bin/activate
```

**2. Dataset** — `data/processed/comm_5k_v2_with_encodings.pt` (the same dataset training used).
Scoring reads FR ground-truth layouts from it and joins predictions by each graph's index.

**3. Prediction files** — one `.npz` per model in `eval/predictions/`. Each baseline writes its own
via the `dump_predictions.py` in that baseline's repo; Φ-layouter's executor writes its own. Typical
set: an executor run, `ablationB`, `deepdrawing`, `gnd_fr`, `gnd_stress`, `smartgd`, `coregd`,
`dnn2`.

<details>
<summary><b>Prediction file format</b> (click)</summary>

`np.savez_compressed` (no pickle) with:

| key | shape / type | meaning |
|---|---|---|
| `source_indices` | int64 `[G]` | each graph's index in the dataset list |
| `n_nodes` | int64 `[G]` | node count per graph (splits `positions`) |
| `positions` | float32 `[Σn, 2]` | final predicted layout, graphs concatenated in order |
| `trajectory` | float32 `[T, Σn, 2]` | *optional* — per-step layout; iterative models (Φ-layouter) only |
| `meta` | JSON string | `{"name":…, "scale_free": bool, …}` |

`scale_free: true` marks a model whose output scale is arbitrary (all baselines); the scorer
fits scale to FR before computing the force metric, uniformly for every model.
</details>

---

## Quantitative — `score_predictions.py`

**Interactive** (lists the prediction files it finds and lets you pick):

```bash
python eval/score_predictions.py
```

**By name** (the `.npz` stem), or **all at once**:

```bash
python eval/score_predictions.py --models vo2 smartgd coregd
python eval/score_predictions.py --all
```

Each selected model gets:
- a metrics table printed to the terminal,
- `eval/results/eval_<name>.json` (full per-graph metrics + summary).

When more than one model is scored, it also prints a **combined comparison table** and writes
`eval/results/comparison.csv` (one row per model, ready to paste into a paper table).

**What the metrics mean**
- **Procrustes MSE** — fidelity to FR's layout, after an O(2)+scale (rotation + reflection +
  uniform scale) alignment. This is the convention DeepDrawing/GND publish, so numbers are directly
  comparable to theirs. Report the **median** (the mean is dragged by a small failure tail).
- **Φ (residual FR force)** — how far the layout is from an FR equilibrium.
- **Quality** — scale-normalized stress, neighborhood preservation, edge crossings, min edge angle,
  community silhouette; FR itself is shown as a reference row.

Options: `--dataset_path`, `--pred_dir`, `--output_dir` all have sensible defaults; override if your
paths differ. `python eval/score_predictions.py -h` lists everything.

---

## Qualitative — `make_qualitative_figure.py`

**Interactive** (pick which models to show as columns):

```bash
python eval/make_qualitative_figure.py
```

**By name**, and optionally choosing the graphs:

```bash
python eval/make_qualitative_figure.py --models deepdrawing gnd_fr gnd_stress dnn2 coregd smartgd
python eval/make_qualitative_figure.py --models smartgd gnd_fr --indices 76 4503 2988
```

Produces one figure (`visualizations/qualitative.png` by default, override with `--out`): rows are
graphs, columns are FR ground truth followed by each chosen model, nodes coloured by community.
Every prediction is Procrustes-aligned to FR so the panels are directly comparable. Omit
`--indices` to auto-pick graphs spread across sizes; column order is fixed (baselines, then Φ-layouter
last) so it reads left-to-right as "… vs ours".

---

## Notes

- Run the scripts from anywhere — paths resolve relative to the repo automatically.
- `eval/results/*.json`, `comparison.csv`, and the figure are **generated outputs**; regenerate
  them with the commands above rather than treating them as source.
- `metrics.py` is the single source of truth, imported by both scripts, so the numbers in the table
  and the figure can never drift apart. Optionally verify it with `eval/validate_metrics.py`.
