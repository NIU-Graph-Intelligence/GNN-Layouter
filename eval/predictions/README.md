# eval/predictions — the model-agnostic prediction format

The models live in mutually incompatible environments (GLIDE needs torch+PyG;
DeepDrawing and SmartGD need torch+PyG; GND needs torch 2.2.1+DGL and cannot
coexist with the others). No single process can import them all, so evaluation
is split:

1. each model dumps predictions **in its own environment** to `<name>.npz`
2. `eval/score_predictions.py` reads those files and scores every model through
   the same `eval/metrics.py`

Predictions are archived, so re-scoring after a metric change takes seconds
rather than re-running six models.

## Format

`np.savez_compressed` with:

| key | dtype / shape | meaning |
|---|---|---|
| `source_indices` | int64 `[G]` | position in the original `comm_5k_v2_with_encodings.pt` list |
| `n_nodes` | int64 `[G]` | node count per graph, used to split `positions` |
| `positions` | float32 `[sum(n), 2]` | final predicted layout, graphs concatenated in order |
| `trajectory` | float32 `[T, sum(n), 2]` | *optional* — per-step rollout, only for iterative models |
| `meta` | str (JSON) | model name, checkpoint path, frame notes |

`positions` is concatenated rather than an object array so the file stays a
plain `.npz` with no pickle. Split with `np.split(positions, np.cumsum(n_nodes)[:-1])`.

## Frame conventions — read before adding a model

Each model outputs in its own frame, and the scorer must be told which:

| model | frame | scoring implication |
|---|---|---|
| GLIDE, ablation B | normalized FR frame (shares `y_mean`/`y_std`) | scale is anchored; SO(2) is meaningful |
| DeepDrawing, GND-FR | arbitrary scale | trained with scale-invariant Procrustes — **must** be scored scale-fitted, and Φ needs a scale fit first |
| SmartGD, GND-stress | stress-optimal scale | never saw FR; excluded from the fidelity table |

`meta["scale_free"] = true` marks a model whose scale carries no information.
The scorer uses it to decide whether Φ gets a scale fit, and it is why every
fidelity number is reported in **both** SO(2) and similarity form.
