"""
score_predictions.py — score model layout predictions against FR ground truth.

Every model (Φ-layouter, its ablations, and every baseline) is scored through this one
script and the shared metrics in `metrics.py`, so no model can get a metric
variant of its own. Each prediction file is an `.npz` in `eval/predictions/`
(see the format table at the bottom of this docstring).

USAGE
-----
Interactive (just run it and pick from a menu):

    python eval/score_predictions.py

Score specific models by name (the .npz stem) or by path:

    python eval/score_predictions.py --models vo2 smartgd
    python eval/score_predictions.py --pred eval/predictions/vo2.npz

Score every prediction file found in eval/predictions/:

    python eval/score_predictions.py --all

Outputs
    * eval/results/eval_<name>.json   — full metrics for each scored model
    * a combined comparison table printed to the terminal (and eval/results/
      comparison.csv) when more than one model is scored.

Fidelity metric (`pmse`): mean squared error after a similarity alignment
(O(2) + uniform scale), the convention DeepDrawing/GND publish, so the numbers
are directly comparable to theirs and no baseline is penalised for a degree of
freedom its objective discarded.

PREDICTION FILE FORMAT (`np.savez_compressed`, no pickle):
    source_indices : int64  [G]        position of each graph in the dataset list
    n_nodes        : int64  [G]        node count per graph (splits `positions`)
    positions      : float32 [sum(n),2] final layout, graphs concatenated in order
    trajectory     : float32 [T,sum(n),2]  OPTIONAL, iterative models only
    meta           : str (JSON)        {"name":..., "scale_free": bool, ...}
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

# Resolve paths relative to the repo root so the script works from any cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from metrics import (adjacency_from_edge_index, community_silhouette,       # noqa: E402
                     denormalize, edge_crossings, fr_force_components,
                     min_edge_angle, neighborhood_preservation,
                     procrustes_align, procrustes_mse, residual_force_gap,
                     rollout_error, scale_normalized_stress,
                     shortest_path_matrix)

DEFAULT_DATASET = os.path.join(ROOT, "data/processed/comm_5k_v2_with_encodings.pt")
DEFAULT_PRED_DIR = os.path.join(HERE, "predictions")
DEFAULT_OUT_DIR = os.path.join(HERE, "results")

# Human-readable labels for known models; unknown names fall back to the stem.
LABELS = {
    "vo2": "Φ-layouter (ours)", "ablationB": "Φ-layouter ablation (endpoint-only)",
    "deepdrawing": "DeepDrawing", "gnd_fr": "GND (FR-supervised)",
    "gnd_stress": "GND (stress)", "smartgd": "SmartGD",
    "coregd": "CoRe-GD", "dnn2": "(DNN)²",
}


# ---------------------------------------------------------------------------
# I/O and discovery
# ---------------------------------------------------------------------------
def discover(pred_dir):
    """Return the sorted list of prediction-file stems in pred_dir."""
    if not os.path.isdir(pred_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(pred_dir) if f.endswith(".npz"))


def resolve_targets(args):
    """
    Turn the CLI selection into a list of (name, path). Precedence:
    explicit --pred / --models  >  --all  >  interactive menu  >  (non-TTY) all.
    """
    stems = discover(args.pred_dir)
    chosen = []

    if args.pred:
        for p in args.pred:
            path = p if os.path.isfile(p) else os.path.join(args.pred_dir, f"{p}.npz")
            chosen.append(path)
    elif args.models:
        chosen = [os.path.join(args.pred_dir, f"{m}.npz") for m in args.models]
    elif args.all:
        chosen = [os.path.join(args.pred_dir, f"{s}.npz") for s in stems]
    elif sys.stdin.isatty():
        chosen = [os.path.join(args.pred_dir, f"{s}.npz") for s in _menu(stems)]
    else:  # non-interactive with no selection: score everything, but say so
        print("No selection given and not a terminal — scoring all discovered "
              f"predictions in {args.pred_dir}.")
        chosen = [os.path.join(args.pred_dir, f"{s}.npz") for s in stems]

    # validate up front with a clear message rather than crashing mid-run
    missing = [p for p in chosen if not os.path.isfile(p)]
    if missing:
        sys.exit("ERROR: prediction file(s) not found:\n  " + "\n  ".join(missing)
                 + f"\n\nAvailable in {args.pred_dir}: "
                 + (", ".join(stems) if stems else "(none)"))
    if not chosen:
        sys.exit(f"ERROR: no prediction files to score. Put <name>.npz files in "
                 f"{args.pred_dir} (see this script's header for the format).")
    return [(os.path.basename(p)[:-4], p) for p in chosen]


def _menu(stems):
    """Interactive picker. Returns a list of selected stems."""
    if not stems:
        sys.exit(f"No prediction files found in {DEFAULT_PRED_DIR}.")
    print("\nPrediction files found:\n")
    for i, s in enumerate(stems, 1):
        print(f"  {i:>2}. {s:<14} {LABELS.get(s, '')}")
    print("\nEnter numbers to score (e.g. '1 3 4'), 'a' for all, 'q' to quit.")
    while True:
        raw = input("> ").strip().lower()
        if raw in ("q", "quit", ""):
            sys.exit("Nothing selected.")
        if raw in ("a", "all"):
            return stems
        try:
            idx = [int(t) for t in raw.replace(",", " ").split()]
            picked = [stems[i - 1] for i in idx if 1 <= i <= len(stems)]
            if picked:
                return picked
        except ValueError:
            pass
        print("Didn't understand that — try e.g. '1 2 5', 'a', or 'q'.")


def load_predictions(path):
    z = np.load(path, allow_pickle=False)
    for key in ("source_indices", "n_nodes", "positions", "meta"):
        if key not in z.files:
            raise ValueError(f"{path}: malformed prediction file, missing '{key}'")
    meta = json.loads(str(z["meta"]))
    n_nodes = z["n_nodes"].astype(int)
    ends = np.cumsum(n_nodes)
    positions = np.split(z["positions"].astype(np.float64), ends[:-1])
    traj = None
    if "trajectory" in z.files:
        t = z["trajectory"].astype(np.float64)
        traj = [t[:, s:e, :] for s, e in zip(np.r_[0, ends[:-1]], ends)]
    return z["source_indices"].astype(int), n_nodes, positions, traj, meta


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def summarise(v):
    a = np.asarray(v, dtype=float)
    ok = np.isfinite(a)
    if not ok.any():
        return {"mean": float("nan"), "median": float("nan"),
                "iqr": float("nan"), "max": float("nan"), "n_valid": 0}
    b = a[ok]
    return {"mean": float(b.mean()), "median": float(np.median(b)),
            "iqr": float(np.percentile(b, 75) - np.percentile(b, 25)),
            "max": float(b.max()), "n_valid": int(ok.sum())}


METRIC_KEYS = ("pmse", "phi_gap", "phi_pred", "phi_true", "att_pred", "rep_pred",
               "mind_pred", "att_true", "rep_true", "mind_true", "N",
               "sns_pred", "sns_true", "alpha_pred", "alpha_true",
               "np_pred", "np_true", "cross_pred", "cross_true",
               "angle_pred", "angle_true", "sil_pred", "sil_true")


def score_one(name, pred_path, data_list):
    """Score one prediction file. Returns the JSON-ready result dict."""
    src_idx, n_nodes, positions, traj, meta = load_predictions(pred_path)
    label = LABELS.get(name, meta.get("name", name))
    scale_free = bool(meta.get("scale_free", False))

    print(f"\nScoring '{name}'  ({label}): {len(src_idx)} graphs"
          f"{'  [scale-free]' if scale_free else ''}"
          f"{'  [has trajectory]' if traj is not None else ''}")

    per_graph = {k: [] for k in METRIC_KEYS}
    roll_aligned, roll_raw = [], []

    for gi, si in enumerate(src_idx):
        if not (0 <= si < len(data_list)):
            raise IndexError(
                f"{name}: source_index {si} is outside the dataset "
                f"(0..{len(data_list) - 1}). The .npz was likely produced against "
                f"a different dataset than {os.path.basename(args_dataset)}.")
        d = data_list[si]
        n = int(d.num_nodes)
        if n != n_nodes[gi]:
            raise ValueError(
                f"{name}: graph {si} has {n} nodes but the prediction has "
                f"{n_nodes[gi]} rows — prediction/dataset mismatch.")

        pred_final = positions[gi]
        target = d.y.float().numpy().astype(np.float64)
        per_graph["pmse"].append(procrustes_mse(pred_final, target))

        if traj is not None:
            true_traj = d.y_traj.float().numpy().reshape(n, -1, 2).transpose(1, 0, 2)
            steps = min(traj[gi].shape[0], true_traj.shape[0])
            roll_aligned.append(rollout_error(traj[gi][:steps], true_traj[:steps], aligned=True))
            roll_raw.append(rollout_error(traj[gi][:steps], true_traj[:steps], aligned=False))

        # Phi is not scale-invariant, so EVERY model is brought to the FR frame by
        # the same similarity fit before measurement (a near-no-op for models
        # already in that frame; large for scale-free ones). Uniform, so fair.
        y_mean, y_std = d.y_mean.numpy(), d.y_std.numpy()
        A = adjacency_from_edge_index(d.edge_index.numpy(), n)
        k = float(getattr(d, "k", np.sqrt(1.0 / n)))
        pred_for_phi, _ = procrustes_align(pred_final, target)

        gap, a, b = residual_force_gap(denormalize(pred_for_phi, y_mean, y_std),
                                       denormalize(target, y_mean, y_std), A, k)
        per_graph["phi_gap"].append(gap); per_graph["phi_pred"].append(a); per_graph["phi_true"].append(b)
        ap_, rp_, mdp = fr_force_components(denormalize(pred_for_phi, y_mean, y_std), A, k)
        at_, rt_, mdt = fr_force_components(denormalize(target, y_mean, y_std), A, k)
        per_graph["att_pred"].append(ap_); per_graph["rep_pred"].append(rp_); per_graph["mind_pred"].append(mdp)
        per_graph["att_true"].append(at_); per_graph["rep_true"].append(rt_); per_graph["mind_true"].append(mdt)
        per_graph["N"].append(n)

        D = shortest_path_matrix(A)
        ei = d.edge_index.numpy()
        und = ei[:, ei[0] < ei[1]].T
        comm = d.community.numpy() if getattr(d, "community", None) is not None else None
        for tag, P in (("pred", pred_final), ("true", target)):
            sns, alpha = scale_normalized_stress(P, D)
            per_graph[f"sns_{tag}"].append(sns)
            per_graph[f"alpha_{tag}"].append(alpha)
            per_graph[f"np_{tag}"].append(neighborhood_preservation(P, A))
            per_graph[f"cross_{tag}"].append(edge_crossings(P, und))
            per_graph[f"angle_{tag}"].append(min_edge_angle(P, A))
            per_graph[f"sil_{tag}"].append(
                community_silhouette(P, comm) if comm is not None else np.nan)

        if (gi + 1) % 100 == 0:
            print(f"  {gi + 1}/{len(src_idx)}")

    stats = {k: summarise(v) for k, v in per_graph.items() if k != "N"}
    _print_one(name, label, per_graph, stats)

    result = {"run_name": name, "label": label, "prediction_file": pred_path,
              "meta": meta, "num_graphs": int(len(src_idx)), "summary": stats,
              "per_graph": {k: [float(x) for x in v] for k, v in per_graph.items()},
              "source_indices": [int(i) for i in src_idx]}
    if roll_aligned:
        ra, rr = np.array(roll_aligned), np.array(roll_raw)
        result |= {"rollout_aligned_mean": ra.mean(axis=0).tolist(),
                   "rollout_raw_mean": rr.mean(axis=0).tolist(),
                   "rollout_aligned_pct": {str(q): np.percentile(ra, q, axis=0).tolist()
                                           for q in (25, 50, 75)}}
        print("  rollout (median)  " + "  ".join(
            f"s{t+1}={np.percentile(ra, 50, axis=0)[t]:.4f}"
            for t in (0, 4, 15, 29, 49) if t < ra.shape[1]))
    return result


def _print_one(name, label, per_graph, stats):
    print("-" * 72)
    print(f"{label}   ({len(per_graph['N'])} graphs, "
          f"N {min(per_graph['N'])}-{max(per_graph['N'])})")
    print(f"{'metric':<28}{'mean':>10}{'median':>10}{'IQR':>10}")
    for key, lab in [("pmse", "Procrustes MSE"),
                     ("phi_pred", "Phi (residual FR force)"),
                     ("phi_true", "  Phi (FR reference)"),
                     ("mind_pred", "min pair distance")]:
        s = stats[key]
        print(f"{lab:<28}{s['mean']:>10.4f}{s['median']:>10.4f}{s['iqr']:>10.4f}")
    print(f"{'quality (median)':<28}{'model':>10}{'FR ref':>10}{'ratio':>10}")
    for pk, tk, lab in [("sns_pred", "sns_true", "stress"),
                        ("np_pred", "np_true", "neighborhood pres."),
                        ("cross_pred", "cross_true", "edge crossings"),
                        ("angle_pred", "angle_true", "min edge angle"),
                        ("sil_pred", "sil_true", "community silhouette")]:
        a, b = stats[pk]["median"], stats[tk]["median"]
        r = a / b if abs(b) > 1e-12 else float("nan")
        print(f"{lab:<28}{a:>10.4f}{b:>10.4f}{r:>9.2f}x")


def print_comparison(results):
    """One row per model — what a reader actually wants to compare."""
    print("\n" + "=" * 96)
    print("COMPARISON  (medians; Procrustes MSE = fidelity to FR, lower better)")
    print("=" * 96)
    hdr = f"{'model':<26}{'ProcMSE':>9}{'Phi':>8}{'stress':>9}{'NP':>8}{'cross':>8}{'angle':>8}{'silhou':>9}"
    print(hdr); print("-" * 96)
    rows = sorted(results, key=lambda r: r["summary"]["pmse"]["median"])
    m = lambda r, k: r["summary"][k]["median"]
    for r in rows:
        print(f"{r['label']:<26}{m(r,'pmse'):>9.4f}{m(r,'phi_pred'):>8.3f}"
              f"{m(r,'sns_pred'):>9.2f}{m(r,'np_pred'):>8.4f}{m(r,'cross_pred'):>8.1f}"
              f"{m(r,'angle_pred'):>8.1f}{m(r,'sil_pred'):>9.4f}")
    # FR reference row from any result (identical across models)
    r0 = results[0]
    print(f"{'FR (reference)':<26}{0.0:>9.4f}{m(r0,'phi_true'):>8.3f}"
          f"{m(r0,'sns_true'):>9.2f}{m(r0,'np_true'):>8.4f}{m(r0,'cross_true'):>8.1f}"
          f"{m(r0,'angle_true'):>8.1f}{m(r0,'sil_true'):>9.4f}")
    print("=" * 96)


def clean_nan(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: clean_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean_nan(x) for x in o]
    return o


args_dataset = ""  # set in main, used only for a clearer error message


def main():
    global args_dataset
    ap = argparse.ArgumentParser(
        description="Score layout predictions against FR ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--models", nargs="+", metavar="NAME",
                     help="prediction stems to score, e.g. --models vo2 smartgd")
    sel.add_argument("--pred", nargs="+", metavar="PATH_OR_NAME",
                     help="explicit .npz path(s) or stem(s) to score")
    sel.add_argument("--all", action="store_true", help="score every .npz found")
    ap.add_argument("--dataset_path", default=DEFAULT_DATASET)
    ap.add_argument("--pred_dir", default=DEFAULT_PRED_DIR)
    ap.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    args_dataset = args.dataset_path

    if not os.path.isfile(args.dataset_path):
        sys.exit(f"ERROR: dataset not found: {args.dataset_path}\n"
                 f"Pass --dataset_path, or place it at the default location.")

    targets = resolve_targets(args)
    print(f"\nLoading dataset {args.dataset_path} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_list = torch.load(args.dataset_path, weights_only=False)
    if isinstance(data_list, dict) and "dataset" in data_list:
        data_list = data_list["dataset"]

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for name, path in targets:
        result = score_one(name, path, data_list)
        dest = os.path.join(args.output_dir, f"eval_{name}.json")
        with open(dest, "w") as f:
            json.dump(clean_nan(result), f, indent=2, allow_nan=False)
        print(f"  saved {dest}")
        results.append(result)

    if len(results) > 1:
        print_comparison(results)
        # also write a flat CSV for easy import into a paper table
        csv_path = os.path.join(args.output_dir, "comparison.csv")
        cols = ["pmse", "phi_pred", "sns_pred", "np_pred", "cross_pred",
                "angle_pred", "sil_pred"]
        with open(csv_path, "w") as f:
            f.write("model," + ",".join(cols) + "\n")
            for r in sorted(results, key=lambda r: r["summary"]["pmse"]["median"]):
                f.write(r["label"].replace(",", "") + "," +
                        ",".join(f"{r['summary'][c]['median']:.4f}" for c in cols) + "\n")
        print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
