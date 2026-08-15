"""
philayouter/executor/evaluate.py

Q5 (the experiment queue): fidelity and step-extrapolation for a trained
Phi_1 checkpoint against the FR trajectories in comm_5k_v2_with_encodings.pt.

Fidelity: roll the model out for T=50 steps (matching the recorded window)
and score against the recorded FR trajectory with eval.metrics.rollout_error
-- the same function/convention eval/evaluate.py already uses for other
models, so this number is comparable to them, not a bespoke metric.

Step extrapolation: continue the rollout to 10T using the same temperature
schedule extended past its recorded window (it clamps to the floor,
1e-6, well before t=49 -- see philayouter/executor/data.py's
temperature_schedule), and check whether node positions stay bounded
(converges), oscillate, or blow up (diverges). No ground truth exists past
step 50 -- this is a qualitative check, not a fidelity number.

Usage:
    .venv/bin/python -m philayouter.executor.evaluate \
        --checkpoint checkpoints/phi1_v1/executor_best.pt \
        --dataset_path data/processed/comm_5k_v2_with_encodings.pt
"""

import argparse

import numpy as np
import torch

from eval.metrics import rollout_error
from .data import graph_to_raw_trajectory, load_split, temperature_schedule
from .model import EquivariantExecutor
from .structural import StructuralEncoder


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]

    encoder = StructuralEncoder(
        out_dim=args["hidden_dim"],
        lap_k=args["lap_k"],
        rw_k=args["rw_k"],
        use_rrwp=args.get("use_rrwp", True),  # old checkpoints predate the flag
    ).to(device)
    # num_teachers: Q7 checkpoints store a `teachers` dict (see train_multi.py);
    # single-teacher checkpoints predate both and default to 1.
    num_teachers = ckpt.get("teachers", None)
    num_teachers = len(num_teachers) if isinstance(num_teachers, dict) else 1
    # use_stride: Q11 (Phi_k) checkpoints store a `strides` list; Phi_1
    # checkpoints predate the flag and default to False.
    use_stride = isinstance(ckpt.get("strides", None), list)
    # use_tau: Q15 no-temperature ablation checkpoints store the flag; other
    # checkpoints predate it and default to True.
    use_tau = args.get("use_tau", True)
    # use_geo_mp: Q16 no-geometric-rewiring ablation checkpoints store it;
    # others default to True.
    use_geo_mp = args.get("use_geo_mp", True)
    # use_equiv_readout: Q17 non-equivariant-readout ablation checkpoints
    # store it; others default to True.
    use_equiv_readout = args.get("use_equiv_readout", True)
    if ckpt.get("model_type") == "MpnnExecutor":
        # Q20 baseline: plain MPNN, no equivariance machinery.
        from .mpnn import MpnnExecutor

        model = MpnnExecutor(
            node_feat_dim=args["hidden_dim"],
            hidden_dim=args["hidden_dim"],
            num_layers=args.get("num_layers", 3),
        ).to(device)
    elif ckpt.get("model_type") == "TripletGMPNN":
        # Q25 baseline: strongest standard NAR processor (triplet GMPNN).
        from .gmpnn import TripletGMPNN

        model = TripletGMPNN(
            node_feat_dim=args["hidden_dim"],
            hidden_dim=args["hidden_dim"],
            num_layers=args.get("num_layers", 3),
        ).to(device)
    elif ckpt.get("model_type") == "GForgetNet":
        # Q26 baseline: gated-history executor (Markov property control).
        from .forgetnet import GForgetNet

        model = GForgetNet(
            node_feat_dim=args["hidden_dim"],
            hidden_dim=args["hidden_dim"],
            num_layers=args.get("num_layers", 3),
        ).to(device)
    else:
        model = EquivariantExecutor(
            node_feat_dim=args["hidden_dim"],
            hidden_dim=args["hidden_dim"],
            k_geo=args["k_geo"],
            num_teachers=num_teachers,
            use_stride=use_stride,
            use_tau=use_tau,
            use_geo_mp=use_geo_mp,
            use_equiv_readout=use_equiv_readout,
        ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    model.eval()
    return encoder, model, ckpt


@torch.no_grad()
def rollout_model(encoder, model, data, device, n_steps: int, temps_raw: np.ndarray, teacher_id: int = 0, stride: int = 1):
    """Roll the model out for n_steps, in k-units internally, returning raw-
    frame positions [n_steps, N, 2] (step 1..n_steps, matching y_traj's
    convention). `teacher_id` (Q7): which teacher's conditioning vector to
    feed -- used for the conditioning-swap matrix (evaluate a trained-in
    teacher against every teacher's ground truth).

    `stride` (Q11, Phi_k): how many algorithm steps one forward pass covers.
    With stride=s the model advances the layout by s recorded iterations per
    forward, so reaching FR step t takes ceil(t/s) forwards and `temps_raw`
    entries are consumed s at a time -- the temperature fed at step j is the
    accumulated budget sum(temps[j*s:(j+1)*s]), matching train_compressed's
    conditioning. stride=1 reproduces Phi_1 exactly."""
    edge_index = data.edge_index.to(device)
    node_feat = encoder.encode_precomputed(data.lap_pe.to(device), data.rrwp_node.to(device))

    pos0_raw, _, k = graph_to_raw_trajectory(data)
    pos = torch.from_numpy((pos0_raw / k).astype(np.float32)).to(device)  # k-normalized

    out = np.empty((n_steps, pos.shape[0], 2), dtype=np.float32)
    stride_t = torch.tensor(float(stride), device=device)
    is_gforget = getattr(model, "HAS_HISTORY", False)
    h_state = None
    for t in range(n_steps):
        s0, s1 = t * stride, min((t + 1) * stride, len(temps_raw))
        tau_val = float(temps_raw[s0:s1].sum() / k)
        tau = torch.full((pos.shape[0], 1), tau_val, device=device)
        if is_gforget:
            dX, h_state = model(node_feat, edge_index, pos, tau, h_state=h_state)
        else:
            dX = model(node_feat, edge_index, pos, tau,
                       teacher_id=torch.full((pos.shape[0],), teacher_id, dtype=torch.long, device=device),
                       stride=stride_t)
        pos = pos + dX
        out[t] = (pos * k).cpu().numpy()  # back to raw frame
    return out


def fidelity(encoder, model, test_graphs, device, n_report=(1, 10, 25, 50), teacher_id: int = 0, label: str = "FR"):
    """Procrustes rollout_error against the recorded T=50 window, averaged
    over the test split."""
    T = 50
    per_step_sum = np.zeros(T)
    per_step_sum_unaligned = np.zeros(T)
    n = 0

    for data in test_graphs:
        pos0_raw, traj_raw, k = graph_to_raw_trajectory(data)
        temps = temperature_schedule(pos0_raw, T)
        pred = rollout_model(encoder, model, data, device, T, temps, teacher_id=teacher_id)

        per_step_sum += rollout_error(pred, traj_raw, aligned=True)
        per_step_sum_unaligned += rollout_error(pred, traj_raw, aligned=False)
        n += 1

    mean_aligned = per_step_sum / n
    mean_unaligned = per_step_sum_unaligned / n

    print(f"\nfidelity vs {label}, {n} test graphs, Procrustes rollout_error (T={T}):")
    for step in n_report:
        print(
            f"  step {step:3d}: aligned {mean_aligned[step-1]:.6f}   "
            f"unaligned {mean_unaligned[step-1]:.6f}"
        )
    return mean_aligned, mean_unaligned


def step_extrapolation(encoder, model, test_graphs, device, factor: int = 10, n_graphs: int = 30):
    """Roll out to factor*T with no ground truth; classify each graph's
    trajectory as converged / oscillating / diverged by whether the mean
    node displacement per step decays, stays bounded and non-monotonic, or
    grows without bound over the second half of the extended rollout."""
    T = 50
    T_ext = T * factor
    outcomes = {"converged": 0, "oscillating": 0, "diverged": 0}

    for data in test_graphs[:n_graphs]:
        pos0_raw, _, k = graph_to_raw_trajectory(data)
        temps = temperature_schedule(pos0_raw, T)
        # Extend the schedule past its recorded window -- linear cooling
        # clamped at the floor (1e-6), matching temperature_schedule's own
        # formula rather than inventing a new one for the extension.
        t0 = float(temps[0])  # temps[0] = t0 - 0*dt = t0 exactly, no clamping at index 0
        dt = t0 / (T + 1)
        temps_ext = np.maximum(t0 - np.arange(T_ext) * dt, 1e-6).astype(np.float32)

        pred = rollout_model(encoder, model, data, device, T_ext, temps_ext)
        step_mag = np.linalg.norm(np.diff(pred, axis=0), axis=-1).mean(axis=-1)  # [T_ext-1]

        first_half = step_mag[: T_ext // 2].mean()
        second_half = step_mag[T_ext // 2 :].mean()
        max_extent = np.abs(pred).max()

        if not np.isfinite(max_extent) or max_extent > 1e4:
            outcomes["diverged"] += 1
        elif second_half < first_half * 0.5 or second_half < 1e-3:
            outcomes["converged"] += 1
        else:
            outcomes["oscillating"] += 1

    print(f"\nstep extrapolation, T={T} -> {T_ext} ({factor}x), {min(n_graphs, len(test_graphs))} graphs:")
    for k_, v in outcomes.items():
        print(f"  {k_}: {v}")
    return outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_path", default="data/processed/comm_5k_v2_with_encodings.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher_id", type=int, default=0,
                        help="which teacher's conditioning vector to feed during "
                             "the rollout (Q7 conditioning-swap matrix). For a "
                             "multi-teacher checkpoint, eval with each teacher_id "
                             "against each dataset's ground truth.")
    parser.add_argument("--dataset_label", default="FR",
                        help="label for this dataset's ground-truth teacher "
                             "(FR/FA2/KK), printed in the fidelity header.")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    encoder, model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"loaded checkpoint from epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.6f}")

    test_graphs = load_split(args.dataset_path, seed=args.seed)["test"]
    print(f"test graphs: {len(test_graphs)}")

    fidelity(encoder, model, test_graphs, device,
             teacher_id=args.teacher_id, label=args.dataset_label)
    step_extrapolation(encoder, model, test_graphs, device)


if __name__ == "__main__":
    main()
