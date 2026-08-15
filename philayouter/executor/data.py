"""
philayouter/executor/data.py

Loads Q4 training pairs from the canonical dataset
(comm_5k_v2_with_encodings.pt), reconstructing the raw FR frame and applying
the k-unit normalization from the scale-normalization convention.

Important: the canonical dataset's own x/y/y_traj are normalized by
(y_mean, y_std) -- the mean and std of THAT GRAPH'S FINAL layout (see
data/generate_final_dataset.py). That's a per-graph affine z-score derived
from the answer, not the k = sqrt(1/N) non-dimensionalization
the scale-normalization convention calls for -- and a step operator conditioned on
(G, X_t, tau_t) must not depend on statistics of the trajectory's own
endpoint. This module undoes that normalization (via eval.metrics.denormalize,
the same function eval/validate_metrics.py already relies on) and reapplies
k-unit normalization instead of training in the dataset's native frame.

Temperature is not stored per step in this dataset. It's reconstructed
exactly, not approximated: t0 = max(coord_span(p0_raw)) * 0.1 and
dt = t0 / (n_iter + 1), matching
data/processed/generate_fr_iterations.py's linear-cooling schedule, because
p0_raw recovered here IS that generator's grid initialization
(data/generate_final_dataset.py stores steps[0]["pos_before"] verbatim as
x's position features). Verified against the recorded trajectories: median
per-node step magnitude matches the reconstructed temperature to 2.9e-6
relative error over 200 graphs x 50 steps (FR's step length is temperature
exactly, by construction of fr_single_step) -- see
check_data_pipeline.py.
"""

from typing import Iterator, List, NamedTuple

import numpy as np
import torch
from torch_geometric.data import Data

from eval.metrics import denormalize


class StepSample(NamedTuple):
    edge_index: torch.Tensor  # [2, E]
    pos_before: torch.Tensor  # [N, 2] k-normalized
    pos_after: torch.Tensor  # [N, 2] k-normalized -- training target
    tau: torch.Tensor  # [N, 1] k-normalized temperature, broadcast per node
    k: float  # sqrt(1/N) for this graph
    teacher_id: torch.Tensor  # [N] long, which teacher this trajectory came from (Q7)
    stride: int  # number of algorithm steps compressed into one forward pass (Q10)


def temperature_schedule(pos0_raw: np.ndarray, n_iter: int) -> np.ndarray:
    """Exact linear-cooling schedule used at generation time. temps[t] is the
    temperature used to produce step t (pos_before -> pos_after)."""
    coord_span = pos0_raw.max(axis=0) - pos0_raw.min(axis=0)
    t0 = max(float(coord_span.max()) * 0.1, 1e-4)
    dt = t0 / (n_iter + 1)
    temps = np.maximum(t0 - np.arange(n_iter) * dt, 1e-6)
    return temps.astype(np.float32)


def graph_to_raw_trajectory(data: Data):
    """Recover the raw FR frame for one graph.

    Returns:
        pos0_raw: [N, 2]
        traj_raw: [T, N, 2] -- traj_raw[t] is the state after step t (0-indexed)
        k:        sqrt(1/N) for this graph
    """
    n = int(data.num_nodes)
    y_mean, y_std = data.y_mean.numpy(), data.y_std.numpy()
    pos0_raw = denormalize(data.x[:, 2:4].numpy(), y_mean, y_std)
    traj_raw = denormalize(
        data.y_traj.numpy().reshape(n, -1, 2).transpose(1, 0, 2), y_mean, y_std
    )
    return pos0_raw, traj_raw, float(data.k)


def iter_steps(data: Data, teacher_id: int = 0, stride: int = 1) -> Iterator[StepSample]:
    """Yield one StepSample per FR iteration recorded for this graph, in the
    k-normalized frame. `teacher_id` (Q7): which teacher this trajectory is
    from -- 0=FR by default, so single-teacher callers (Q4/Q5) don't need to
    change.

    `stride` (Q10): the number of algorithm steps compressed into one forward
    pass -- Phi_k's stride on the already-recorded trajectory, with NO new data
    collection (PAPER_PLAN.md §5). For stride=s the target is pos_{t+s} - pos_t
    and tau is the *accumulated* temperature budget over the s-step window
    (sum of temps[t:t+s]), so a trained Phi_k model's readout magnitude is
    conditioned on the total displacement budget it must spend. stride=1
    reproduces Phi_1 exactly. The final s-1 steps of each trajectory are
    dropped (no recorded pos_{t+s} to target)."""
    pos0_raw, traj_raw, k = graph_to_raw_trajectory(data)
    n_iter = traj_raw.shape[0]
    temps = temperature_schedule(pos0_raw, n_iter)

    full_raw = np.concatenate([pos0_raw[None], traj_raw], axis=0)  # [T+1, N, 2]
    n = full_raw.shape[1]
    teacher_id_t = torch.full((n,), teacher_id, dtype=torch.long)

    for t in range(n_iter - (stride - 1)):
        pos_before = torch.from_numpy((full_raw[t] / k).astype(np.float32))
        pos_after = torch.from_numpy((full_raw[t + stride] / k).astype(np.float32))
        tau_val = float(temps[t : t + stride].sum() / k)
        tau = torch.full((n, 1), tau_val, dtype=torch.float32)
        yield StepSample(
            edge_index=data.edge_index,
            pos_before=pos_before,
            pos_after=pos_after,
            tau=tau,
            k=k,
            teacher_id=teacher_id_t,
            stride=stride,
        )


def load_split(dataset_path: str, seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> dict:
    """Same split convention as the rest of the repo: seeded shuffle,
    80/10/10 -- reused rather than re-derived so Q4 trains on the identical
    train/val/test partition every other reported number in this repo uses."""
    import random

    dataset: List[Data] = torch.load(dataset_path, weights_only=False)
    indices = list(range(len(dataset)))
    # Mirrors the repo's canonical create_data_loaders split exactly (global
    # random.seed + random.shuffle, not a dedicated Random instance) so the
    # split is bit-identical to what every other reported number in this repo
    # trains/evaluates on.
    random.seed(seed)
    random.shuffle(indices)

    n = len(dataset)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    return {
        "train": [dataset[i] for i in indices[:n_train]],
        "val": [dataset[i] for i in indices[n_train : n_train + n_val]],
        "test": [dataset[i] for i in indices[n_train + n_val :]],
    }
