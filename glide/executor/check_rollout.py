"""
glide/executor/check_rollout.py

End-to-end check for Q3 (the experiment queue): StructuralEncoder is
called once per graph, its output reused unchanged across a multi-step
rollout, and the geometric neighbourhood + readout are recomputed each step.
Confirms equivariance survives composition across T steps (not just one),
since that's the actual usage pattern training/inference will follow, and a
bug that only shows up after several steps (e.g. drift from float precision,
or an accidental dependence on absolute step count) wouldn't be caught by
check_equivariance.py's single-step check.

Synthetic random graphs only -- no real dataset, no GPU required.

Run: .venv/bin/python -m glide.executor.check_rollout
"""

import math

import torch

from .model import EquivariantExecutor
from .structural import StructuralEncoder

TOLERANCE = 1e-3  # looser than the single-step check: T steps compound float error
T_STEPS = 10


def _random_rigid_motion(reflect: bool, generator: torch.Generator):
    theta = torch.rand(1, generator=generator).item() * 2 * math.pi
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    if reflect:
        R = R @ torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    t = torch.randn(2, generator=generator)
    return R, t


def _random_graph(n: int, edge_prob: float, generator: torch.Generator):
    adj = torch.rand(n, n, generator=generator) < edge_prob
    adj.fill_diagonal_(False)
    # symmetrize so every graph has a well-defined Laplacian (Q3 needs a
    # connected-ish undirected topology; the training data itself is
    # always undirected community graphs)
    adj = adj | adj.T
    row, col = adj.nonzero(as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)
    pos0 = torch.randn(n, 2, generator=generator)
    return edge_index, pos0


def rollout(encoder, model, edge_index, num_nodes, pos0, tau_schedule):
    node_feat = encoder(edge_index, num_nodes)  # computed ONCE, reused every step
    pos = pos0
    with torch.no_grad():
        for tau_val in tau_schedule:
            tau = torch.full((num_nodes, 1), tau_val)
            dX = model(node_feat, edge_index, pos, tau)
            pos = pos + dX
    return pos


def check(seed: int, n: int = 14, reflect: bool = False) -> float:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    encoder = StructuralEncoder(out_dim=16, lap_k=4, rw_k=4)
    encoder.eval()
    model = EquivariantExecutor(node_feat_dim=16, hidden_dim=16, k_geo=4)
    model.eval()

    edge_index, pos0 = _random_graph(n, edge_prob=0.3, generator=gen)
    tau_schedule = [0.3 * (1 - i / T_STEPS) + 0.02 for i in range(T_STEPS)]

    pos_final = rollout(encoder, model, edge_index, n, pos0, tau_schedule)

    R, t = _random_rigid_motion(reflect, gen)
    pos0_moved = pos0 @ R.T + t
    pos_final_moved = rollout(encoder, model, edge_index, n, pos0_moved, tau_schedule)

    predicted = pos_final @ R.T + t
    return (pos_final_moved - predicted).abs().max().item()


def main():
    max_err = 0.0
    all_pass = True
    for seed in range(6):
        for reflect in (False, True):
            err = check(seed=seed, reflect=reflect)
            max_err = max(max_err, err)
            ok = err < TOLERANCE
            all_pass &= ok
            tag = "reflect" if reflect else "rotate "
            print(
                f"seed={seed:2d} {tag} T={T_STEPS} "
                f"max|pos_T(Rx+t) - (pos_T(x)@R^T + t)| = {err:.2e}  {'PASS' if ok else 'FAIL'}"
            )

    print(f"\noverall max error: {max_err:.2e}  {'PASS' if all_pass else 'FAIL'} (tolerance {TOLERANCE:.0e})")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
