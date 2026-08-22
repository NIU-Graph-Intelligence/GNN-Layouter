"""
philayouter/executor/check_equivariance.py

Numeric equivariance check for EquivariantExecutor. Verifies that the
construction-level guarantee the paper claims actually holds in code: for a random
rigid motion (rotation, translation, and reflection) applied to the input
positions, the model's output displacement must transform the same way.

Synthetic random graphs only -- no real dataset needed, no GPU required.

Run: .venv/bin/python -m philayouter.executor.check_equivariance
"""

import math

import torch

from .model import EquivariantExecutor

TOLERANCE = 1e-4


def _random_rigid_motion(reflect: bool, generator: torch.Generator):
    theta = torch.rand(1, generator=generator).item() * 2 * math.pi
    c, s = math.cos(theta), math.sin(theta)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    if reflect:
        R = R @ torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    t = torch.randn(2, generator=generator)
    return R, t


def _random_graph(n: int, node_feat_dim: int, edge_prob: float, generator: torch.Generator):
    adj = torch.rand(n, n, generator=generator) < edge_prob
    adj.fill_diagonal_(False)
    row, col = adj.nonzero(as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)

    node_feat = torch.randn(n, node_feat_dim, generator=generator)
    pos = torch.randn(n, 2, generator=generator)
    tau_val = torch.rand(1, generator=generator).item() + 0.1  # keep away from 0
    return node_feat, edge_index, pos, tau_val


def check(seed: int, n: int = 12, node_feat_dim: int = 4, reflect: bool = False) -> float:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model = EquivariantExecutor(node_feat_dim=node_feat_dim, hidden_dim=16, k_geo=4)
    model.eval()

    node_feat, edge_index, pos, tau_val = _random_graph(n, node_feat_dim, 0.3, gen)
    tau = torch.full((n, 1), tau_val)

    with torch.no_grad():
        dX = model(node_feat, edge_index, pos, tau)

        R, t = _random_rigid_motion(reflect, gen)
        pos_moved = pos @ R.T + t
        dX_moved = model(node_feat, edge_index, pos_moved, tau)

    predicted = dX @ R.T
    return (dX_moved - predicted).abs().max().item()


def main():
    max_err = 0.0
    all_pass = True
    for seed in range(10):
        for reflect in (False, True):
            err = check(seed=seed, reflect=reflect)
            max_err = max(max_err, err)
            ok = err < TOLERANCE
            all_pass &= ok
            tag = "reflect" if reflect else "rotate "
            print(f"seed={seed:2d} {tag} max|dX(Rx+t) - dX(x)@R^T| = {err:.2e}  {'PASS' if ok else 'FAIL'}")

    print(f"\noverall max error: {max_err:.2e}  {'PASS' if all_pass else 'FAIL'} (tolerance {TOLERANCE:.0e})")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
