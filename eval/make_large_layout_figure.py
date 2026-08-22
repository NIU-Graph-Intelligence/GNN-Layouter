"""make_large_layout_figure.py - Phi-layouter vs sfdp on a single
large graph (grid at N=10^5), rendered legibly by downsampling.

One rendering showing the Phi-layouter layout beside sfdp on
the same graph at N >= 10^5. The full 10^5-node drawing is a solid block of
ink, so nodes/edges are downsampled for display; the caption text returned
states exactly that. Layout coordinates are each panel's own (the two methods
emit different frames), normalised to the panel's span so the drawing reads.

Usage:
    .venv/bin/python eval/make_large_layout_figure.py \
        --graph data/corpora/grid_100000.pt \
        --phi eval/results/large_layouts/phi_grid_100000.npy \
        --sfdp eval/results/large_layouts/sfdp_grid_100000.npy \
        --out paper/figures/fig_large_layout
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
})


def bfs_window(ei: np.ndarray, n: int, n_draw: int, seed: int):
    """Sample a CONNECTED local window of the graph by BFS from a root, so
    the edges inside the window are mostly preserved and the drawing shows
    layout structure rather than disconnected scatter. Returns (sample,
    edges) where edges is the undirected edge set with both endpoints in
    sample, renumbered to 0..k-1."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import breadth_first_order

    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])),
                      shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    rng = np.random.default_rng(seed)
    # root near the "centre" of the graph: a random node is fine for a
    # connected graph; grid BFS balls are compact squares in either layout.
    root = int(rng.integers(0, n))
    order = breadth_first_order(A, root, directed=False,
                                return_predecessors=False)
    order = order[:n_draw]
    sample = np.sort(np.asarray(order, dtype=np.int64))
    keep = np.isin(ei[0], sample) & np.isin(ei[1], sample)
    edges = ei[:, keep].T
    edges = edges[edges[:, 0] < edges[:, 1]]
    remap = {int(u): i for i, u in enumerate(sample)}
    edges_r = np.array([[remap[int(a)], remap[int(b)]] for a, b in edges],
                       dtype=np.int64)
    return sample, edges_r


def load_ei(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        import torch
        ei = torch.load(path, weights_only=True).numpy()
    elif ext == ".mtx":
        import scipy.sparse as sp
        m = sp.io.mmread(path).tocoo()
        ei = np.stack([m.row, m.col], 0).astype(np.int64)
    else:
        a = np.loadtxt(path)
        ei = np.stack([a[:, 0], a[:, 1]], 0).astype(np.int64)
    if ei.shape[0] != 2:
        raise ValueError(f"edge_index must be [2,E], got {ei.shape}")
    keep = ei[0] != ei[1]
    ei = ei[:, keep]
    uniq = np.unique(np.concatenate([ei[0], ei[1]]))
    remap = {int(u): i for i, u in enumerate(uniq)}
    r = np.array([remap[int(x)] for x in ei[0]], dtype=np.int64)
    c = np.array([remap[int(x)] for x in ei[1]], dtype=np.int64)
    return np.stack([r, c], 0)


def draw(ax, pos, edges, n_keep_edges, title=None):
    """Draw a downsampled panel. `pos` and `edges` are already restricted to
    the sampled node set. Edge order shuffled so the LineCollection draws in a
    stable-ish pattern."""
    rng = np.random.default_rng(0)
    if len(edges) > n_keep_edges:
        edges = edges[rng.choice(len(edges), n_keep_edges, replace=False)]
    seg = np.stack([pos[edges[:, 0]], pos[edges[:, 1]]], axis=1)
    ax.add_collection(LineCollection(seg, colors="#B0B0B0", linewidths=0.35,
                                     alpha=0.8, zorder=1))
    ax.scatter(pos[:, 0], pos[:, 1], s=1.2, c="#4C72B0", zorder=2,
               edgecolors="none")
    span = max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1])) or 1.0
    cx, cy = pos[:, 0].mean(), pos[:, 1].mean()
    ax.set_xlim(cx - span * 0.53, cx + span * 0.53)
    ax.set_ylim(cy - span * 0.53, cy + span * 0.53)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10.5, pad=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--phi", required=True)
    ap.add_argument("--sfdp", required=True)
    ap.add_argument("--out", default="paper/figures/fig_large_layout")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_draw_nodes", type=int, default=3000,
                    help="nodes downsampled to for drawing")
    ap.add_argument("--n_draw_edges", type=int, default=12000,
                    help="edges drawn per panel after sampling (subset)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ei = load_ei(args.graph)
    n = int(ei.max()) + 1

    def load_pos(path):
        p = np.load(path)
        if p.shape[0] != n:
            raise ValueError(f"{path}: {p.shape[0]} rows vs n={n}")
        return p

    pos_phi = load_pos(args.phi)
    pos_sfdp = load_pos(args.sfdp)

    # BFS window: a connected local patch of the graph, so edges survive the
    # downsample and the panel shows layout structure, not scatter.
    sample, edges_r = bfs_window(ei, n, min(args.n_draw_nodes, n), args.seed)
    p_phi = pos_phi[sample]
    p_sfdp = pos_sfdp[sample]
    if edges_r.size == 0:
        raise SystemExit("no edges survive the BFS window; raise --n_draw_nodes")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.7))
    draw(axes[0], p_phi, edges_r, args.n_draw_edges, title="$\\Phi$-layouter")
    draw(axes[1], p_sfdp, edges_r, args.n_draw_edges, title="sfdp")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.005,
                        wspace=0.04)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.out + ".pdf", bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}.png / {args.out}.pdf")
    print("CAPTION: {} nodes (N={}), a BFS-connected local window of the graph "
          "so edges survive the downsample (edges drawn down to {} per panel "
          "for legibility); full layouts were scored without downsampling.".format(
        len(sample), n, args.n_draw_edges))


if __name__ == "__main__":
    main()
