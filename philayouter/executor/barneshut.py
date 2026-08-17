"""
philayouter/executor/barneshut.py

Barnes-Hut approximation of the Fruchterman-Reingold step (Q22, PAPER_PLAN
§7). FR's O(N^2) cost is the all-pairs repulsion term; Barnes-Hut groups
distant nodes into quadtree cells and replaces each cell by its centre of
mass when the cell subtends a small angle (theta), giving O(N log N).

Exactly the term it approximates (identical to `fr_single_step` in
data/processed/generate_fr_iterations.py):

    disp_rep[i] = sum_j  k^2 * (p_i - p_j) / |p_i - p_j|^2

i.e. magnitude k^2/d along the delta direction (the 1/r force). A cell with
`mass` points at centroid c contributes to point i:

    k^2 * mass * (p_i - c) / |p_i - c|^2      when  cell_width / d < theta

Attraction over edges is O(E) and computed exactly, unchanged.

Implementation: a flat-array quadtree (children/bbox/centroid/mass in numpy
arrays) with an iterative DFS, JIT-compiled with numba -- the O(N log N)
kernel, not the O(N^2) all-pairs sum. The tree is built once per step and
one force query runs per node. Theta=0.7 is the standard accuracy/cost
trade-off (same order of error as sfdp's default).

This is a *baseline* for the wall-clock table, so the emphasis is a
reproducible timing of a correct classical implementation. Numba is the only
reason a Python Barnes-Hut is usable at N=10^5-10^6; the pure-Python
`_accumulate_ref` path exists for validation only (check_barneshut.py).

Usage (validated in check_barneshut.py):
    .venv/bin/python -m philayouter.executor.check_barneshut
"""

import numpy as np

BUCKET = 16  # leaf capacity before splitting
_EPS = 1e-12


def _build_tree(x, y):
    """Return the flat-array quadtree over points (x, y).

    Arrays (all length n_nodes):
        x0, y0, w, h     -- cell bounding box
        cx, cy, mass     -- centroid and point count of the subtree
        child0..child3   -- child indices, -1 for a leaf
        leaf_start, leaf_count -- range into `leaf_points` for a leaf
    """
    n = x.shape[0]
    max_nodes = 2 * n + 8  # 4-ary tree: internal nodes <= leaves <= n
    x0 = np.zeros(max_nodes)
    y0 = np.zeros(max_nodes)
    w = np.zeros(max_nodes)
    h = np.zeros(max_nodes)
    cx = np.zeros(max_nodes)
    cy = np.zeros(max_nodes)
    mass = np.zeros(max_nodes)
    child0 = np.full(max_nodes, -1, dtype=np.int64)
    child1 = np.full(max_nodes, -1, dtype=np.int64)
    child2 = np.full(max_nodes, -1, dtype=np.int64)
    child3 = np.full(max_nodes, -1, dtype=np.int64)
    leaf_start = np.full(max_nodes, -1, dtype=np.int64)
    leaf_count = np.zeros(max_nodes, dtype=np.int64)
    leaf_points = np.zeros(n, dtype=np.int64)

    # root covers the data bbox with a tiny margin
    lo_x, lo_y = float(x.min()), float(y.min())
    span_x = float(x.max()) - lo_x
    span_y = float(y.max()) - lo_y
    x0[0] = lo_x - _EPS
    y0[0] = lo_y - _EPS
    w[0] = span_x + 2 * _EPS
    h[0] = span_y + 2 * _EPS

    order = np.arange(n, dtype=np.int64)
    stack_i = np.zeros(2 * max_nodes, dtype=np.int64)
    stack_s = np.zeros(2 * max_nodes, dtype=np.int64)
    stack_e = np.zeros(2 * max_nodes, dtype=np.int64)
    sp = 0
    stack_i[sp], stack_s[sp], stack_e[sp] = 0, 0, n
    sp += 1
    n_nodes = 1
    n_leaf_pts = 0
    tmp = np.zeros(n, dtype=np.int64)

    while sp > 0:
        sp -= 1
        node, s, e = stack_i[sp], stack_s[sp], stack_e[sp]
        cnt = e - s
        if cnt == 0:
            continue
        # bbox of this node's points
        seg_x = x[order[s:e]]
        seg_y = y[order[s:e]]
        minx, maxx = float(seg_x.min()), float(seg_x.max())
        miny, maxy = float(seg_y.min()), float(seg_y.max())
        node_w = maxx - minx
        node_h = maxy - miny
        if cnt <= BUCKET or node_w < _EPS or node_h < _EPS:
            # leaf
            cx[node] = float(seg_x.mean())
            cy[node] = float(seg_y.mean())
            mass[node] = cnt
            leaf_start[node] = n_leaf_pts
            leaf_count[node] = cnt
            leaf_points[n_leaf_pts:n_leaf_pts + cnt] = order[s:e]
            n_leaf_pts += cnt
            continue
        # internal: create four children, partition the point range by quadrant
        mx = (minx + maxx) * 0.5
        my = (miny + maxy) * 0.5
        base = n_nodes
        child0[node] = base
        child1[node] = base + 1
        child2[node] = base + 2
        child3[node] = base + 3
        x0[base] = minx;  y0[base] = miny
        x0[base + 1] = mx; y0[base + 1] = miny
        x0[base + 2] = minx; y0[base + 2] = my
        x0[base + 3] = mx; y0[base + 3] = my
        for k in range(4):
            w[base + k] = (maxx - minx) * 0.5
            h[base + k] = (maxy - miny) * 0.5
        # count points per quadrant
        q = np.zeros(4, dtype=np.int64)
        for k in range(s, e):
            idx = order[k]
            qx = 0 if x[idx] < mx else 1
            qy = 0 if y[idx] < my else 1
            q[qx + 2 * qy] += 1
        # node-local scatter (indices 0..cnt-1), then copy back into order[s:e]
        seg_tmp = tmp[:cnt]
        pos = np.zeros(4, dtype=np.int64)
        acc = 0
        for k in range(4):
            pos[k] = acc
            acc += q[k]
        for k in range(s, e):
            idx = order[k]
            qx = 0 if x[idx] < mx else 1
            qy = 0 if y[idx] < my else 1
            qq = qx + 2 * qy
            seg_tmp[pos[qq]] = idx
            pos[qq] += 1
        order[s:e] = seg_tmp
        for k in range(4):
            stack_i[sp], stack_s[sp], stack_e[sp] = base + k, s + sum(q[:k]), s + sum(q[: k + 1])
            sp += 1
        n_nodes += 4

    # aggregate mass / centroid bottom-up. Children always have HIGHER indices
    # than their parent (created during the parent's split), so iterating in
    # reverse index order visits every child before its parent.
    for node in range(n_nodes - 1, -1, -1):
        if child0[node] < 0:
            continue
        m = 0.0
        sx = 0.0
        sy = 0.0
        for c in (child0[node], child1[node], child2[node], child3[node]):
            m += mass[c]
            sx += cx[c] * mass[c]
            sy += cy[c] * mass[c]
        mass[node] = m
        if m > 0:
            cx[node] = sx / m
            cy[node] = sy / m

    return {
        "x0": x0[:n_nodes], "y0": y0[:n_nodes], "w": w[:n_nodes], "h": h[:n_nodes],
        "cx": cx[:n_nodes], "cy": cy[:n_nodes], "mass": mass[:n_nodes],
        "child0": child0[:n_nodes], "child1": child1[:n_nodes],
        "child2": child2[:n_nodes], "child3": child3[:n_nodes],
        "leaf_start": leaf_start[:n_nodes], "leaf_count": leaf_count[:n_nodes],
        "leaf_points": leaf_points[:n_leaf_pts],
    }


class _QuadNode:
    """2-D quadtree node for the pure-Python reference backend. Either a
    leaf holding point indices, or internal with four children."""

    __slots__ = ("x0", "y0", "w", "h", "cx", "cy", "mass",
                 "indices", "children")

    def __init__(self, x0, y0, w, h):
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.cx = 0.0
        self.cy = 0.0
        self.mass = 0.0
        self.indices = []
        self.children = None

    def _split(self, points):
        hw, hh = self.w * 0.5, self.h * 0.5
        mx, my = self.x0 + hw, self.y0 + hh
        self.children = [
            _QuadNode(self.x0, self.y0, hw, hh),
            _QuadNode(mx, self.y0, hw, hh),
            _QuadNode(self.x0, my, hw, hh),
            _QuadNode(mx, my, hw, hh),
        ]
        idxs, self.indices = self.indices, None
        for idx in idxs:
            p = points[idx]
            child = 0 if p[0] < mx else 1
            child += 0 if p[1] < my else 2
            self.children[child].indices.append(idx)


def _build_ref_tree(points):
    n = points.shape[0]
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    w = max(hi[0] - lo[0], 1e-12)
    h = max(hi[1] - lo[1], 1e-12)
    root = _QuadNode(lo[0], lo[1], w * 1.001 + 1e-12, h * 1.001 + 1e-12)
    for i in range(n):
        node = root
        while node.children is not None:
            p = points[i]
            hw, hh = node.w * 0.5, node.h * 0.5
            mx, my = node.x0 + hw, node.y0 + hh
            child = 0 if p[0] < mx else 1
            child += 0 if p[1] < my else 2
            node = node.children[child]
        node.indices.append(i)
        if len(node.indices) > 16:
            node._split(points)

    def _finalize(node):
        if node.children is None:
            if node.indices:
                pts = points[node.indices]
                node.cx = float(pts[:, 0].mean())
                node.cy = float(pts[:, 1].mean())
                node.mass = len(node.indices)
            return
        for c in node.children:
            _finalize(c)
            node.cx += c.cx * c.mass
            node.cy += c.cy * c.mass
            node.mass += c.mass
        if node.mass > 0:
            node.cx /= node.mass
            node.cy /= node.mass

    _finalize(root)
    return root


def _accumulate_ref(node, pos, p, i, k_sq, theta, eps, disp):
    """Pure-Python reference accumulation (validation path only)."""
    if node.mass == 0:
        return
    dx = p[0] - node.cx
    dy = p[1] - node.cy
    d2 = dx * dx + dy * dy
    d = np.sqrt(d2)
    if d < eps:
        d = eps
        d2 = eps * eps
    if node.children is None:
        for j in node.indices:
            if j == i:
                continue
            pj = pos[j]
            ex = p[0] - pj[0]
            ey = p[1] - pj[1]
            dist2 = ex * ex + ey * ey
            dist = np.sqrt(dist2)
            if dist < eps:
                dist = eps
                dist2 = eps * eps
            c = k_sq / dist2
            disp[i, 0] += c * ex
            disp[i, 1] += c * ey
        return
    s = max(node.w, node.h)
    if s / d < theta:
        c = k_sq * node.mass / d2
        disp[i, 0] += c * dx
        disp[i, 1] += c * dy
        return
    for child in node.children:
        _accumulate_ref(child, pos, p, i, k_sq, theta, eps, disp)


def _accumulate_jit(x, y, T, k_sq, theta, eps):
    """JIT accumulation over the flat tree T. Returns disp [N,2]."""
    import numba

    @numba.njit(parallel=False, cache=True, fastmath=True)
    def _run(x, y, x0, y0, w, h, cx, cy, mass,
             c0, c1, c2, c3, lstart, lcount, lpts, k_sq, theta, eps):
        n = x.shape[0]
        disp = np.zeros((n, 2))
        stack = np.zeros(2 * cx.shape[0] + 16, dtype=np.int64)
        for i in range(n):
            px = x[i]
            py = y[i]
            sp = 1
            stack[0] = 0
            while sp > 0:
                sp -= 1
                node = stack[sp]
                if mass[node] == 0:
                    continue
                dx = px - cx[node]
                dy = py - cy[node]
                d2 = dx * dx + dy * dy
                d = np.sqrt(d2)
                if d < eps:
                    d = eps
                    d2 = eps * eps
                if c0[node] < 0:
                    # leaf: exact over its points
                    s0 = lstart[node]
                    c0n = lcount[node]
                    for k in range(c0n):
                        j = lpts[s0 + k]
                        if j == i:
                            continue
                        ex = px - x[j]
                        ey = py - y[j]
                        dist2 = ex * ex + ey * ey
                        dist = np.sqrt(dist2)
                        if dist < eps:
                            dist = eps
                            dist2 = eps * eps
                        cc = k_sq / dist2
                        disp[i, 0] += cc * ex
                        disp[i, 1] += cc * ey
                else:
                    cell_w = w[node]
                    cell_h = h[node]
                    s = cell_w if cell_w > cell_h else cell_h
                    if s / d < theta:
                        cc = k_sq * mass[node] / d2
                        disp[i, 0] += cc * dx
                        disp[i, 1] += cc * dy
                    else:
                        stack[sp] = c0[node]; sp += 1
                        stack[sp] = c1[node]; sp += 1
                        stack[sp] = c2[node]; sp += 1
                        stack[sp] = c3[node]; sp += 1
        return disp

    return _run(x, y, T["x0"], T["y0"], T["w"], T["h"], T["cx"], T["cy"],
                T["mass"], T["child0"], T["child1"], T["child2"], T["child3"],
                T["leaf_start"], T["leaf_count"], T["leaf_points"],
                float(k_sq), float(theta), float(eps))


def fr_repulsion_barnes_hut(pos: np.ndarray, k: float, theta: float = 0.7,
                            eps: float = 0.01, backend: str = "jit") -> np.ndarray:
    """Repulsive displacement [N,2] via Barnes-Hut.

    backend: "jit" (numba, default, for timing) or "ref" (pure Python, for
    validation). Matches brute force `k^2 * sum_j (p_i-p_j)/|p_i-p_j|^2` up
    to the theta approximation error.
    """
    n = pos.shape[0]
    k_sq = float(k) ** 2
    if n == 0:
        return np.zeros((0, 2))
    if backend == "ref":
        root = _build_ref_tree(pos)
        disp = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            _accumulate_ref(root, pos, pos[i], i, k_sq, theta, eps, disp)
        return disp
    T = _build_tree(pos[:, 0].astype(np.float64), pos[:, 1].astype(np.float64))
    return _accumulate_jit(pos[:, 0].astype(np.float64),
                           pos[:, 1].astype(np.float64), T, k_sq, theta, eps)


def fr_step(pos: np.ndarray, edges: np.ndarray, k: float, temperature: float,
            theta: float = 0.7, eps: float = 0.01, backend: str = "jit") -> np.ndarray:
    """One full FR step with Barnes-Hut repulsion; attraction exact over E.

    edges: [2, E] may be bidirectional (both (u,v) and (v,u)) or undirected
    (each pair once).  The attraction term deduplicates to unique undirected
    pairs before summing so the formula is correct in both cases.
    displacement = repulsion + attraction, capped by temperature.
    """
    disp = fr_repulsion_barnes_hut(pos, k, theta=theta, eps=eps, backend=backend)

    # Deduplicate to unique undirected edges so each pair is counted once.
    # Fix (R1-v2, 2026-08-17): original ex = pos[src]-pos[dst] was wrong-sign
    # (repulsive).  Correct: ex = pos[dst]-pos[src] (toward dst = attractive).
    e_sorted = np.sort(np.stack([edges[0], edges[1]], axis=1), axis=1)
    _, uniq_idx = np.unique(e_sorted, axis=0, return_index=True)
    src_u = e_sorted[uniq_idx, 0]
    dst_u = e_sorted[uniq_idx, 1]
    ex = pos[dst_u] - pos[src_u]                   # toward dst
    d = np.linalg.norm(ex, axis=1)
    d = np.maximum(d, eps)
    c = d / k
    ux = ex[:, 0] / d
    uy = ex[:, 1] / d
    np.add.at(disp[:, 0], src_u,  c * ux)          # pull src toward dst
    np.add.at(disp[:, 1], src_u,  c * uy)
    np.add.at(disp[:, 0], dst_u, -c * ux)          # pull dst toward src
    np.add.at(disp[:, 1], dst_u, -c * uy)

    length = np.linalg.norm(disp, axis=1)
    length = np.where(length < eps, 0.1, length)
    return pos + np.einsum("ij,i->ij", disp, temperature / length)
