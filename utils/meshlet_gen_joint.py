"""Stage-2 joint-cost greedy meshlet generator.

Six features per candidate triangle:
    1. plane_resid:   incremental PCA 3rd-eigval sqrt after adding cand
    2. boundary_perim: change in count of boundary edges (perimeter delta)
    3. strip_cont:    1 if cand does NOT continue current strip tail, else 0
    4. normal_sim:    1 - dot(cand_normal, avg_normal)         (planarity proxy)
    5. shared_edges:  1 - (# adj already in meshlet) / 3       (compactness)
    6. bfs_depth:     BFS hops from seed to cand                (compactness)

cost(cand) = sum_i w_i * znorm(feature_i)
Pick argmin cost.

Default weights uniform 1/6. Stage 3 (weight search) tunes them.
Feature normalization: z-score using sidecar JSON, calibrated once on
stanford-bunny+greedy. Without normalization, raw scales differ ~1000x
and weight search just learns inverse scales.
"""

import numpy as np
import heapq


# ============================================================
# Default normalization (calibrated; can be overridden via sidecar)
# ============================================================

# These are coarse defaults; calibrate via meshlet_weight_search if needed.
DEFAULT_FEATURE_NORMS = {
    "plane_resid":    {"mean": 1e-3, "std": 1e-3},
    "boundary_perim": {"mean": 1.0,  "std": 1.0},
    "strip_cont":     {"mean": 0.7,  "std": 0.5},
    "normal_sim":     {"mean": 0.1,  "std": 0.1},
    "shared_edges":   {"mean": 0.6,  "std": 0.2},
    "bfs_depth":      {"mean": 5.0,  "std": 3.0},
}


def _znorm(val, key, norms):
    n = norms.get(key, DEFAULT_FEATURE_NORMS[key])
    return (val - n["mean"]) / max(1e-9, n["std"])


def _resolve_norm(key: str, norms) -> tuple[float, float]:
    """Return (-mean, inv_std) for a feature. Inlined into the hot path so
    z-scoring is a single fused subtract-multiply."""
    n = norms.get(key, DEFAULT_FEATURE_NORMS[key])
    return -float(n["mean"]), 1.0 / max(1e-9, float(n["std"]))


def _min_eig_3x3_sym(m00, m01, m02, m11, m12, m22):
    """Smallest eigenvalue of a 3x3 symmetric matrix in closed form.

    Smith's trigonometric algorithm. ~100x faster than np.linalg.eigvalsh
    on a 3x3 (dispatch + setup is the bulk of numpy's cost).
    """
    p1 = m01 * m01 + m02 * m02 + m12 * m12
    q = (m00 + m11 + m22) / 3.0
    if p1 < 1e-30:
        return min(m00, m11, m22)
    p2 = ((m00 - q) ** 2 + (m11 - q) ** 2 + (m22 - q) ** 2 + 2.0 * p1)
    p = (p2 / 6.0) ** 0.5
    if p < 1e-30:
        return q
    b00 = (m00 - q) / p
    b11 = (m11 - q) / p
    b22 = (m22 - q) / p
    b01 = m01 / p
    b02 = m02 / p
    b12 = m12 / p
    detB = (b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02))
    r = max(-1.0, min(1.0, 0.5 * detB))
    import math
    phi = math.acos(r) / 3.0
    return q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)


# ============================================================
# Incremental running stats per meshlet
# ============================================================

class _MeshletState:
    """Tracks running stats for a growing meshlet.

    Boundary edges: edges with exactly one tri in meshlet.
    Plane fit: closed-form 3x3 SVD on vertex covariance.
    """

    def __init__(self, seed, tris_np, tri_adj, face_normals, verts_np):
        self.tris_np = tris_np
        self.tri_adj = tri_adj
        self.face_normals = face_normals
        self.verts_np = verts_np
        self.tri_set = {seed}
        self.tris = [seed]
        self.verts = set(int(v) for v in tris_np[seed])
        sn = face_normals[seed]
        # Avg normal as Python scalars — avoid numpy dispatch in inner loop.
        self.an_x = float(sn[0])
        self.an_y = float(sn[1])
        self.an_z = float(sn[2])
        # Boundary edges: tri's 3 edges all on boundary initially
        self.bnd_edges = set()
        for (a, b) in self._tri_edges(seed, tris_np):
            self.bnd_edges.add((min(a, b), max(a, b)))
        # Strip tail tracking: last added tri + edge into it
        self.tail = seed
        self.tail_in_edge = None
        # BFS depth from seed
        self.depth = {seed: 0}
        # Incremental PCA accumulators as Python floats (faster than numpy
        # dispatch on size-1-to-3 arrays). 3 sums + 6 unique outer-product
        # entries cover a symmetric 3x3.
        self._sx = self._sy = self._sz = 0.0
        self._o00 = self._o11 = self._o22 = 0.0
        self._o01 = self._o02 = self._o12 = 0.0
        for v in tris_np[seed]:
            p = verts_np[int(v)]
            px, py, pz = float(p[0]), float(p[1]), float(p[2])
            self._sx += px; self._sy += py; self._sz += pz
            self._o00 += px * px; self._o11 += py * py; self._o22 += pz * pz
            self._o01 += px * py; self._o02 += px * pz; self._o12 += py * pz

    @staticmethod
    def _tri_edges(t_idx_or_tri, tris_np=None):
        if tris_np is not None:
            tri = tris_np[t_idx_or_tri]
        else:
            tri = t_idx_or_tri
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        return [(a, b), (b, c), (c, a)]

    def shared_with_meshlet(self, cand):
        return sum(1 for nb in self.tri_adj[cand] if nb in self.tri_set)

    def cand_strip_continues(self, cand):
        """True if cand shares an edge with current tail."""
        return cand in self.tri_adj[self.tail]

    def perim_delta_if_added(self, cand):
        """Change in boundary edge count if cand were added."""
        added = 0
        removed = 0
        for (a, b) in self._tri_edges(cand, self.tris_np):
            e = (min(a, b), max(a, b))
            if e in self.bnd_edges:
                removed += 1   # interior edge now
            else:
                added += 1     # new boundary edge
        return added - removed

    def plane_resid_if_added(self, cand):
        """3rd PCA eigval sqrt of vertex set after adding cand.

        Incremental: only the cand's *new* verts (≤3) contribute new terms
        to the running sum + outer-product. Hypothetical update; does not
        commit. Smallest eigenvalue via closed-form 3x3 symmetric solver.
        """
        verts = self.verts
        ts = self.tris_np[cand]
        n_curr = len(verts)
        ns_x, ns_y, ns_z = self._sx, self._sy, self._sz
        n00, n11, n22 = self._o00, self._o11, self._o22
        n01, n02, n12 = self._o01, self._o02, self._o12
        n_new = 0
        for vi in ts:
            v = int(vi)
            if v in verts:
                continue
            p = self.verts_np[v]
            px, py, pz = float(p[0]), float(p[1]), float(p[2])
            ns_x += px; ns_y += py; ns_z += pz
            n00 += px * px; n11 += py * py; n22 += pz * pz
            n01 += px * py; n02 += px * pz; n12 += py * pz
            n_new += 1
        n_total = n_curr + n_new
        if n_total < 4:
            return 0.0
        inv_n = 1.0 / n_total
        mx, my, mz = ns_x * inv_n, ns_y * inv_n, ns_z * inv_n
        denom = max(1, n_total - 1)
        c00 = (n00 - n_total * mx * mx) / denom
        c11 = (n11 - n_total * my * my) / denom
        c22 = (n22 - n_total * mz * mz) / denom
        c01 = (n01 - n_total * mx * my) / denom
        c02 = (n02 - n_total * mx * mz) / denom
        c12 = (n12 - n_total * my * mz) / denom
        emin = _min_eig_3x3_sym(c00, c01, c02, c11, c12, c22)
        return (max(0.0, emin)) ** 0.5

    def normal_dissim(self, cand):
        nx, ny, nz = self.an_x, self.an_y, self.an_z
        nn2 = nx * nx + ny * ny + nz * nz
        if nn2 < 1e-24:
            return 1.0
        fn = self.face_normals[cand]
        cos = (float(fn[0]) * nx + float(fn[1]) * ny + float(fn[2]) * nz) \
            / (nn2 ** 0.5)
        return 1.0 - cos

    def add(self, cand):
        # Update boundary edges
        for (a, b) in self._tri_edges(cand, self.tris_np):
            e = (min(a, b), max(a, b))
            if e in self.bnd_edges:
                self.bnd_edges.remove(e)
            else:
                self.bnd_edges.add(e)
        # Update tri/vert sets
        new_v = set(int(v) for v in self.tris_np[cand]) - self.verts
        self.tris.append(cand)
        self.tri_set.add(cand)
        self.verts.update(new_v)
        # Commit PCA accumulators for the newly added verts (Python floats)
        for v in new_v:
            p = self.verts_np[v]
            px, py, pz = float(p[0]), float(p[1]), float(p[2])
            self._sx += px; self._sy += py; self._sz += pz
            self._o00 += px * px; self._o11 += py * py; self._o22 += pz * pz
            self._o01 += px * py; self._o02 += px * pz; self._o12 += py * pz
        # Running normal — scalar math
        cnt = len(self.tris)
        fn = self.face_normals[cand]
        prev = cnt - 1
        inv = 1.0 / cnt
        self.an_x = (self.an_x * prev + float(fn[0])) * inv
        self.an_y = (self.an_y * prev + float(fn[1])) * inv
        self.an_z = (self.an_z * prev + float(fn[2])) * inv
        # Strip tail: if cand is adj to old tail, advance
        if cand in self.tri_adj[self.tail]:
            self.tail = cand
        # else tail unchanged (strip restart inside meshlet — costs)
        # Depth: cand depth = min(depth of adj tri in meshlet) + 1
        d = min((self.depth[nb] for nb in self.tri_adj[cand]
                 if nb in self.depth), default=0) + 1
        self.depth[cand] = d


# ============================================================
# Joint-cost generator
# ============================================================

DEFAULT_WEIGHTS = {
    "w1_plane_resid":    1.0 / 6,
    "w2_boundary_perim": 1.0 / 6,
    "w3_strip_cont":     1.0 / 6,
    "w4_normal_sim":     1.0 / 6,
    "w5_shared_edges":   1.0 / 6,
    "w6_bfs_depth":      1.0 / 6,
}


# Bayesian-opt-tuned weights on stanford-bunny (gp_minimize, n_calls=20).
# BPV gain vs uniform: 30.41 -> 30.00 (-0.41 BPV).
# Key insight: plane_resid + boundary_perim + normal_sim near-zero;
# all leverage in compactness (bfs_depth + shared_edges) + strip_cont.
LEARNED_WEIGHTS = {
    "w1_plane_resid":    0.0008,
    "w2_boundary_perim": 0.0008,
    "w3_strip_cont":     0.2086,
    "w4_normal_sim":     0.0008,
    "w5_shared_edges":   0.2638,
    "w6_bfs_depth":      0.5250,
}


def generate_meshlets_joint(tris_np, tri_adj, face_normals, face_centroids,
                            verts_np, max_tris=256, max_verts=256,
                            weights=None, feature_norms=None,
                            use_numba: bool = True):
    """Region-grow with joint cost. See module docstring for features.

    use_numba=True dispatches to the numba kernel (50-100x faster on big
    meshes, bit-exact output on identical inputs). Falls back to the Python
    implementation if numba is not installed.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if feature_norms is None:
        feature_norms = DEFAULT_FEATURE_NORMS

    if use_numba:
        try:
            from utils.meshlet_gen_joint_nb import generate_meshlets_joint_nb
            w1 = weights["w1_plane_resid"]
            w2 = weights["w2_boundary_perim"]
            w3 = weights["w3_strip_cont"]
            w4 = weights["w4_normal_sim"]
            w5 = weights["w5_shared_edges"]
            w6 = weights["w6_bfs_depth"]
            pr_nm, pr_is = _resolve_norm("plane_resid",    feature_norms)
            bp_nm, bp_is = _resolve_norm("boundary_perim", feature_norms)
            sc_nm, sc_is = _resolve_norm("strip_cont",     feature_norms)
            ns_nm, ns_is = _resolve_norm("normal_sim",     feature_norms)
            se_nm, se_is = _resolve_norm("shared_edges",   feature_norms)
            bd_nm, bd_is = _resolve_norm("bfs_depth",      feature_norms)
            K_const = (w1 * pr_is, w2 * bp_is, w3 * sc_is,
                       w4 * ns_is, w5 * se_is, w6 * bd_is)
            NM_const = (pr_nm, bp_nm, sc_nm, ns_nm, se_nm, bd_nm)
            return generate_meshlets_joint_nb(
                tris_np, tri_adj, face_normals, verts_np,
                max_tris=max_tris, max_verts=max_verts,
                K_const=K_const, NM_const=NM_const)
        except ImportError:
            pass  # fall through to Python implementation

    n = len(tris_np)
    visited = np.zeros(n, dtype=bool)
    meshlets = []

    # Seed order: min-degree first (mirrors greedy DFS strip cover spirit).
    # Pre-sort once (stable) — preserves prior tie-break (lowest tri-index).
    # Pointer advances past visited; O(n) total over all meshlets vs O(n)
    # per meshlet via np.where + argmin.
    deg = np.array([len(tri_adj[i]) for i in range(n)], dtype=np.int32)
    seed_order = np.argsort(deg, kind='stable')
    seed_ptr = 0

    # Pre-resolve weight constants so the inner loop avoids dict lookups.
    w1 = weights["w1_plane_resid"]
    w2 = weights["w2_boundary_perim"]
    w3 = weights["w3_strip_cont"]
    w4 = weights["w4_normal_sim"]
    w5 = weights["w5_shared_edges"]
    w6 = weights["w6_bfs_depth"]
    # Fold (-mean) and (inv_std) into per-feature constants and pre-multiply
    # by weight so the inner loop is: w_inv * (f + neg_mean).
    pr_nm, pr_is = _resolve_norm("plane_resid",    feature_norms)
    bp_nm, bp_is = _resolve_norm("boundary_perim", feature_norms)
    sc_nm, sc_is = _resolve_norm("strip_cont",     feature_norms)
    ns_nm, ns_is = _resolve_norm("normal_sim",     feature_norms)
    se_nm, se_is = _resolve_norm("shared_edges",   feature_norms)
    bd_nm, bd_is = _resolve_norm("bfs_depth",      feature_norms)
    K1 = w1 * pr_is
    K2 = w2 * bp_is
    K3 = w3 * sc_is
    K4 = w4 * ns_is
    K5 = w5 * se_is
    K6 = w6 * bd_is

    def cost_for(state, cand):
        f1 = state.plane_resid_if_added(cand)
        f2 = float(state.perim_delta_if_added(cand))
        f3 = 0.0 if state.cand_strip_continues(cand) else 1.0
        f4 = state.normal_dissim(cand)
        f5 = 1.0 - state.shared_with_meshlet(cand) / 3.0
        f6 = float(min((state.depth[nb] for nb in tri_adj[cand]
                        if nb in state.depth), default=0) + 1)
        return (K1 * (f1 + pr_nm)
                + K2 * (f2 + bp_nm)
                + K3 * (f3 + sc_nm)
                + K4 * (f4 + ns_nm)
                + K5 * (f5 + se_nm)
                + K6 * (f6 + bd_nm))

    for _ in range(n):
        # Advance past any newly-visited entries (added by inner loop).
        while seed_ptr < n and visited[seed_order[seed_ptr]]:
            seed_ptr += 1
        if seed_ptr >= n:
            break
        seed = int(seed_order[seed_ptr])
        seed_ptr += 1
        visited[seed] = True

        state = _MeshletState(seed, tris_np, tri_adj, face_normals, verts_np)

        # Frontier = unvisited adj of any tri in meshlet
        frontier = set(nb for nb in tri_adj[seed] if not visited[nb])

        while frontier:
            if len(state.tris) >= max_tris:
                break
            if len(state.verts) >= max_verts:
                break
            # Score all frontier candidates; pick min cost
            best_cand, best_cost = -1, np.inf
            for cand in frontier:
                if visited[cand]:
                    continue
                # Vert overflow guard
                new_v = set(int(v) for v in tris_np[cand]) - state.verts
                if len(state.verts) + len(new_v) > max_verts:
                    continue
                if len(state.tris) + 1 > max_tris:
                    continue
                c = cost_for(state, cand)
                if c < best_cost:
                    best_cost, best_cand = c, cand
            if best_cand < 0:
                break
            visited[best_cand] = True
            state.add(best_cand)
            frontier.discard(best_cand)
            for nb in tri_adj[best_cand]:
                if not visited[nb]:
                    frontier.add(nb)

        meshlets.append(state.tris)

    return meshlets
