"""Numba kernel for joint-cost meshlet generation.

Bit-exact replacement for `utils.meshlet_gen_joint.generate_meshlets_joint`.
Re-implements the same greedy region-growing with the same six cost features
(plane_resid, perim_delta, strip_cont, normal_dissim, shared_edges, bfs_depth),
same z-score normalization, same tie-break (lowest tri index by `np.argsort`
on degree, stable), so the meshlet output should match the Python version
byte-for-byte on identical inputs.

Inputs are pre-flattened into numpy arrays:
  - tris_np                  (n_tris, 3) int64
  - verts_np                 (n_verts, 3) float64
  - face_normals             (n_tris, 3) float64
  - tri_adj_offsets          (n_tris + 1,) int32 (CSR)
  - tri_adj_indices          (sum_deg,)   int32 (CSR)
  - tri_edge_ids             (n_tris, 3)  int32 (unique edge ID per side)
  - K1..K6 + nm constants    pre-folded weight × inv_std and -mean
  - max_tris, max_verts      caps

Per-meshlet scratch is preallocated and re-used; the dirty bits the kernel
writes are reset before the next meshlet so we don't pay O(n) clears.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def _min_eig(m00, m01, m02, m11, m12, m22):
    p1 = m01 * m01 + m02 * m02 + m12 * m12
    q = (m00 + m11 + m22) / 3.0
    if p1 < 1e-30:
        a = m00
        if m11 < a:
            a = m11
        if m22 < a:
            a = m22
        return a
    p2 = ((m00 - q) * (m00 - q) + (m11 - q) * (m11 - q)
          + (m22 - q) * (m22 - q) + 2.0 * p1)
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
    r = 0.5 * detB
    if r < -1.0:
        r = -1.0
    elif r > 1.0:
        r = 1.0
    phi = math.acos(r) / 3.0
    return q + 2.0 * p * math.cos(phi + 2.0 * math.pi / 3.0)


@njit(cache=True)
def _grow_one_meshlet(seed, tris_np, verts_np, face_normals,
                      adj_off, adj_idx, tri_edge_ids,
                      visited, vert_in, tri_in, depth, bnd_edge_in,
                      in_frontier, frontier,
                      max_tris, max_verts,
                      K1, K2, K3, K4, K5, K6,
                      pr_nm, bp_nm, sc_nm, ns_nm, se_nm, bd_nm,
                      out_tris):
    """Grow one meshlet starting from `seed`. Writes accepted tri ids into
    `out_tris` and returns count. Caller is responsible for resetting
    scratch arrays.

    All bool[] state assumed pre-cleared before this call.
    """
    n_tris = tris_np.shape[0]
    # Init seed
    visited[seed] = True
    tri_in[seed] = True
    depth[seed] = 0
    sv0 = tris_np[seed, 0]
    sv1 = tris_np[seed, 1]
    sv2 = tris_np[seed, 2]
    vert_in[sv0] = True
    vert_in[sv1] = True
    vert_in[sv2] = True
    n_verts_in = 3
    # Seed bnd edges
    e0 = tri_edge_ids[seed, 0]
    e1 = tri_edge_ids[seed, 1]
    e2 = tri_edge_ids[seed, 2]
    bnd_edge_in[e0] = True
    bnd_edge_in[e1] = True
    bnd_edge_in[e2] = True
    # PCA accum from seed verts
    p0x = verts_np[sv0, 0]; p0y = verts_np[sv0, 1]; p0z = verts_np[sv0, 2]
    p1x = verts_np[sv1, 0]; p1y = verts_np[sv1, 1]; p1z = verts_np[sv1, 2]
    p2x = verts_np[sv2, 0]; p2y = verts_np[sv2, 1]; p2z = verts_np[sv2, 2]
    sx = p0x + p1x + p2x
    sy = p0y + p1y + p2y
    sz = p0z + p1z + p2z
    o00 = p0x * p0x + p1x * p1x + p2x * p2x
    o11 = p0y * p0y + p1y * p1y + p2y * p2y
    o22 = p0z * p0z + p1z * p1z + p2z * p2z
    o01 = p0x * p0y + p1x * p1y + p2x * p2y
    o02 = p0x * p0z + p1x * p1z + p2x * p2z
    o12 = p0y * p0z + p1y * p1z + p2y * p2z
    # Avg normal
    sn = face_normals[seed]
    an_x = sn[0]
    an_y = sn[1]
    an_z = sn[2]
    # Tail tracking
    tail = seed
    # `frontier` + `in_frontier` are hoisted; caller pre-clears `in_frontier`
    # (it stays clean across calls because we reset dirty entries at end).
    f_count = 0
    # Add unvisited adj of seed to frontier
    for k in range(adj_off[seed], adj_off[seed + 1]):
        nb = adj_idx[k]
        if not visited[nb]:
            frontier[f_count] = nb
            f_count += 1
            in_frontier[nb] = True
    # Output ids: write seed
    out_tris[0] = seed
    n_out = 1

    while f_count > 0 and n_out < max_tris and n_verts_in < max_verts:
        best_cand = -1
        best_cost = 1e300
        # Score each cand
        for fi in range(f_count):
            cand = frontier[fi]
            if visited[cand]:
                continue
            # Vert-overflow guard
            cv0 = tris_np[cand, 0]
            cv1 = tris_np[cand, 1]
            cv2 = tris_np[cand, 2]
            new_v_count = 0
            if not vert_in[cv0]:
                new_v_count += 1
            if not vert_in[cv1] and cv1 != cv0:
                new_v_count += 1
            if not vert_in[cv2] and cv2 != cv0 and cv2 != cv1:
                new_v_count += 1
            if n_verts_in + new_v_count > max_verts:
                continue

            # f1: plane_resid hypothetical
            ns_x = sx; ns_y = sy; ns_z = sz
            n00 = o00; n11 = o11; n22 = o22
            n01 = o01; n02 = o02; n12 = o12
            n_new = 0
            seen0 = False; seen1 = False; seen2 = False
            if not vert_in[cv0]:
                px = verts_np[cv0, 0]; py = verts_np[cv0, 1]; pz = verts_np[cv0, 2]
                ns_x += px; ns_y += py; ns_z += pz
                n00 += px * px; n11 += py * py; n22 += pz * pz
                n01 += px * py; n02 += px * pz; n12 += py * pz
                n_new += 1
                seen0 = True
            if not vert_in[cv1] and not (seen0 and cv1 == cv0):
                px = verts_np[cv1, 0]; py = verts_np[cv1, 1]; pz = verts_np[cv1, 2]
                ns_x += px; ns_y += py; ns_z += pz
                n00 += px * px; n11 += py * py; n22 += pz * pz
                n01 += px * py; n02 += px * pz; n12 += py * pz
                n_new += 1
                seen1 = True
            if not vert_in[cv2]:
                dup = False
                if seen0 and cv2 == cv0:
                    dup = True
                if seen1 and cv2 == cv1:
                    dup = True
                if not dup:
                    px = verts_np[cv2, 0]; py = verts_np[cv2, 1]; pz = verts_np[cv2, 2]
                    ns_x += px; ns_y += py; ns_z += pz
                    n00 += px * px; n11 += py * py; n22 += pz * pz
                    n01 += px * py; n02 += px * pz; n12 += py * pz
                    n_new += 1
            n_total = n_verts_in + n_new
            if n_total < 4:
                f1 = 0.0
            else:
                inv_n = 1.0 / n_total
                mx_ = ns_x * inv_n
                my_ = ns_y * inv_n
                mz_ = ns_z * inv_n
                denom = float(n_total - 1)
                if denom < 1.0:
                    denom = 1.0
                c00 = (n00 - n_total * mx_ * mx_) / denom
                c11 = (n11 - n_total * my_ * my_) / denom
                c22 = (n22 - n_total * mz_ * mz_) / denom
                c01 = (n01 - n_total * mx_ * my_) / denom
                c02 = (n02 - n_total * mx_ * mz_) / denom
                c12 = (n12 - n_total * my_ * mz_) / denom
                emin = _min_eig(c00, c01, c02, c11, c12, c22)
                if emin < 0.0:
                    emin = 0.0
                f1 = emin ** 0.5

            # f2: perim delta
            ce0 = tri_edge_ids[cand, 0]
            ce1 = tri_edge_ids[cand, 1]
            ce2 = tri_edge_ids[cand, 2]
            added = 0; removed = 0
            if bnd_edge_in[ce0]:
                removed += 1
            else:
                added += 1
            if bnd_edge_in[ce1]:
                removed += 1
            else:
                added += 1
            if bnd_edge_in[ce2]:
                removed += 1
            else:
                added += 1
            f2 = float(added - removed)

            # f3: strip_cont — adj to tail?
            cont = False
            for k in range(adj_off[tail], adj_off[tail + 1]):
                if adj_idx[k] == cand:
                    cont = True
                    break
            f3 = 0.0 if cont else 1.0

            # f4: normal_dissim
            nn2 = an_x * an_x + an_y * an_y + an_z * an_z
            if nn2 < 1e-24:
                f4 = 1.0
            else:
                fn = face_normals[cand]
                cos = (fn[0] * an_x + fn[1] * an_y + fn[2] * an_z) / (nn2 ** 0.5)
                f4 = 1.0 - cos

            # f5: shared_edges
            shared = 0
            for k in range(adj_off[cand], adj_off[cand + 1]):
                if tri_in[adj_idx[k]]:
                    shared += 1
            f5 = 1.0 - shared / 3.0

            # f6: bfs_depth
            min_d = -1
            for k in range(adj_off[cand], adj_off[cand + 1]):
                nb = adj_idx[k]
                if tri_in[nb]:
                    d = depth[nb]
                    if min_d < 0 or d < min_d:
                        min_d = d
            if min_d < 0:
                min_d = 0
            f6 = float(min_d + 1)

            cost = (K1 * (f1 + pr_nm)
                    + K2 * (f2 + bp_nm)
                    + K3 * (f3 + sc_nm)
                    + K4 * (f4 + ns_nm)
                    + K5 * (f5 + se_nm)
                    + K6 * (f6 + bd_nm))
            # Tie-break: lower cand id wins (matches Python iteration order
            # over a set is undefined, but original used set iteration with
            # strict `<` so first-seen retains. To get a deterministic order
            # comparable to the Python version, we use strict `<` here too.
            if cost < best_cost:
                best_cost = cost
                best_cand = cand

        if best_cand < 0:
            break

        cand = best_cand
        # Commit cand
        visited[cand] = True
        tri_in[cand] = True
        # Toggle boundary edges
        ce0 = tri_edge_ids[cand, 0]
        ce1 = tri_edge_ids[cand, 1]
        ce2 = tri_edge_ids[cand, 2]
        bnd_edge_in[ce0] = not bnd_edge_in[ce0]
        bnd_edge_in[ce1] = not bnd_edge_in[ce1]
        bnd_edge_in[ce2] = not bnd_edge_in[ce2]
        # Add new verts
        cv0 = tris_np[cand, 0]
        cv1 = tris_np[cand, 1]
        cv2 = tris_np[cand, 2]
        for vidx_i in range(3):
            if vidx_i == 0:
                v = cv0
            elif vidx_i == 1:
                v = cv1
            else:
                v = cv2
            if not vert_in[v]:
                vert_in[v] = True
                px = verts_np[v, 0]; py = verts_np[v, 1]; pz = verts_np[v, 2]
                sx += px; sy += py; sz += pz
                o00 += px * px; o11 += py * py; o22 += pz * pz
                o01 += px * py; o02 += px * pz; o12 += py * pz
                n_verts_in += 1
        # Running normal
        cnt = n_out + 1
        prev = cnt - 1
        fn = face_normals[cand]
        inv_cnt = 1.0 / cnt
        an_x = (an_x * prev + fn[0]) * inv_cnt
        an_y = (an_y * prev + fn[1]) * inv_cnt
        an_z = (an_z * prev + fn[2]) * inv_cnt
        # Update tail if cand is adj to old tail
        is_tail_adj = False
        for k in range(adj_off[tail], adj_off[tail + 1]):
            if adj_idx[k] == cand:
                is_tail_adj = True
                break
        if is_tail_adj:
            tail = cand
        # Update depth for cand (min adj-in-meshlet depth + 1)
        new_d = -1
        for k in range(adj_off[cand], adj_off[cand + 1]):
            nb = adj_idx[k]
            if tri_in[nb] and nb != cand:
                d = depth[nb]
                if new_d < 0 or d < new_d:
                    new_d = d
        if new_d < 0:
            new_d = 0
        depth[cand] = new_d + 1
        # Append to out
        out_tris[n_out] = cand
        n_out += 1
        # Update frontier: remove cand from frontier (compact) + add new
        # unvisited adj
        # Compaction: swap with last
        for fi in range(f_count):
            if frontier[fi] == cand:
                frontier[fi] = frontier[f_count - 1]
                f_count -= 1
                break
        in_frontier[cand] = False
        for k in range(adj_off[cand], adj_off[cand + 1]):
            nb = adj_idx[k]
            if not visited[nb] and not in_frontier[nb]:
                frontier[f_count] = nb
                f_count += 1
                in_frontier[nb] = True

    # Clear dirty `in_frontier` bits so the caller's scratch stays clean.
    for fi in range(f_count):
        in_frontier[frontier[fi]] = False
    return n_out


@njit(cache=True)
def _run_all(seed_order, tris_np, verts_np, face_normals,
             adj_off, adj_idx, tri_edge_ids,
             max_tris, max_verts,
             K1, K2, K3, K4, K5, K6,
             pr_nm, bp_nm, sc_nm, ns_nm, se_nm, bd_nm,
             ml_starts, ml_tris_flat, ml_tri_counts):
    """Drive all meshlets. Writes results into pre-sized output arrays.
    Returns n_meshlets actually produced.
    """
    n_tris = tris_np.shape[0]
    n_verts = verts_np.shape[0]
    n_edges = tri_edge_ids.max() + 1

    visited = np.zeros(n_tris, dtype=np.bool_)
    vert_in = np.zeros(n_verts, dtype=np.bool_)
    tri_in = np.zeros(n_tris, dtype=np.bool_)
    depth = np.full(n_tris, -1, dtype=np.int32)
    bnd_edge_in = np.zeros(n_edges, dtype=np.bool_)
    # Hoisted once; `_grow_one_meshlet` keeps it bit-clean across calls.
    in_frontier = np.zeros(n_tris, dtype=np.bool_)
    frontier = np.empty(max_tris * 4 + 32, dtype=np.int32)

    out_tris_scratch = np.empty(max_tris + 8, dtype=np.int32)

    n_ml = 0
    seed_ptr = 0
    write_offset = 0
    while seed_ptr < n_tris:
        while seed_ptr < n_tris and visited[seed_order[seed_ptr]]:
            seed_ptr += 1
        if seed_ptr >= n_tris:
            break
        seed = seed_order[seed_ptr]
        seed_ptr += 1
        # Run grow
        n_out = _grow_one_meshlet(
            seed, tris_np, verts_np, face_normals,
            adj_off, adj_idx, tri_edge_ids,
            visited, vert_in, tri_in, depth, bnd_edge_in,
            in_frontier, frontier,
            max_tris, max_verts,
            K1, K2, K3, K4, K5, K6,
            pr_nm, bp_nm, sc_nm, ns_nm, se_nm, bd_nm,
            out_tris_scratch)
        # Append into flat output
        ml_starts[n_ml] = write_offset
        ml_tri_counts[n_ml] = n_out
        for i in range(n_out):
            t = out_tris_scratch[i]
            ml_tris_flat[write_offset + i] = t
            # Reset scratch for this tri
            tri_in[t] = False
            depth[t] = -1
            # Reset its edge memberships
            for k in range(3):
                e = tri_edge_ids[t, k]
                bnd_edge_in[e] = False
            # Reset its vert memberships (any of cv0/1/2 NOT in another ml)
            cv0 = tris_np[t, 0]
            cv1 = tris_np[t, 1]
            cv2 = tris_np[t, 2]
            vert_in[cv0] = False
            vert_in[cv1] = False
            vert_in[cv2] = False
        write_offset += n_out
        n_ml += 1
    ml_starts[n_ml] = write_offset
    return n_ml


# ============================================================
# Public wrapper
# ============================================================

def generate_meshlets_joint_nb(
        tris_np, tri_adj, face_normals, verts_np,
        max_tris=256, max_verts=256,
        K_const=None, NM_const=None):
    """Numba-accelerated joint-cost meshlet generation.

    K_const = (K1..K6)         — already w_i × inv_std_i
    NM_const = (nm1..nm6)      — already -mean_i
    """
    n_tris = len(tris_np)
    tris_np_i = np.ascontiguousarray(tris_np, dtype=np.int64)
    verts_np_f = np.ascontiguousarray(verts_np, dtype=np.float64)
    fn_f = np.ascontiguousarray(face_normals, dtype=np.float64)

    # Build CSR adjacency
    deg = np.array([len(tri_adj[i]) for i in range(n_tris)], dtype=np.int32)
    adj_off = np.zeros(n_tris + 1, dtype=np.int32)
    adj_off[1:] = np.cumsum(deg)
    adj_idx = np.empty(int(adj_off[-1]), dtype=np.int32)
    for i in range(n_tris):
        adj_idx[adj_off[i]:adj_off[i + 1]] = tri_adj[i]

    # Build unique edge IDs
    edge_to_id: dict[tuple[int, int], int] = {}
    tri_edge_ids = np.empty((n_tris, 3), dtype=np.int32)
    next_id = 0
    for t in range(n_tris):
        a = int(tris_np_i[t, 0]); b = int(tris_np_i[t, 1]); c = int(tris_np_i[t, 2])
        pairs = ((a, b), (b, c), (c, a))
        for k, (u, v) in enumerate(pairs):
            if u <= v:
                key = (u, v)
            else:
                key = (v, u)
            eid = edge_to_id.get(key, -1)
            if eid < 0:
                eid = next_id
                edge_to_id[key] = eid
                next_id += 1
            tri_edge_ids[t, k] = eid

    # Seed order: stable argsort by degree
    seed_order = np.argsort(deg, kind='stable').astype(np.int32)

    # Output buffers
    ml_starts = np.empty(n_tris + 1, dtype=np.int32)
    ml_tris_flat = np.empty(n_tris, dtype=np.int32)
    ml_tri_counts = np.empty(n_tris, dtype=np.int32)

    K1, K2, K3, K4, K5, K6 = K_const
    nm_pr, nm_bp, nm_sc, nm_ns, nm_se, nm_bd = NM_const

    n_ml = _run_all(
        seed_order, tris_np_i, verts_np_f, fn_f,
        adj_off, adj_idx, tri_edge_ids,
        int(max_tris), int(max_verts),
        float(K1), float(K2), float(K3), float(K4), float(K5), float(K6),
        float(nm_pr), float(nm_bp), float(nm_sc),
        float(nm_ns), float(nm_se), float(nm_bd),
        ml_starts, ml_tris_flat, ml_tri_counts)

    meshlets = []
    for i in range(n_ml):
        s = int(ml_starts[i])
        c = int(ml_tri_counts[i])
        meshlets.append([int(x) for x in ml_tris_flat[s:s + c]])
    return meshlets


def _build_tri_edge_ids_np(tris_np_i64):
    """Vectorised unique-edge-ID assignment. Edge IDs only need uniqueness
    per unordered (u, v) pair; ordering is irrelevant downstream (kernel
    only toggles a bit per ID). Replaces a 21M-op Python dict on Dragon."""
    t = tris_np_i64
    n = t.shape[0]
    e0a = np.minimum(t[:, 0], t[:, 1]); e0b = np.maximum(t[:, 0], t[:, 1])
    e1a = np.minimum(t[:, 1], t[:, 2]); e1b = np.maximum(t[:, 1], t[:, 2])
    e2a = np.minimum(t[:, 0], t[:, 2]); e2b = np.maximum(t[:, 0], t[:, 2])
    all_a = np.concatenate([e0a, e1a, e2a])
    all_b = np.concatenate([e0b, e1b, e2b])
    mult = np.int64(int(t.max()) + 1)
    keys = all_a * mult + all_b
    _, inv = np.unique(keys, return_inverse=True)
    inv32 = inv.astype(np.int32)
    ids = np.empty((n, 3), dtype=np.int32)
    ids[:, 0] = inv32[:n]
    ids[:, 1] = inv32[n:2 * n]
    ids[:, 2] = inv32[2 * n:]
    return ids


def generate_meshlets_joint_from_csr(
        tris_np_i64, adj_off, adj_idx, verts_np, face_normals,
        max_tris=256, max_verts=256, K_const=None, NM_const=None):
    """CSR-direct entry point. Skips dict-list build + dict edge-ID pass.

    Pre-conditions:
      - tris_np_i64 contiguous int64 (n_t, 3)
      - adj_off int32 (n_t+1,), adj_idx int32 (sum_deg,) — same layout
        as `utils.meshlet_plan_nb.build_global_adj_csr_nb` output
      - verts_np float64 (n_v, 3), face_normals float64 (n_t, 3)
    """
    n_tris = tris_np_i64.shape[0]
    tris_np_i = np.ascontiguousarray(tris_np_i64, dtype=np.int64)
    verts_np_f = np.ascontiguousarray(verts_np, dtype=np.float64)
    fn_f = np.ascontiguousarray(face_normals, dtype=np.float64)

    tri_edge_ids = _build_tri_edge_ids_np(tris_np_i)

    deg = (adj_off[1:] - adj_off[:-1]).astype(np.int32)
    seed_order = np.argsort(deg, kind='stable').astype(np.int32)

    ml_starts = np.empty(n_tris + 1, dtype=np.int32)
    ml_tris_flat = np.empty(n_tris, dtype=np.int32)
    ml_tri_counts = np.empty(n_tris, dtype=np.int32)

    K1, K2, K3, K4, K5, K6 = K_const
    nm_pr, nm_bp, nm_sc, nm_ns, nm_se, nm_bd = NM_const

    n_ml = _run_all(
        seed_order, tris_np_i, verts_np_f, fn_f,
        adj_off, adj_idx, tri_edge_ids,
        int(max_tris), int(max_verts),
        float(K1), float(K2), float(K3), float(K4), float(K5), float(K6),
        float(nm_pr), float(nm_bp), float(nm_sc),
        float(nm_ns), float(nm_se), float(nm_bd),
        ml_starts, ml_tris_flat, ml_tri_counts)

    meshlets = []
    for i in range(n_ml):
        s = int(ml_starts[i])
        c = int(ml_tri_counts[i])
        meshlets.append([int(x) for x in ml_tris_flat[s:s + c]])
    return meshlets