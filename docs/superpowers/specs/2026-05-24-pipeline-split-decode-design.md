# Pipeline-Split Decode for STRIDE-dup CUDA Kernel

**Date:** 2026-05-24
**Status:** Approved design
**Author:** maletsden (with Claude Code)

## Problem

Current `stride_dup_decode_kernel` (v3 buffered-bit-reader) is compute-bound on lane 0:

| Metric | Value |
|---|---|
| Dragon throughput | 1798 mtps (best-of-3) |
| DGF baseline | 2169 mtps |
| Gap | 0.83× |
| Compute SM throughput | 79.6% |
| Avg active threads per warp | **1.19 / 32** |
| L1 hit rate | 92% (good) |
| Theoretical occupancy | 50% (SMEM-limited) |

Lane 0 does all bit-stream work serially. 31/32 lanes idle = 96% of warp ALU wasted.

Rice/Exp-Golomb decode has inherent bit-chain dependency that cannot be lane-parallelized without bit-offset metadata (would defeat Rice's purpose). Profile-driven alternative: split decode work into independent stages handled by separate lanes within the same warp.

## Goal

Reach DGF parity or better on Dragon (≥2169 mtps), bit-exact tri reconstruction, sub-1e-4 vertex error vs CPU reference. Maintain or improve gains on smaller meshes (no regressions worse than 5%).

## Approach

**Pipeline split across two lanes** in the same warp:

- **Lane 0:** decode connectivity (strip + FIFO) → walk_pack[], walk_kind_bm[], tris_s[]. Then run predictor loop pulling pre-decoded residuals from SMEM.
- **Lane 1:** decode anchor + tag headers + ALL Rice/EG residuals into `resid_s[][3]` SMEM.

Both lanes run concurrently from their own bit cursors. Single `__syncwarp` at join, then existing Phase 4 (lanes 0..31 parallel vert/tri emit) runs unchanged.

## Encoder changes

### Residual stream reorder (the only bitstream-format change)

Current per-meshlet stream:
```
[hdr | conn | anchor + 6 tag headers | walk_residuals interleaved by kind]
```

New per-meshlet stream:
```
[hdr | conn | anchor + 6 tag headers | KIND0 residuals... | KIND1 residuals...]
```

Within each kind region, triples remain (rx, ry, rz) interleaved per item using that kind's (tag, k).

Rationale: lane 1 must decode without knowing the per-item kind (that info comes from lane 0's conn decode). Grouping by kind lets lane 1 decode (n_kind0 − 1) delta triples then n_kind1 para triples sequentially without conditionals.

### Side-table additions (no in-bitstream cost)

Two new per-meshlet arrays appended to the blob (alongside existing `ml_off_bits[]`, etc.):

- `ml_resid_off_bits[i]` (u64) — bit offset within `buf` where this meshlet's anchor + tag headers + residuals begin.
- `ml_n_kind0[i]` (u32) — count of kind=0 (delta) walk entries.

Cost: 12 B/meshlet. Dragon 33K meshlets ≈ 400 KB device-side metadata. Negligible vs ~14 MB total compressed payload (~3%). Not counted in BPV (consistent with current side tables).

### Blob format bump

`VERSION_DUP` 2 → 3. Reader rejects version < 3 with clear error. Old `*.dup.blob` files re-generated from cache.

## Kernel restructure

### Two BR cursors, two staging buffers

```c
__shared__ i16 resid_s_pool[WARPS_PER_BLOCK][MAX_V][3];
__shared__ i32 anchor_s_pool[WARPS_PER_BLOCK][3];  // L1 publishes, L0 reads
```

### Lane 0 (conn + predictor)

```c
if (lane == 0) {
    BR r;
    br_init(r, buf, ml_off_bits[bi]);
    // existing conn decode: read n_local/n_tris/n_strips header,
    // strip loop produces walk_pack[], walk_kind_bm[], tris_s[]
    // ...

    __syncwarp(0x3);  // wait for L1 to finish residuals + anchor

    // predictor-only loop (no Rice reads)
    i32 ax = anchor_s[0], ay = anchor_s[1], az = anchor_s[2];
    i32 prev_x = 0, prev_y = 0, prev_z = 0;
    bool first = true;
    u32 cur_d = 0, cur_p = 0;
    for (u32 i = 0; i < walk_count; ++i) {
        u32 pk = walk_pack[i];
        u32 v = pk & 0xFFu;
        u32 kind = (walk_kind_bm[i>>5] >> (i&31)) & 1u;
        i32 cx, cy, cz;
        if (kind == 0u) {
            if (first) { cx = ax; cy = ay; cz = az; first = false; }
            else {
                i16* r3 = &resid_s[cur_d * 3];
                cx = prev_x + zz_to_signed_i16(r3[0]);
                cy = prev_y + zz_to_signed_i16(r3[1]);
                cz = prev_z + zz_to_signed_i16(r3[2]);
            }
            ++cur_d;
        } else {
            // unpack a,b,c indices from pk
            // load codes_s[a/b/c]
            // pred = a + b - c
            i16* r3 = &resid_s[(n_kind0 + cur_p) * 3];
            cx = ax_pred + zz_to_signed_i16(r3[0]);
            // ...
            ++cur_p;
        }
        codes_s[v] = pack_i16x3(cx, cy, cz);
        prev_x = cx; prev_y = cy; prev_z = cz;
    }
}
```

### Lane 1 (residual decoder)

```c
else if (lane == 1) {
    BR r1;
    br_init(r1, buf, ml_resid_off_bits[bi]);

    i32 ax = (i32)br_read_bits(r1, g_bits0);
    i32 ay = (i32)br_read_bits(r1, g_bits1);
    i32 az = (i32)br_read_bits(r1, g_bits2);
    anchor_s[0] = ax; anchor_s[1] = ay; anchor_s[2] = az;

    u32 d_tag[3], d_k[3], p_tag[3], p_k[3];
    for (int d = 0; d < 3; ++d) {
        d_tag[d] = br_read_bits(r1, 8);
        d_k[d]   = br_read_bits(r1, 8);
    }
    for (int d = 0; d < 3; ++d) {
        p_tag[d] = br_read_bits(r1, 8);
        p_k[d]   = br_read_bits(r1, 8);
    }

    u32 n_k0 = ml_n_kind0[bi];
    // decode (n_k0 - 1) delta triples (first kind0 item uses anchor, no resid)
    for (u32 i = 0; i + 1 < n_k0; ++i) {
        resid_s[i*3 + 0] = decode_one(r1, d_tag[0], d_k[0]);
        resid_s[i*3 + 1] = decode_one(r1, d_tag[1], d_k[1]);
        resid_s[i*3 + 2] = decode_one(r1, d_tag[2], d_k[2]);
    }
    // decode n_k1 para triples
    u32 n_k1 = n_local - n_k0;
    for (u32 i = 0; i < n_k1; ++i) {
        resid_s[(n_k0 + i)*3 + 0] = decode_one(r1, p_tag[0], p_k[0]);
        // ...
    }
}
```

Note: `resid_s` slot index 0..n_k0-2 holds delta residuals; slot n_k0..n_k0+n_k1-1 holds para residuals. Lane 0's predictor uses `cur_d` and `cur_p` cursors that map to these regions.

### Decode helper signature

```c
__device__ __forceinline__ i16 decode_one(BR& r, u32 tag, u32 k) {
    u32 u = (tag == 1u) ? br_read_rice(r, k) : br_read_exp_golomb(r, k);
    return (i16)zz_to_signed(u);
}
```

i16 fits since q12-bbox residuals bounded by ±2048 = 12 bits zigzagged → 13 bits unsigned, sign-extended into i16 OK.

### Sync model

Single `__syncwarp()` at join. Lane 0 stalls if it finishes conn before lane 1 finishes residuals (expected case). No fine-grain producer-consumer.

Lanes 2..31 idle during phase 1 — wait at the syncwarp. They are recruited for phase 4 parallel emit (unchanged).

## SMEM budget

| Buffer | Per warp | × 4 warps |
|---|---|---|
| codes_s (existing) | 2048 B | 8192 B |
| tris_s (existing) | 768 B | 3072 B |
| walk_pack (existing) | 1024 B | 4096 B |
| walk_kind_bm (existing) | 32 B | 128 B |
| bi_smem (existing) | 4 B | 16 B |
| **resid_s (new)** | 1536 B | 6144 B |
| **anchor_s (new)** | 12 B | 48 B |
| **Total** | 5424 B | **21696 B** |

Block-limit SMEM jumps from 6 → 4 (CC 8.6 carveout 100 KB). Occupancy 50% → 33%.

### Occupancy mitigation

Pivot to **2 warps/block** if 33% hurts measurably:
- SMEM/block halves to 10848 B
- Block limit SMEM returns to 8
- Achieved occupancy back to ~50%
- Same total parallelism (more blocks, fewer warps each)

Fallback compactions if 2-warp variant still under target:
- Pack residual triple as `i32` (3 × 10-bit zigzag fits) — halves resid_s.
- Drop walk_pack[] kind=1 fields, store only v; recompute a/b/c on predictor pass — saves 768 B/warp but more decode work.

Defer these until measured.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Encoder reorder breaks bit-exactness | Add CPU reference decoder that mirrors new lane-1 logic; `verify_dup_cuda.py` covers parity end-to-end |
| n_kind0 derivation mismatches kernel count | Encoder counts during walk construction; unit-test on bunny + Monkey |
| L1 finishes long before L0 (small meshes) | Both lanes idle inside their loops; no extra cost. Already serial today. |
| Occupancy 33% → throughput regression on small meshes | Switch to 2 warps/block as default; revisit if data shows otherwise |
| SMEM bank conflicts on resid_s | i16 stride-3 access pattern — measure with ncu after impl; pad to stride-4 if needed |

## Expected gain (Dragon, back-of-envelope)

- T_serial_now ≈ T_conn(150) + T_anchor_tags(9) + T_resid+predict(650) ≈ 810 BR-ops-equivalent
- T_split = max(T_conn + T_predict_only, T_resid_decode) ≈ max(150 + 200, 600) ≈ 600
- Speedup ≈ 810 / 600 ≈ **1.35×**
- Dragon: 1798 → ~2430 mtps (1.12× DGF)

Smaller meshes (fandisk, s-bunny) get smaller wins — L1's residual block is too small to fully hide L0's conn decode. Expect ~1.1-1.2× there.

## Implementation order

1. **Encoder**: add residual reorder + n_kind0 count + resid_off_bits computation in `encoder/paradelta_v5_dup.py`. Bump VERSION_DUP to 3.
2. **Blob writer/reader**: extend `scripts/dump_stride_dup_blob.py` to emit `ml_resid_off_bits[]` + `ml_n_kind0[]`. Extend `bench_cpp/stride_dup_decode_bench.cu` reader accordingly.
3. **CPU reference parity**: update `verify_dup_cuda.py` or its Python decoder shim to consume v3 format. Verify bunny + Monkey tris exact, verts close.
4. **Kernel**: implement two-lane split inside existing kernel; add resid_s + anchor_s SMEM.
5. **Bench**: full 8-mesh sweep, log to `bench_dup_opt_v4.csv`.
6. **Occupancy check**: if 4-warp regresses, rebuild with `WARPS_PER_BLOCK=2`.
7. **Compare to DGF + memory update**.

## Success criteria

- Tris bit-exact vs CPU reference on all 8 benchmark meshes.
- Verts within 1e-4 vs CPU reference (existing tolerance).
- Dragon mtps ≥ 2200 (beats DGF). Stretch: ≥ 2500.
- No regression worse than 5% on any mesh.

## Out of scope

- K=6 axis-level substreams (more parallelism, more BPV overhead) — kept as follow-up if K=2 doesn't reach target.
- Ring-buffer producer-consumer between L0 and L1 — kept as follow-up if SMEM budget is binding.
- Warp-coop bulk unary scan within a single residual stream — separate idea, deferred.
- Changing connectivity coding (CLERS, EdgeBreaker) — outside this work.
