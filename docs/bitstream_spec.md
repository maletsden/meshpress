# STRIDE bitstream specification

Reference layout for the STRIDE per-meshlet bitstream emitted by
`encoder/paradelta_v5.py` and consumed by the fused CUDA kernel in
`utils/paradelta_v5_cuda.py`. Cross-references to paper sections refer to
[`docs/paper_visual_computer_v6.md`](paper_visual_computer_v6.md).

All multi-byte values are stored little-endian. The bitstream is logically
divided into a small **header**, a per-meshlet **side table**, and an opaque
**body** holding the per-meshlet blocks. The body is referenced by the side
table; meshlet blocks are not required to be contiguous in the body, but in
practice they are.

---

## 1. Global header

| Offset | Type     | Field            | Description |
|-------:|----------|------------------|-------------|
| 0      | `u32`    | `magic`          | `0x53544944` (`"STID"`). |
| 4      | `u16`    | `version`        | Format version (current: `5`). |
| 6      | `u16`    | `flags`          | Bit 0 = `irlp_mode` (1 if IRLP header present). Bits 1–15 reserved. |
| 8      | `u32`    | `n_meshlets`     | Total number of meshlets. |
| 12     | `u32`    | `n_v_dup`        | Total unique-and-replicated vertex count (decoded vertex buffer length). |
| 16     | `u32`    | `n_t`            | Total triangle count. |
| 20     | `u8`     | `b_quant`        | Per-axis quantization bit width (default 12). |
| 21     | 3× `u8`  | `_pad`           | Padding to align floats. |
| 24     | 3× `f32` | `bbox_min[3]`    | Global axis-aligned bounding box minimum. |
| 36     | 3× `f32` | `bbox_max[3]`    | Global bounding box maximum. |
| 48     | `u32`    | `body_off_bytes` | Offset (bytes) to the start of the body section. |
| 52     | `u32`    | `body_len_bytes` | Body section length in bytes. |
| 56     |          |                  | End of fixed header. |

If `flags.irlp_mode == 1`, a 21-byte IRLP weight block follows immediately:

```
struct IrlpWeights {
    int16 n[3][3];   // per-axis numerators (paper §3.5, equation (4))
    uint8 K[3];      // per-axis right-shift exponents
};
```

The IRLP block is otherwise omitted, and the decoder uses the canonical
`(n0, n1, n2, K) = (1, 1, -1, 0)` everywhere.

---

## 2. Per-meshlet side table

Following the (optional) IRLP block lies the side table: `n_meshlets` records
of 50 bytes each.

| Offset | Type   | Field            | Description |
|-------:|--------|------------------|-------------|
| 0      | `u32`  | `bit_off`        | Bit-offset into the body where this meshlet begins. |
| 4      | `u16`  | `n_local`        | Number of vertices in this meshlet (≤ 256). |
| 6      | `u16`  | `n_tris`         | Number of triangles in this meshlet (≤ 256). |
| 8      | `u16`  | `n_strips`       | Number of disjoint strips inside this meshlet. |
| 10     | `u32`  | `v_off`          | Offset into the decoded vertex buffer for this meshlet. |
| 14     | `u32`  | `t_off`          | Offset into the decoded index buffer for this meshlet. |
| 18     | `u32`  | `resid_off_bits` | Bit-offset from `bit_off` to the start of the residual block. |
| 22     | `u8`   | `n_kind0`        | Anchor + delta vertex count (paper §3.3). |
| 23     | `u8`   | `_pad`           | Padding. |
| 24     | 5× `u16` | `axis_offs[5]` | Offsets to the heads of the `y`, `z`, parallelogram-`x`, parallelogram-`y`, parallelogram-`z` substreams. The `x` delta substream begins immediately after the anchor + tag headers. |
| 34     | 16     | `_reserved`      | Reserved. |
| 50     |        |                  | End of record. |

---

## 3. Per-meshlet body block

Each meshlet occupies a contiguous bit-aligned region of the body starting at
`bit_off`. Reading order:

1. **Meshlet header** (48 bits): redundant copy of `n_local`, `n_tris`,
   `n_strips`. Provides resilience if the side table is corrupted; the
   decoder reads from the side table by default.

2. **Connectivity stream — AMD GTS packed-local-index layout**, per strip:
   - 16-bit strip length.
   - 3 full-width local identifiers for the root triangle.
   - For each subsequent triangle: 1-bit `L/R` flag + one local-identifier
     token for the strip-emit vertex.

   Local identifiers pass through a 16-entry move-to-front (MTF) reuse FIFO.
   A 1-bit flag distinguishes a 5-bit FIFO index from a raw
   `⌈log₂ n_local⌉`-bit identifier.

3. **Anchor and tag headers** (read after seeking to `resid_off_bits`):
   - 3 × `b_quant` bits: anchor integer triple (paper §3.3).
   - 6 × (8 + 8) bits: `(tag, k)` pairs for {delta, IRLP} × {`x`, `y`, `z`}.
     - `tag == 1`: Rice with parameter `k`.
     - `tag == 2`: Exp-Golomb of order `k`.

4. **Axis-separated residual substreams** (six total):
   - `delta-x`: `n_kind0 − 1` residuals beginning immediately after the tag
     headers.
   - `delta-y`, `delta-z`, `irlp-x`, `irlp-y`, `irlp-z`: each substream begins
     at the offset given by the corresponding entry in `axis_offs`.

   Signed residuals are zig-zag-mapped: `u = (r << 1) ^ (r >> 31)`. The
   decoder applies the inverse zig-zag after entropy decoding.

The meshlet block ends at the bit offset
`bit_off + resid_off_bits + (axis_offs[4] + irlp_z_length)`.

---

## 4. Decoder reading order

The decoder is implemented as a single fused CUDA kernel (paper §4). Per
meshlet, one warp:

1. Reads the side-table record on lane 0.
2. Walks the connectivity stream on lane 0, populating a per-meshlet
   local-index table and the strip-walk packing.
3. On lane 0, seeks to `bit_off + resid_off_bits` and parses the anchor and
   six `(tag, k)` pairs.
4. Lanes 0, 1, and 2 each initialise a bit cursor at the start of the
   `x`, `y`, `z` axes and decode the three delta substreams in lockstep.
5. The same three lanes proceed to the three IRLP substreams.
6. Lane 0 walks the strip order a second time, applying the previous-vertex
   delta and the IRLP predictor (paper equation (4)).
7. All 32 lanes co-operate on the final dequantization
   (`q → float32 × bbox_step + bbox_min`) and emit the vertex / index
   buffers into mesh-shader-visible memory.

The IRLP step is unconditional and applies regardless of whether `flags.irlp_mode`
is set — when the bit is clear, the `(n, K)` parameters default to the
canonical Touma-Gotsman tuple.

---

## 5. Crack-free property

Cross-meshlet vertex identity is structural: positions are quantized onto a
single global integer lattice (paper §3.2, equation (3)) *before*
partitioning. Two meshlets that share a vertex carry identical integer codes
in their per-meshlet vertex arrays, and the dequantization step uses a
single `(bbox_min, bbox_max, b_quant)` triplet for the entire mesh. There
is no per-meshlet floating-point scale that could drift across meshlet
boundaries.

The verifier `scripts/verify_stride_rework.py` re-encodes every test mesh
and asserts that all globally-shared vertices reconstruct to bit-identical
integer codes from every meshlet that contains them.
