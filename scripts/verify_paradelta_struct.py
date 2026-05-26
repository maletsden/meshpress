"""Verify decode_paradelta_to_struct + numpy-reference reconstruction
produces identical verts/tris vs decode_paradelta.

The numpy reference uses the same logic the CUDA kernel will use:
  per meshlet (sequential):
    - copy boundary verts from bnd_pos_norm via l2g
    - walk order; for each entry apply predictor + residual
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    encode_from_prepared, decode_paradelta,
    decode_paradelta_to_struct,
    KIND_PARA, KIND_MID, KIND_ONE, KIND_NONE,
    PREDICTOR_LIN5, PREDICTOR_PLAIN,
)


def numpy_reference_decode(s: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mimic the CUDA kernel in pure numpy. Sequential per meshlet."""
    n_v = s["n_v"]
    n_t = s["n_t"]
    n_meshlets = s["n_meshlets"]
    bnd_pos = s["bnd_pos_norm"]
    delta = float(s["delta"])
    pmode = int(s["predictor_mode"])
    lin5_w3 = s["lin5_w3"].astype(np.float64)
    lin5_w5 = s["lin5_w5"].astype(np.float64)

    verts = np.zeros((n_v, 3), dtype=np.float64)
    verts[: bnd_pos.shape[0]] = bnd_pos.astype(np.float64)
    tris = np.zeros((n_t, 3), dtype=np.int64)
    tri_cursor = 0

    for m in range(n_meshlets):
        n_bnd = int(s["ml_n_bnd"][m])
        n_int = int(s["ml_n_int"][m])
        n_tris_m = int(s["ml_n_tris"][m])
        n_local = n_bnd + n_int

        l2g_s, l2g_e = s["ml_l2g_off"][m], s["ml_l2g_off"][m + 1]
        l2g = s["ml_l2g"][l2g_s:l2g_e]  # (n_local,)

        tri_s, tri_e = s["ml_tris_off"][m], s["ml_tris_off"][m + 1]
        local_tris = s["ml_tris"][tri_s:tri_e]  # (n_tris_m, 3)

        code_s, code_e = s["ml_codes_off"][m], s["ml_codes_off"][m + 1]
        codes = s["ml_codes"][code_s:code_e].astype(np.float64)  # (n_int, 3)

        order_s, order_e = s["ml_order_off"][m], s["ml_order_off"][m + 1]
        order = s["ml_order"][order_s:order_e]  # (n_int, 7)

        # Local recon buffer
        recon = np.zeros((n_local, 3), dtype=np.float64)
        for lid in range(n_bnd):
            recon[lid] = verts[l2g[lid]]
        fallback = recon[:n_bnd].mean(axis=0) if n_bnd > 0 \
            else np.zeros(3, dtype=np.float64)

        for i in range(n_int):
            v_local, kind = int(order[i, 0]), int(order[i, 1])
            a, b, c, d_ac, d_bc = (int(order[i, j]) for j in range(2, 7))

            if pmode == PREDICTOR_PLAIN or kind != KIND_PARA:
                # Touma-Gotsman / mid / one / none baseline
                if kind == KIND_PARA:
                    pred = recon[a] + recon[b] - recon[c]
                elif kind == KIND_MID:
                    pred = 0.5 * (recon[a] + recon[b])
                elif kind == KIND_ONE:
                    pred = recon[a]
                else:
                    pred = fallback
            else:
                # LIN5: full 5-tap when both apexes valid, else lin3
                if d_ac >= 0 and d_bc >= 0:
                    pred = (lin5_w5[0] * recon[a]
                            + lin5_w5[1] * recon[b]
                            + lin5_w5[2] * recon[c]
                            + lin5_w5[3] * recon[d_ac]
                            + lin5_w5[4] * recon[d_bc])
                else:
                    pred = (lin5_w3[0] * recon[a]
                            + lin5_w3[1] * recon[b]
                            + lin5_w3[2] * recon[c])

            rec = pred + codes[i] * delta
            recon[v_local] = rec
            verts[l2g[v_local]] = rec

        # Push tris with decoder global IDs
        tris[tri_cursor: tri_cursor + n_tris_m] = l2g[local_tris]
        tri_cursor += n_tris_m

    verts_world = (verts * float(s["scale"])
                   + s["center"].astype(np.float64))
    return verts_world.astype(np.float32), tris


def check(path: str) -> None:
    name = Path(path).name
    prep = load_or_prepare(
        path, max_verts=256, max_tris=256, precision_error=0.0005,
        gen_method="joint_learned", strip_method="multiseed", verbose=False)
    data = encode_from_prepared(prep, predictor="linear5", verbose=False)
    print(f"[{name}] encoded={len(data):,} B")

    t0 = time.time()
    v_ref, t_ref = decode_paradelta(data)
    t_ref_s = time.time() - t0
    print(f"  decode_paradelta:    verts={len(v_ref)} tris={len(t_ref)} "
          f"({t_ref_s:.2f}s)")

    t0 = time.time()
    s = decode_paradelta_to_struct(data)
    t_parse_s = time.time() - t0

    t0 = time.time()
    v_struct, t_struct = numpy_reference_decode(s)
    t_math_s = time.time() - t0
    print(f"  struct parse:        {t_parse_s:.2f}s")
    print(f"  numpy reference:     {t_math_s:.2f}s")

    assert v_ref.shape == v_struct.shape, \
        f"verts shape {v_ref.shape} vs {v_struct.shape}"
    assert t_ref.shape == t_struct.shape, \
        f"tris shape {t_ref.shape} vs {t_struct.shape}"

    max_v_err = float(np.abs(v_ref - v_struct).max())
    tri_eq = np.array_equal(t_ref, t_struct)
    print(f"  max_vert_diff={max_v_err:.6g}   tris_match={tri_eq}")
    if max_v_err > 1e-4 or not tri_eq:
        print("  FAIL — investigate")
        sys.exit(1)
    print("  OK")


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "D:/meshpress/assets/fandisk.obj",
        "D:/meshpress/assets/Monkey.obj",
    ]
    for p in paths:
        check(p)