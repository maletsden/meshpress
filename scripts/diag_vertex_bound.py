"""Verify vertex-level L∞ bound on reconstructed positions.

Encodes mesh, decodes, compares decoded vertex to the integer-quantized
source vertex (the target the bound is defined against). Reports per-axis
max |decoded - quantized| in units of Δ/2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reader import Reader
from utils.mesh_clean import clean_mesh
from encoder.paradelta_codec import (
    prepare_paradelta, encode_from_prepared, decode_paradelta,
)


MESHES = [
    ("fandisk",        "assets/fandisk.obj"),
    ("stanford-bunny", "assets/stanford-bunny.obj"),
    ("horse",          "assets/horse.obj"),
    ("Monkey",         "assets/Monkey.obj"),
]


def main():
    for name, path in MESHES:
        m = Reader.read_from_file(path)
        m, _ = clean_mesh(m, verbose=False)
        verts = np.array([[v.x, v.y, v.z] for v in m.vertices], dtype=np.float64)
        # bbox-fractional q12 grid
        b = 12
        g_min = verts.min(0); g_max = verts.max(0)
        ext = g_max - g_min
        step = ext / (2**b - 1)
        # Integer-quantized target per source vertex
        q_target = np.round((verts - g_min) / step).astype(np.int64)
        # source positions on the lattice (dequantized integer codes)
        v_quant = g_min + q_target * step

        prep = prepare_paradelta(
            m, max_verts=256, max_tris=256,
            precision_error=1.0 / (2**b),
            precision_mode="bbox_frac",
            gen_method="joint_learned", strip_method="multiseed")
        blob = encode_from_prepared(prep, predictor="linear3", verbose=False)
        v_dec, t_dec = decode_paradelta(blob)
        # Map decoded vertex order back to source via nearest match on
        # quantized target (decoder may permute order).
        # Cheap: kd-tree by quantized integer codes
        from scipy.spatial import cKDTree
        q_dec = np.round((v_dec.astype(np.float64) - g_min) / step).astype(np.int64)
        # Match decoded → source by exact quantized code
        src_keys = {tuple(q): i for i, q in enumerate(q_target)}
        max_err_axis = np.zeros(3)
        unmatched = 0
        for qd, vd in zip(q_dec, v_dec):
            key = tuple(qd)
            if key not in src_keys:
                unmatched += 1
                continue
            i = src_keys[key]
            err = np.abs(vd.astype(np.float64) - v_quant[i])
            max_err_axis = np.maximum(max_err_axis, err)
        half_step = step / 2.0
        print(f"[{name}]")
        print(f"  extent             = {ext}")
        print(f"  step               = {step}  (Δ/2 = {half_step})")
        print(f"  max |dec-vq| axis  = {max_err_axis}")
        print(f"  ratio max/(Δ/2)    = {max_err_axis / half_step}")
        print(f"  unmatched verts    = {unmatched}")
        print()


if __name__ == "__main__":
    main()
