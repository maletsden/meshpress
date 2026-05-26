"""Minimal OBJ writer for round-trip tests."""

from __future__ import annotations

import numpy as np


def save_obj(path: str, verts: np.ndarray, tris: np.ndarray) -> None:
    verts = np.asarray(verts)
    tris = np.asarray(tris)
    with open(path, "w", encoding="ascii") as f:
        for v in verts:
            f.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        for t in tris:
            # OBJ is 1-indexed.
            f.write(f"f {int(t[0]) + 1} {int(t[1]) + 1} {int(t[2]) + 1}\n")