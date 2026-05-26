"""Multi-angle PNG renders of a triangle mesh via matplotlib.

CPU only; quality is OK for debugging round-trip not for production
visuals. For nicer renders use Open3D or trimesh elsewhere.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render_multi_angle(verts: np.ndarray, tris: np.ndarray, out_path: str,
                       title: str = "", angles=((20, 30), (20, 120),
                                                 (20, 210), (60, 30)),
                       face_color="#9ec5fe", edge_color="#1a3a6e",
                       size_px: int = 480) -> None:
    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    tri_verts = verts[tris]  # (T, 3, 3)

    fig = plt.figure(
        figsize=(size_px * len(angles) / 100, size_px / 100), dpi=100)
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, len(angles), i + 1, projection="3d")
        coll = Poly3DCollection(
            tri_verts, facecolors=face_color, edgecolors=edge_color,
            linewidths=0.1, alpha=0.95)
        ax.add_collection3d(coll)
        mn = verts.min(axis=0)
        mx = verts.max(axis=0)
        c = 0.5 * (mn + mx)
        r = 0.5 * (mx - mn).max() * 1.05
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"elev={elev}° azim={azim}°", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)