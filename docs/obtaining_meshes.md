# Obtaining the test-mesh corpus

The STRIDE paper (§5.1) uses an eight-mesh corpus. Meshes are **not committed**
to the repository (they are large binaries); instead they are fetched or sourced
locally. Provenance and licenses are also summarised in
[`assets/README.md`](../assets/README.md).

## Auto-fetchable (5 of 8)

```bash
python scripts/download_models.py --paper
```

This downloads and converts to OBJ:

| Mesh | Source | License |
|------|--------|---------|
| `fandisk.obj`       | Common 3D Test Models (alecjacobson) | CC0 |
| `stanford-bunny.obj`| Stanford 3D Scanning Repository      | research/educational use |
| `horse.obj`         | Common 3D Test Models (alecjacobson) | CC0 |
| `happy_buddha.obj`  | Stanford 3D Scanning Repository      | research/educational use |
| `xyzrgb_dragon.obj` | Stanford 3D Scanning Repository (XYZ RGB) | research/educational use |

Files land in `assets/`. Vertex/triangle counts after the encoder's dedup
pre-pass match paper Table 2.

## Sourced separately (3 of 8)

These three are not redistributable from a single public scientific repository
and must be supplied locally under `assets/` with the exact filenames below.

### `Monkey.obj` — Blender "Suzanne"
- **Source:** Blender (built-in mesh, `Add → Mesh → Monkey`).
- **How to reproduce:** in Blender, add Suzanne, apply a Subdivision Surface
  modifier at **6 levels** (Catmull–Clark), then export as OBJ
  (`File → Export → Wavefront (.obj)`, "Selection Only", no normals/UVs needed).
  Target: 504,482 vertices / 1,007,616 triangles (paper Table 2).
- **License:** Blender's bundled assets are public domain / CC0.

### `crab.obj` — "Dark Finger Reef Crab"
- **Source:** Threedscans.com (https://threedscans.com), an open scan archive.
- **How to obtain:** download the "Dark Finger Reef Crab" model, convert to OBJ
  (e.g., with `trimesh` or MeshLab). Target: 1,079,516 vertices /
  2,141,596 triangles.
- **License:** CC0 (Threedscans releases all scans into the public domain).

### `tank.obj` — game asset
- **Source:** a third-party game asset (positions + indices only).
- **License:** **unverified.** It is included for benchmarking reproducibility
  only and is **not redistributed** in this repository. If you cannot obtain the
  identical asset, substitute any mesh of comparable size (≈ 1.79 M vertices /
  3.51 M triangles); STRIDE's behaviour tracks topology regularity, not the
  specific asset, so a same-scale substitute reproduces the qualitative result.
  Target: 1,790,492 vertices / 3,505,328 triangles.

## Notes

- All auto-fetched raw downloads are cached under `assets/_dl/` and removed after
  conversion unless `--keep-dl` is passed.
- If a Stanford mirror is temporarily unavailable, the same `.ply` files are
  widely mirrored; point `scripts/download_models.py` at any mirror by editing
  the `STANFORD` base URL.
