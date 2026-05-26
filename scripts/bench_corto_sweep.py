"""One-shot: run Corto v12 over the 7 default benchmark meshes, dump CSV."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_competitors import bench_corto, _read_mesh  # noqa: E402

MESHES = [
    "assets/fandisk.obj",
    "assets/bunny.obj",
    "assets/eyeball.obj",
    "assets/horse.obj",
    "assets/stanford-bunny.obj",
    "assets/Monkey.obj",
    "assets/tank.obj",
]

rows = []
for p in MESHES:
    full = ROOT / p
    if not full.exists():
        print(f"missing: {p}")
        continue
    print(f"\n=== {p} ===")
    _, pts, faces = _read_mesh(str(full))
    n_v = len(pts); n_t = len(faces)
    print(f"  verts={n_v}, tris={n_t}")
    r = bench_corto(str(full), n_v, n_t, qb=12)
    if r is None:
        print("  corto failed")
        continue
    print(f"  bpv={r['bpv']:.2f}  enc={r['enc_ms']:.1f} ms  dec={r['dec_us']:.1f} us  mtps={r['mtps']}")
    rows.append({"mesh": p, "n_v": n_v, "n_t": n_t, **r})

out = ROOT / "bench_corto_sweep.csv"
with open(out, "w", newline="") as f:
    if rows:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
print(f"\nWritten: {out}")
