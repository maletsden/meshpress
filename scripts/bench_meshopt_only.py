"""Run only bench_meshopt across all meshes; append to bench_competitors.csv."""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bench_competitors import bench_meshopt, _read_mesh, ROOT

PATHS = [
    "assets/fandisk.obj", "assets/bunny.obj", "assets/eyeball.obj",
    "assets/horse.obj", "assets/stanford-bunny.obj",
    "assets/Monkey.obj", "assets/tank.obj",
]

rows = []
for p in PATHS:
    if not Path(p).exists():
        continue
    _, pts, faces = _read_mesh(p)
    n_v, n_t = len(pts), len(faces)
    print(f"[{Path(p).name}] n_v={n_v:,} n_t={n_t:,}")
    r = bench_meshopt(p, n_v, n_t)
    if r is None:
        print("  skip"); continue
    print(f"  {r['name']}: {r['size_b']:,} B  BPV={r['bpv']:.2f}  "
          f"dec={r['dec_us']:.1f} µs  {r['mtps']:.1f} M tris/s")
    rows.append({"mesh": Path(p).name, "n_v": n_v, "n_t": n_t, **r})

# Append to existing CSV
csv_path = ROOT / "bench_competitors.csv"
keys = ["mesh", "n_v", "n_t", "name", "size_b", "bpv",
        "max_err", "enc_ms", "dec_us", "mtps", "gpu", "note"]
with open(csv_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    for r in rows:
        w.writerow(r)
print(f"\nAppended {len(rows)} rows to {csv_path}")