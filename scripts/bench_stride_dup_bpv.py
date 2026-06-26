"""Recompute the STRIDE-dup compression-rate column of paper Table 3.

BPV = (len(encode_dup(prep, canonical)) + 46 * n_meshlets) * 8 / n_v_source.
The resident payload is the self-contained dup blob plus the 50-byte/meshlet
random-access side table; the blob already carries a 4-byte/meshlet offset
table, so 46 additional bytes/meshlet are charged — matching the
"BPV including all side tables" caption of Table 3. BPV is over the source
(post-dedup) vertex count, the same basis as the `_bpv` helper in
bench_competitors.py.

Competitor columns (Draco/meshopt/Corto/DGF) are partition-independent and are
not recomputed here. Output: bench_stride_dup_bpv.csv.

Usage:
    python scripts/bench_stride_dup_bpv.py
    python scripts/bench_stride_dup_bpv.py --only fandisk stanford-bunny
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# label -> assets filename, in paper Table 3 order.
CORPUS = [
    ("fandisk",        "fandisk.obj"),
    ("stanford-bunny", "stanford-bunny.obj"),
    ("horse",          "horse.obj"),
    ("Monkey",         "Monkey.obj"),
    ("Happy Buddha",   "happy_buddha.obj"),
    ("Crab",           "crab.obj"),
    ("tank",           "tank.obj"),
    ("xyz-dragon",     "xyzrgb_dragon.obj"),
]

# Published Table 3 STRIDE column, for the delta report.
PUBLISHED = {
    "fandisk": 45.68, "stanford-bunny": 42.53, "horse": 41.61, "Monkey": 33.35,
    "Happy Buddha": 43.25, "Crab": 42.16, "tank": 34.07, "xyz-dragon": 33.62,
}

PRECISION_ERROR = 1.0 / 4096.0  # 12-bit bbox-relative grid


def compute_bpv(obj_path: Path):
    from utils.paradelta_cache import load_or_prepare
    from encoder.paradelta_v5_dup import encode_dup
    prep = load_or_prepare(
        str(obj_path), max_verts=256, max_tris=256,
        precision_error=PRECISION_ERROR, precision_mode="bbox_frac",
    )
    data = encode_dup(prep, predictor="canonical")
    n_v = int(prep["n_v"])
    n_meshlets = int(prep["n_meshlets"])
    payload = len(data) + 46 * n_meshlets   # blob + side table (50 B - 4 B offset already in blob)
    bpv = payload * 8 / n_v
    return n_v, payload, bpv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    sel = CORPUS if not args.only else [c for c in CORPUS if c[0] in set(args.only)]

    rows = []
    print(f"{'mesh':<15} {'n_v':>9} {'bytes':>10} {'BPV':>7} {'pub':>7} {'delta':>7}")
    for label, fname in sel:
        obj = ROOT / "assets" / fname
        if not obj.exists():
            print(f"  [skip] {label}: assets/{fname} absent")
            continue
        n_v, nbytes, bpv = compute_bpv(obj)
        pub = PUBLISHED.get(label)
        delta = (bpv - pub) if pub is not None else None
        rows.append((label, n_v, nbytes, bpv, pub, delta))
        ds = f"{delta:+.2f}" if delta is not None else "   -"
        print(f"  {label:<13} {n_v:>9,} {nbytes:>10,} {bpv:>7.2f} "
              f"{(pub if pub else 0):>7.2f} {ds:>7}")

    out = ROOT / "bench_stride_dup_bpv.csv"
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["mesh", "n_v", "blob_bytes", "bpv", "published_bpv", "delta"])
        for r in rows:
            wr.writerow([r[0], r[1], r[2], f"{r[3]:.2f}",
                         "" if r[4] is None else f"{r[4]:.2f}",
                         "" if r[5] is None else f"{r[5]:.2f}"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
