"""
Final benchmark: run all meshlet-based encoders on all models.
Produces comparison table for CLAUDE.md.
"""

from reader import Reader
from encoder import (
    BaselineEncoder,
    MeshletWaveletEB,
    MeshletWaveletAMD,
)
import time


def run_encoder(name, encoder, model):
    try:
        t0 = time.time()
        result = encoder.encode(model)
        elapsed = time.time() - t0
        return {
            "name": name, "size": len(result.data),
            "bpv": result.bits_per_vertex, "bpt": result.bits_per_triangle,
            "time": elapsed,
        }
    except Exception as e:
        return {"name": name, "size": 0, "bpv": 0, "bpt": 0, "time": 0, "error": str(e)}


def test_model(path, max_verts=256):
    model = Reader.read_from_file(path)
    if model is None:
        print(f"  Failed to load {path}")
        return []

    n_v = len(model.vertices)
    n_t = len(model.triangles)
    short_name = path.split('/')[-1].replace('.obj', '')

    raw_size = n_v * 12 + n_t * 12  # raw float32 verts + uint32 indices

    encoders = [
        ("Baseline", BaselineEncoder()),
        ("EB+Bez+Wav mv256", MeshletWaveletEB(
            max_verts=max_verts, precision_error=0.0005,
            use_bezier=True, verbose=True)),
        ("EB+Direct mv256", MeshletWaveletEB(
            max_verts=max_verts, precision_error=0.0005,
            use_bezier=False, verbose=True)),
        ("AMD+Bez+Wav mv256", MeshletWaveletAMD(
            max_verts=max_verts, precision_error=0.0005,
            use_bezier=True, verbose=True)),
        ("AMD+Direct mv256", MeshletWaveletAMD(
            max_verts=max_verts, precision_error=0.0005,
            use_bezier=False, verbose=True)),
    ]

    print(f"\n{'='*70}")
    print(f"{short_name}: {n_v:,} verts, {n_t:,} tris (raw: {raw_size:,} B)")
    print(f"{'='*70}")

    results = []
    for name, enc in encoders:
        print(f"\n  --- {name} ---")
        r = run_encoder(name, enc, model)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        results.append(r)

    # Print table
    baseline = results[0]["size"] if results[0]["size"] > 0 else raw_size
    print(f"\n  {'Encoder':<25} {'Size':>10} {'BPV':>8} {'BPT':>8} {'Ratio':>7} {'Time':>6}")
    print(f"  {'-'*64}")
    for r in results:
        if r["size"] > 0:
            ratio = baseline / r["size"]
            print(f"  {r['name']:<25} {r['size']:>10,} {r['bpv']:>8.2f} "
                  f"{r['bpt']:>8.2f} {ratio:>6.1f}x {r['time']:>5.1f}s")

    return [(path.split('/')[-1], r) for r in results]


if __name__ == "__main__":
    all_results = []

    models = [
        "assets/bunny.obj",
        "assets/torus.obj",
        "assets/stanford-bunny.obj",
        "assets/Monkey.obj",
    ]

    for path in models:
        try:
            results = test_model(path, max_verts=256)
            all_results.extend(results)
        except Exception as e:
            import traceback; traceback.print_exc()

    # Print final summary table (markdown-ready)
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON (Markdown table)")
    print(f"{'='*70}\n")
    print("| Model | Encoder | Size | BPV | Ratio |")
    print("|-------|---------|-----:|----:|------:|")
    for fname, r in all_results:
        if r["size"] > 0 and r["name"] != "Baseline":
            short = fname.replace('.obj', '')
            print(f"| {short} | {r['name']} | {r['size']:,} B | {r['bpv']:.2f} | "
                  f"{r.get('ratio', 0):.1f}x |")
