"""
Compare all encoders on all available meshes.
Includes both new encoders (AdaptivePatches, MeshletPredictor)
and existing encoders for reference.
"""

from reader import Reader
from encoder import (
    BaselineEncoder,
    PackedGTSQuantizator,
    PackedGTSEllipsoidFitter,
    AdaptivePatchesEncoder,
    MeshletPredictorEncoder,
    Packing,
)


def run_encoder(name, encoder, model):
    try:
        result = encoder.encode(model)
        size = len(result.data)
        return {
            "name": name,
            "size": size,
            "bpv": result.bits_per_vertex,
            "bpt": result.bits_per_triangle,
        }
    except Exception as e:
        return {"name": name, "size": 0, "bpv": 0, "bpt": 0, "error": str(e)}


def test_model(path):
    print(f"\n{'='*75}")
    print(f"Model: {path}")
    print(f"{'='*75}")

    model = Reader.read_from_file(path)
    if model is None:
        print(f"  Failed to load {path}")
        return

    n_v = len(model.vertices)
    n_t = len(model.triangles)
    print(f"  Vertices: {n_v:,}  Triangles: {n_t:,}")
    print(f"  Raw: {n_v * 12:,} B (vertices) + {n_t * 12:,} B (triangles)")

    encoders = []

    # Baseline (reference)
    encoders.append(("Baseline (f32)", BaselineEncoder()))

    # Existing encoders (skip for larger models - strip gen is O(n^2))
    if n_v <= 5000:
        encoders.append(("PackedGTS+Radix", PackedGTSQuantizator(
            pack_strip=Packing.RADIX_BINARY_TREE, verbose=False)))
        encoders.append(("GTS+Ellipsoid", PackedGTSEllipsoidFitter(
            vertex_quantization_error=0.0005)))

    # New encoders
    encoders.append(("C3 Patches K=2", AdaptivePatchesEncoder(
        K=2, precision_error=0.0005, verbose=True)))

    if n_v <= 600000:
        encoders.append(("Meshlet mt=128", MeshletPredictorEncoder(
            max_tris=128, precision_error=0.0005, verbose=True)))

    results = []
    for name, enc in encoders:
        print(f"\n  --- {name} ---")
        r = run_encoder(name, enc, model)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Size: {r['size']:,} B  |  BPV: {r['bpv']:.2f}  |  BPT: {r['bpt']:.2f}")
        results.append(r)

    # Summary table
    baseline_size = results[0]["size"] if results else 1
    print(f"\n  {'Encoder':<22} {'Size':>10} {'BPV':>8} {'BPT':>8} {'Ratio':>7}")
    print(f"  {'-'*57}")
    for r in results:
        if r["size"] > 0:
            ratio = baseline_size / r["size"]
            print(f"  {r['name']:<22} {r['size']:>10,} {r['bpv']:>8.2f} "
                  f"{r['bpt']:>8.2f} {ratio:>6.2f}x")


if __name__ == "__main__":
    models = [
        "assets/bunny.obj",
        "assets/torus.obj",
        "assets/stanford-bunny.obj",
        "assets/Monkey.obj",
        "assets/tank.obj",
    ]

    for path in models:
        try:
            test_model(path)
        except Exception as e:
            import traceback
            print(f"\nError with {path}:")
            traceback.print_exc()