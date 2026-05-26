import numpy as np
import pytest
from pathlib import Path

from encoder import STRIDEEncoder
from encoder.paradelta_codec import prepare_paradelta_arrays
from encoder.paradelta_v5_dup import encode_dup
from reader.fast_obj import load_mesh_npy
from utils.types import Model


ASSET = Path("assets/stanford-bunny.obj")


def _skip_if_no_asset():
    if not ASSET.exists():
        pytest.skip(f"{ASSET} not present")


def test_stride_encoder_bit_exact_vs_raw_calls():
    _skip_if_no_asset()
    verts, tris = load_mesh_npy(str(ASSET))
    prep = prepare_paradelta_arrays(
        verts, tris,
        max_verts=256, max_tris=256,
        precision_error=0.0005,
        precision_mode="world",
        gen_method="joint_learned",
        strip_method="multiseed",
    )
    raw = encode_dup(prep, predictor="generalized", verbose=False)

    model = Model.from_arrays(verts, tris)
    out = STRIDEEncoder(
        max_verts=256, max_tris=256, precision_error=0.0005,
    ).encode(model)

    assert out.data == raw
    assert out.bits_per_vertex == pytest.approx(
        len(raw) * 8 / len(verts), rel=1e-12)
    assert out.bits_per_triangle == pytest.approx(
        len(raw) * 8 / len(tris), rel=1e-12)


def test_stride_encoder_from_obj_roundtrip_size():
    _skip_if_no_asset()
    model = Model.from_obj(str(ASSET))
    out = STRIDEEncoder(precision_error=0.0005).encode(model)
    assert 20.0 < out.bits_per_vertex < 60.0
    assert len(out.data) > 0