"""STRIDE encoder (v5-dup variant).

Thin wrapper of `prepare_paradelta_arrays` + `encode_dup` implementing the
project Encoder interface. Matches the encoder used in the Visual Computer
paper (Table 6).

The default `predictor="generalized"` activates the per-mesh integer
rational linear predictor (IRLP) described in paper §3.5.
"""
from __future__ import annotations

from encoder.encoder import Encoder
from encoder.paradelta_codec import prepare_paradelta_arrays
from encoder.paradelta_v5_dup import encode_dup
from utils.types import Model, CompressedModel


class STRIDEEncoder(Encoder):
    def __init__(self, *,
                 max_verts: int = 256,
                 max_tris: int = 256,
                 precision_error: float = 0.0005,
                 precision_mode: str = "world",
                 gen_method: str = "joint_learned",
                 strip_method: str = "multiseed",
                 predictor: str = "generalized",
                 verbose: bool = False):
        self.max_verts = max_verts
        self.max_tris = max_tris
        self.precision_error = precision_error
        self.precision_mode = precision_mode
        self.gen_method = gen_method
        self.strip_method = strip_method
        self.predictor = predictor
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        prep = prepare_paradelta_arrays(
            model.vertices_np, model.triangles_np,
            max_verts=self.max_verts, max_tris=self.max_tris,
            precision_error=self.precision_error,
            precision_mode=self.precision_mode,
            gen_method=self.gen_method,
            strip_method=self.strip_method,
        )
        data = encode_dup(prep, verbose=self.verbose,
                          predictor=self.predictor)
        n_v = int(prep["n_v"])
        n_t = int(prep["n_t"])
        return CompressedModel(
            data=data,
            bits_per_vertex=len(data) * 8 / n_v,
            bits_per_triangle=len(data) * 8 / n_t,
        )