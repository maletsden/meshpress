"""Pickle cache for `prepare_paradelta` output.

Keyed on (model_path mtime, gen_method, max_verts, max_tris, precision_error,
strip_method). Lets predictor-mode experiments skip the slow meshlet
generation + per-meshlet plan build.

Cache files live in `cache/paradelta/`.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time

from reader import Reader
from encoder.paradelta_codec import prepare_paradelta
from utils.mesh_clean import clean_mesh


CACHE_DIR = os.path.join("cache", "paradelta")


def _cache_key(model_path: str, **prep_args) -> str:
    mtime = os.path.getmtime(model_path)
    parts = [model_path, str(mtime)] + [
        f"{k}={prep_args[k]}" for k in sorted(prep_args)]
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    base = os.path.splitext(os.path.basename(model_path))[0]
    return f"{base}_{h}.pkl"


def load_or_prepare(model_path: str, *, max_verts: int = 256,
                    max_tris: int = 256, precision_error: float = 0.0005,
                    precision_mode: str = "world",
                    gen_method: str = "joint_learned",
                    strip_method: str = "multiseed",
                    clean: bool = True,
                    force: bool = False, verbose: bool = False) -> dict:
    """Return cached prep dict if available; else build, cache, return."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    prep_args = dict(
        max_verts=max_verts, max_tris=max_tris,
        precision_error=precision_error,
        precision_mode=precision_mode,
        gen_method=gen_method, strip_method=strip_method,
        clean=int(clean))
    key = _cache_key(model_path, **prep_args)
    cache_path = os.path.join(CACHE_DIR, key)
    if not force and os.path.exists(cache_path):
        if verbose:
            print(f"  [cache hit] {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    if verbose:
        print(f"  [cache miss] preparing {model_path} ...")
    t0 = time.time()
    model = Reader.read_from_file(model_path)
    if clean:
        model, stats = clean_mesh(model, verbose=verbose)
    prep_kwargs = {k: v for k, v in prep_args.items() if k != "clean"}
    prep = prepare_paradelta(model, **prep_kwargs)
    if verbose:
        print(f"  prepare: {time.time() - t0:.2f}s  "
              f"meshlets={prep['n_meshlets']}")
    with open(cache_path, "wb") as f:
        pickle.dump(prep, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"  [cache write] {cache_path}")
    return prep
