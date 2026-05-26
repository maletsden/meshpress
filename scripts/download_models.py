"""Fetch Stanford 3D Scanning Repo models, convert to OBJ for benchmarks.

Usage:
    python scripts/download_models.py            # default set (no thai/asian-dragon)
    python scripts/download_models.py --big      # add asian_dragon + thai_statue
    python scripts/download_models.py --only armadillo dragon
"""

import argparse
import gzip
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DL = ASSETS / "_dl"
ASSETS.mkdir(parents=True, exist_ok=True)
DL.mkdir(parents=True, exist_ok=True)

STANFORD = "http://graphics.stanford.edu"
COMMON3D = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data"

MODELS = {
    "fandisk": {
        "url": f"{COMMON3D}/fandisk.obj",
        "kind": "obj",
        "out": "fandisk.obj",
        "expect_verts": 6475,
    },
    "horse": {
        "url": f"{COMMON3D}/horse.obj",
        "kind": "obj",
        "out": "horse.obj",
        "expect_verts": 48485,
    },
    "armadillo": {
        "url": f"{STANFORD}/pub/3Dscanrep/armadillo/Armadillo.ply.gz",
        "kind": "ply.gz",
        "out": "armadillo.obj",
        "expect_verts": 172974,
    },
    "dragon": {
        "url": f"{STANFORD}/pub/3Dscanrep/dragon/dragon_recon.tar.gz",
        "kind": "tar.gz",
        "inner": "dragon_recon/dragon_vrip.ply",
        "out": "dragon.obj",
        "expect_verts": 437645,
    },
    "happy_buddha": {
        "url": f"{STANFORD}/pub/3Dscanrep/happy/happy_recon.tar.gz",
        "kind": "tar.gz",
        "inner": "happy_recon/happy_vrip.ply",
        "out": "happy_buddha.obj",
        "expect_verts": 543652,
    },
    "lucy": {
        "url": f"{STANFORD}/data/3Dscanrep/lucy.tar.gz",
        "kind": "tar.gz",
        "inner": "lucy.ply",
        "out": "lucy.obj",
        "expect_verts": 14027872,
    },
    "asian_dragon": {
        "url": f"{STANFORD}/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz",
        "kind": "ply.gz",
        "out": "asian_dragon.obj",
        "expect_verts": 3609455,
    },
    "thai_statue": {
        "url": f"{STANFORD}/data/3Dscanrep/xyzrgb/xyzrgb_statuette.ply.gz",
        "kind": "ply.gz",
        "out": "thai_statue.obj",
        "expect_verts": 4999996,
    },
    # --- STRIDE paper §5.1 test corpus ---
    "stanford-bunny": {
        "url": f"{STANFORD}/pub/3Dscanrep/bunny/bunny.tar.gz",
        "kind": "tar.gz",
        "inner": "bunny/reconstruction/bun_zipper.ply",
        "out": "stanford-bunny.obj",
        "expect_verts": 35947,
    },
    "xyzrgb_dragon": {
        "url": f"{STANFORD}/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz",
        "kind": "ply.gz",
        "out": "xyzrgb_dragon.obj",
        "expect_verts": 3609600,
    },
}

# Paper §5.1 test corpus. Three meshes (Monkey, crab, tank) are not
# auto-fetchable from public scientific repositories and must be supplied
# locally — see docs/obtaining_meshes.md for sourcing notes.
PAPER_CORPUS = [
    "fandisk", "stanford-bunny", "horse",
    "happy_buddha", "xyzrgb_dragon",
]

DEFAULT = ["fandisk", "horse", "armadillo", "dragon", "happy_buddha", "lucy"]
BIG_EXTRA = ["asian_dragon", "thai_statue"]


def _dl(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  [skip] {dst.name} exists ({dst.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  [get] {url}")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    print(f"  [ok]  {dst.name}  {dst.stat().st_size/1e6:.1f} MB  {time.time()-t0:.1f}s")


def _extract_ply_gz(src: Path, ply_out: Path) -> None:
    if ply_out.exists():
        return
    print(f"  [gunzip] {src.name} -> {ply_out.name}")
    with gzip.open(src, "rb") as fi, open(ply_out, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=1 << 20)


def _extract_tar_gz(src: Path, member: str, ply_out: Path) -> None:
    if ply_out.exists():
        return
    print(f"  [untar] {src.name}::{member}")
    with tarfile.open(src, "r:gz") as tf:
        try:
            mem = tf.getmember(member)
        except KeyError:
            names = [n for n in tf.getnames() if n.endswith(".ply")]
            print(f"  [warn] expected {member}, found: {names[:5]}")
            mem = tf.getmember(names[0])
        with tf.extractfile(mem) as fi, open(ply_out, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=1 << 20)


def _ply_to_obj(ply_path: Path, obj_path: Path) -> tuple[int, int]:
    if obj_path.exists():
        m = trimesh.load(obj_path, process=False, force="mesh")
        return len(m.vertices), len(m.faces)
    print(f"  [conv] {ply_path.name} -> {obj_path.name}")
    t0 = time.time()
    m = trimesh.load(ply_path, process=False, force="mesh")
    m.export(obj_path)
    print(f"  [ok]   verts={len(m.vertices):,}  tris={len(m.faces):,}  "
          f"{time.time()-t0:.1f}s")
    return len(m.vertices), len(m.faces)


def fetch_one(name: str, spec: dict) -> None:
    print(f"\n--- {name} ---")
    url = spec["url"]
    fname = url.rsplit("/", 1)[1]
    src = DL / fname
    obj = ASSETS / spec["out"]
    if obj.exists():
        m = trimesh.load(obj, process=False, force="mesh")
        print(f"  [done] {obj.name}  verts={len(m.vertices):,}  "
              f"tris={len(m.faces):,}")
        return
    if spec["kind"] == "obj":
        _dl(url, obj)
        m = trimesh.load(obj, process=False, force="mesh")
        nv, nt = len(m.vertices), len(m.faces)
        print(f"  [done] {obj.name}  verts={nv:,}  tris={nt:,}")
        exp = spec.get("expect_verts")
        if exp and abs(nv - exp) / max(1, exp) > 0.05:
            print(f"  [warn] verts={nv:,} differs from expected {exp:,}")
        return
    _dl(url, src)
    if spec["kind"] == "ply.gz":
        ply = DL / (fname[:-3] if fname.endswith(".gz") else fname + ".ply")
        _extract_ply_gz(src, ply)
    elif spec["kind"] == "tar.gz":
        inner = spec["inner"]
        ply = DL / Path(inner).name
        _extract_tar_gz(src, inner, ply)
    else:
        raise ValueError(f"unknown kind: {spec['kind']}")
    nv, nt = _ply_to_obj(ply, obj)
    exp = spec.get("expect_verts")
    if exp and abs(nv - exp) / max(1, exp) > 0.05:
        print(f"  [warn] verts={nv:,} differs from expected {exp:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true",
                    help="include asian_dragon + thai_statue (~1 GB extra)")
    ap.add_argument("--paper", action="store_true",
                    help="fetch the STRIDE paper §5.1 fetchable corpus "
                         "(fandisk, stanford-bunny, horse, happy_buddha, "
                         "xyzrgb_dragon)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="fetch only listed models")
    ap.add_argument("--keep-dl", action="store_true",
                    help="keep raw downloads in assets/_dl/")
    args = ap.parse_args()

    if args.only:
        names = args.only
    elif args.paper:
        names = list(PAPER_CORPUS)
    else:
        names = list(DEFAULT)
        if args.big:
            names += BIG_EXTRA

    for n in names:
        if n not in MODELS:
            print(f"[skip] unknown model: {n}")
            continue
        try:
            fetch_one(n, MODELS[n])
        except Exception as e:
            print(f"  [FAIL] {n}: {e}")

    if not args.keep_dl:
        print("\n[cleanup] removing assets/_dl/")
        shutil.rmtree(DL, ignore_errors=True)


if __name__ == "__main__":
    main()