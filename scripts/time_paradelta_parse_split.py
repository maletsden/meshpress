"""Break parse into: bit-decode vs greedy_order. Skip greedy_order to isolate."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from utils.bit_codec import BitReader
from collections import deque
from utils.paradelta_cache import load_or_prepare
from encoder.paradelta_codec import (
    encode_from_prepared, decode_paradelta_to_struct,
    MAGIC, VERSION, PREDICTOR_LIN5, PREDICTOR_PLAIN,
    _idx_bits_for, REUSE_BUF_SIZE, _read_vert,
)


def parse_no_greedy(data: bytes) -> dict:
    r = BitReader(data)
    assert r.read_fixed(32) == MAGIC
    assert r.read_fixed(8) == VERSION
    [r.read_f32() for _ in range(3)]                    # center
    r.read_f32()                                        # scale
    r.read_f32()                                        # per_coord_err
    [r.read_f32() for _ in range(3)]                    # g_min
    [r.read_f32() for _ in range(3)]                    # g_range
    g_bits = [r.read_fixed(8) for _ in range(3)]
    n_v = r.read_fixed(32); n_t = r.read_fixed(32)
    n_boundary = r.read_fixed(32); n_meshlets = r.read_fixed(32)
    pmode = r.read_fixed(8)
    if pmode == PREDICTOR_LIN5:
        [r.read_f32() for _ in range(8)]

    # Boundary
    for d in range(3):
        if n_boundary == 0: continue
        r.read_fixed(int(g_bits[d]))
        if n_boundary > 1:
            k = r.read_fixed(8)
            for _ in range(n_boundary - 1): r.read_rice(k)

    pad = (-r.bit_pos()) & 7
    if pad: r.read_bits(pad)
    for _ in range(n_meshlets): r.read_fixed(32)

    total_tris = 0
    for ml in range(n_meshlets):
        n_bnd = r.read_fixed(16); n_int = r.read_fixed(16)
        n_tris_m = r.read_fixed(16); n_strips = r.read_fixed(16)
        n_local = n_bnd + n_int
        if n_bnd > 0:
            r.read_fixed(32)
            if n_bnd > 1:
                k = r.read_fixed(8)
                for _ in range(n_bnd - 1): r.read_rice(k)
        idx_bits = _idx_bits_for(n_local)
        reuse_fifo: deque = deque(maxlen=REUSE_BUF_SIZE)
        for _s in range(n_strips):
            strip_len = r.read_fixed(16)
            _read_vert(r, reuse_fifo, idx_bits)
            _read_vert(r, reuse_fifo, idx_bits)
            _read_vert(r, reuse_fifo, idx_bits)
            for _ in range(strip_len - 1):
                r.read_bits(1)
                _read_vert(r, reuse_fifo, idx_bits)
        total_tris += n_tris_m
        if n_int > 0:
            for d in range(3):
                tag = r.read_fixed(8)
                if tag == 0:
                    r.read_fixed(16); bw = r.read_fixed(8)
                    for _ in range(n_int): r.read_fixed(bw)
                elif tag == 1:
                    k = r.read_fixed(8)
                    for _ in range(n_int): r.read_rice(k)
                else:
                    k = r.read_fixed(8)
                    for _ in range(n_int): r.read_exp_golomb(k)
    return total_tris


for path in ["D:/meshpress/assets/fandisk.obj",
             "D:/meshpress/assets/stanford-bunny.obj",
             "D:/meshpress/assets/Monkey.obj"]:
    prep = load_or_prepare(path, max_verts=256, max_tris=256,
                           precision_error=0.0005,
                           gen_method="joint_learned",
                           strip_method="multiseed", verbose=False)
    data = encode_from_prepared(prep, predictor="linear5", verbose=False)
    name = Path(path).name

    # Warmup
    parse_no_greedy(data)
    decode_paradelta_to_struct(data)

    t0 = time.perf_counter()
    for _ in range(3): parse_no_greedy(data)
    t_bitonly = (time.perf_counter() - t0) / 3

    t0 = time.perf_counter()
    for _ in range(3): decode_paradelta_to_struct(data)
    t_full = (time.perf_counter() - t0) / 3

    print(f"[{name:20s}]  bit_decode={t_bitonly*1000:7.1f} ms  "
          f"full(+greedy)={t_full*1000:7.1f} ms  "
          f"greedy={max(0,t_full-t_bitonly)*1000:7.1f} ms")