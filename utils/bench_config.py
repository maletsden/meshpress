"""Shared precision config for STRIDE bench scripts.

When STRIDE_Q12BBOX=1 in the environment, all STRIDE encode calls run in
bbox-fractional q12 mode (12-bit uniform per-axis grid relative to bbox),
and bench outputs land in `bench_*_q12bbox.csv`. Otherwise the historical
world-units mode at precision_error=0.0005 is used.
"""
from __future__ import annotations

import os


def stride_precision() -> dict:
    """Kwargs for `load_or_prepare` / `prepare_paradelta`."""
    if os.environ.get("STRIDE_Q12BBOX") == "1":
        return dict(precision_error=1.0 / 4096.0,
                    precision_mode="bbox_frac")
    return dict(precision_error=0.0005,
                precision_mode="world")


def csv_suffix() -> str:
    return "_q12bbox" if os.environ.get("STRIDE_Q12BBOX") == "1" else ""


def mode_label() -> str:
    return "bbox_frac q12" if os.environ.get("STRIDE_Q12BBOX") == "1" \
                            else "world eps=5e-4"
