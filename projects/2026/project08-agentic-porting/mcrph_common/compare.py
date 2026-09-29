"""Elementwise comparison of a port's output against the Fortran reference."""

from __future__ import annotations

import numpy as np


def compare(name, got, ref, rtol, atol=0.0):
    """Assert ``got`` ~= ``ref`` elementwise via ``np.isclose``, raising a
    diagnostic listing the offending points on failure.

    Tolerances are explicit rather than defaulted: what counts as agreement is a
    per-variant, often per-field judgement, so the calling harness owns it.
    """
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    close = np.isclose(got, ref, rtol=rtol, atol=atol)
    if not np.all(close):
        bad = np.where(~close)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(got - ref) / np.abs(ref)
        raise AssertionError(
            f"{name}: {bad.size} of {got.size} values differ beyond "
            f"rtol={rtol}, atol={atol}\n"
            f"  indices : {bad.tolist()}\n"
            f"  got     : {got[bad].tolist()}\n"
            f"  ref     : {ref[bad].tolist()}\n"
            f"  abs err : {np.abs(got - ref)[bad].tolist()}\n"
            f"  rel err : {rel[bad].tolist()}"
        )
