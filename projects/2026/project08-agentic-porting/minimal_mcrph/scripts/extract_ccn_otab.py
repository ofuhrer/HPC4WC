"""One-time extraction of get_otab's hardcoded lookup table from Fortran source.

Not imported by the package at runtime -- run this once (`uv run python
minimal_mcrph/scripts/extract_ccn_otab.py`, from anywhere) to (re)produce
minimal_mcrph/data/ccn_otab.npz, which the package loads instead of depending on
the Fortran source being present at runtime.

Mechanical regex extraction, not manual retyping, specifically to avoid
transcription errors across ~450 hardcoded calibration numbers (the failure
mode already hit twice this session with much smaller data sets -- see
port_log.md, 2026-07-23 entries).
"""

import re
import sys
from pathlib import Path

import numpy as np

_VARIANT = Path(__file__).resolve().parents[1]
SRC = _VARIANT / "fortran" / "mo_2mom_mcrph_processes.f90"
OUT = _VARIANT / "data" / "ccn_otab.npz"


def fortran_num(tok: str) -> float:
    tok = tok.strip()
    # Fortran double literals use d/D for the exponent instead of e/E
    tok = tok.replace("d", "e").replace("D", "E")
    return float(tok)


def parse_vector(text: str, name: str) -> np.ndarray:
    m = re.search(rf"otab%{name}\s*=\s*\(/([^/]+)/\)", text)
    if not m:
        raise ValueError(f"vector {name} not found")
    return np.array([fortran_num(t) for t in m.group(1).split(",")])


def parse_ltable(text: str, n1: int, n2: int, n3: int, n4: int) -> np.ndarray:
    # otab%n3 = n_ncn + 1 = 9, otab%n4 = n_wcb + 1 = 5 (see get_otab)
    table = np.zeros((n1, n2, n3, n4))
    pattern = re.compile(r"otab%ltable\((\d+),(\d+),2:otab%n3,(\d+)\)\s*=\s*\(/([^/]+)/\)")
    count = 0
    for m in pattern.finditer(text):
        i, j, l = int(m.group(1)), int(m.group(2)), int(m.group(3))
        values = [fortran_num(t) for t in m.group(4).split(",")]
        if len(values) != n3 - 1:
            raise ValueError(f"expected {n3 - 1} values at ({i},{j},:,{l}), got {len(values)}")
        table[i - 1, j - 1, 1:n3, l - 1] = values
        count += 1
    expected = n1 * n2 * (n4 - 1)  # l=1 (wcb=0) is the all-zero row, not matched here
    if count != expected:
        raise ValueError(f"parsed {count} ltable rows, expected {expected}")
    # l=1 (wcb=0.0) and k=0 (n_cn=0.0) rows are explicitly zero in the Fortran
    # (otab%ltable(:,:,:,1) = 0.0d0 and otab%ltable(:,:,1,:) = 0.0d0) --
    # table is already zero-initialized, so nothing more to do there.
    return table


def main() -> None:
    full_text = SRC.read_text()

    # Restrict to get_otab's body to avoid accidentally matching unrelated text
    start = full_text.index("SUBROUTINE get_otab")
    end = full_text.index("END SUBROUTINE get_otab")
    body = full_text[start:end]

    x1 = parse_vector(body, "x1")
    x2 = parse_vector(body, "x2")
    x3 = parse_vector(body, "x3")
    x4 = parse_vector(body, "x4")
    assert x1.shape == (3,), x1
    assert x2.shape == (5,), x2
    assert x3.shape == (9,), x3
    assert x4.shape == (5,), x4

    ltable = parse_ltable(body, n1=3, n2=5, n3=9, n4=5)

    # Independent spot checks against the literal source text (re-read by eye,
    # not re-derived from this same regex):
    #   otab%ltable(1,1,2:9,2) starts with 42.2d06, ends with 397.5d06 (extrapolated tail)
    #   otab%ltable(3,5,2:9,5) ends with 2464.7d06
    checks = [
        (ltable[0, 0, 1, 1], 42.2e6),
        (ltable[0, 0, 8, 1], 397.5e6),
        (ltable[2, 4, 8, 4], 2464.7e6),
    ]
    for got, expected in checks:
        ok = abs(got - expected) < 1e-6 * abs(expected)
        print(f"check: got={got} expected={expected} ok={ok}")
        if not ok:
            sys.exit(1)

    expected_nonzero = x1.size * x2.size * (x3.size - 1) * (x4.size - 1)
    if np.count_nonzero(ltable) != expected_nonzero:
        raise ValueError(
            f"nonzero count {np.count_nonzero(ltable)} != expected {expected_nonzero} "
            "(n_cn=0 and wcb=0 slices should be all-zero, everything else populated)"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, x1=x1, x2=x2, x3=x3, x4=x4, ltable=ltable)
    print(f"saved {OUT}, ltable shape: {ltable.shape}, nonzero count: {np.count_nonzero(ltable)}")


if __name__ == "__main__":
    main()
