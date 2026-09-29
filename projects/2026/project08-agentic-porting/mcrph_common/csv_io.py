"""Read/write the headed column CSVs the Fortran drivers speak.

One header row of comma-separated names, then one data row per model level. Every
variant uses this format -- only the set of columns differs (4 for satad_only, 23
for the two-moment schemes) -- so the primitives here are keyed by column name and
carry no notion of which fields a scheme actually has. Variant-specific assembly
(e.g. ``minimal_mcrph``'s ``ColumnState``) belongs in the variant.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

# 17 significant digits round-trips float64 exactly, so a column written here and
# read back by the Fortran is bit-identical -- file precision never limits a
# comparison between the two implementations.
FLOAT_FMT = "%.17e"


def read_columns(path: str | Path) -> dict[str, np.ndarray]:
    """Return ``{column_name: float64 array}`` for every column in ``path``."""
    with open(path) as fh:
        header = fh.readline().strip()
    names = [name.strip() for name in header.split(",")]
    # ndmin=2 rather than atleast_2d: a one-column file (hhl.csv) must come back
    # as (nrows, 1), not as the (1, nrows) that atleast_2d would give it.
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != len(names):
        raise ValueError(
            f"{path}: header lists {len(names)} columns but rows have {data.shape[1]}"
        )
    return {name: data[:, i].astype(np.float64) for i, name in enumerate(names)}


def read_column(path: str | Path, name: str) -> np.ndarray:
    """Return a single named column -- for one-column files such as ``hhl.csv``."""
    return read_columns(path)[name]


def write_columns(
    path: str | Path, columns: Mapping[str, np.ndarray], fmt: str = FLOAT_FMT
) -> None:
    """Write ``columns`` in mapping order, one header row then one row per level."""
    names = list(columns)
    arr = np.column_stack([np.asarray(columns[n], dtype=np.float64).ravel() for n in names])
    np.savetxt(path, arr, delimiter=",", header=",".join(names), comments="", fmt=fmt)
