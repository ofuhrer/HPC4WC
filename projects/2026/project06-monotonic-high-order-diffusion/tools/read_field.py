"""Reads and compares stencil2d binary field files.

A field file is written by `write_field_to_file` in `m_utils.F90`: six
int32 header values followed by the field data. Fields written with
different halo widths compare equal on their interiors, which is the
invariant behind the `--num_halo` validation check.

Usage:
    read_field.py FIELD.dat                  Print header and statistics.
    read_field.py A.dat B.dat                Compare interiors exactly.
    read_field.py A.dat B.dat --rtol 1e-5    Compare within a tolerance.

Authors:
    Stefanie Boersig <stefanie.boersig@env.ethz.ch>
    Boaz Ko <boazko@student.ethz.ch>
    Ben Bullinger <ben.bullinger@inf.ethz.ch>
"""

import sys

import numpy as np

_HEADER_VALUES = 6


def read_field(filename, crop_halo=True):
    """Reads a stencil2d field file.

    Args:
        filename: Path to a binary field file.
        crop_halo: Whether to strip the halo and return the interior only.

    Returns:
        A tuple of the field as a NumPy array and the halo width.
    """
    rank, nbits, num_halo, nx, ny, nz = np.fromfile(
        filename, dtype=np.int32, count=_HEADER_VALUES)
    offset = (3 + rank) * 32 // nbits
    dtype = np.float32 if nbits == 32 else np.float64
    data = np.fromfile(filename, dtype=dtype, count=nz * ny * nx + offset)
    shape = (nz, ny, nx) if rank == 3 else (ny, nx)
    field = np.reshape(data[offset:], shape)
    if crop_halo and num_halo > 0:
        field = field[..., num_halo:-num_halo, num_halo:-num_halo]
    return field, int(num_halo)


def _flag_value(flag, default=0.0):
    """Returns the value following `flag` on the command line."""
    if flag not in sys.argv:
        return default
    return float(sys.argv[sys.argv.index(flag) + 1])


def _describe(filename):
    """Prints the interior shape and statistics of one field."""
    field, num_halo = read_field(filename)
    print(f"{filename}: interior shape {field.shape} "
          f"(halo {num_halo} cropped), min {field.min():.6g}, "
          f"max {field.max():.6g}, mean {field.mean():.6g}")


def _compare(first, second, rtol, atol):
    """Compares two field interiors.

    Args:
        first: Path to the reference field file.
        second: Path to the field file under test.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Returns:
        True if the interiors match within the tolerances.
    """
    a, halo_a = read_field(first)
    b, halo_b = read_field(second)
    if a.shape != b.shape:
        print(f"FAIL: interior shapes differ: {a.shape} (halo {halo_a}) "
              f"vs {b.shape} (halo {halo_b})")
        return False
    max_diff = np.abs(a - b).max()
    matched = np.allclose(a, b, rtol=rtol, atol=atol)
    verdict = "OK" if matched else "FAIL"
    print(f"{verdict}: interiors {'match' if matched else 'differ'} "
          f"(halo {halo_a} vs {halo_b}), max |diff| = {max_diff:.3g}")
    return matched


def main():
    """Describes one field, or compares two."""
    paths = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if len(paths) == 1:
        _describe(paths[0])
        return
    matched = _compare(paths[0], paths[1],
                       _flag_value("--rtol"), _flag_value("--atol"))
    sys.exit(0 if matched else 1)


if __name__ == "__main__":
    main()
