"""Checks that a stencil2d run did not create new extrema.

Diffusion must not push the field outside the range it started in, so the
output bounds must lie within the input bounds up to a single-precision
epsilon. With `--expect-violation` the check is inverted, which is how the
unlimited filter is confirmed to violate the bounds it is meant to break.

Usage:
    check_monotonic.py IN.dat OUT.dat [--expect-violation]

Exits 0 if the expectation holds, 1 otherwise.

Authors:
    Stefanie Boersig <stefanie.boersig@env.ethz.ch>
    Boaz Ko <boazko@student.ethz.ch>
    Ben Bullinger <ben.bullinger@inf.ethz.ch>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# pylint: disable=wrong-import-position
from read_field import read_field  # noqa: E402

_RELATIVE_EPSILON = 1e-6


def is_monotone(initial, final):
    """Reports whether the final field stays inside the initial bounds.

    Args:
        initial: Field before the run.
        final: Field after the run.

    Returns:
        True if no new extremum was created.
    """
    epsilon = _RELATIVE_EPSILON * float(initial.max() - initial.min())
    return bool(final.min() >= initial.min() - epsilon
                and final.max() <= initial.max() + epsilon)


def main():
    """Compares the bounds of two fields against the expectation."""
    paths = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    expect_violation = "--expect-violation" in sys.argv

    initial, _ = read_field(paths[0])
    final, _ = read_field(paths[1])
    monotone = is_monotone(initial, final)
    overshoot = max(0.0, float(final.max() - initial.max()))
    undershoot = max(0.0, float(initial.min() - final.min()))

    print(f"in  bounds [{initial.min():.6g}, {initial.max():.6g}]")
    print(f"out bounds [{final.min():.6g}, {final.max():.6g}]  "
          f"(over/undershoot: {overshoot:.3g} / {undershoot:.3g})")
    print("MONOTONE" if monotone else "VIOLATION")

    if monotone == expect_violation:
        print("FAIL: expected the unlimited filter to over/undershoot, "
              "but it did not" if expect_violation
              else "FAIL: limiter did not preserve monotonicity")
        sys.exit(1)


if __name__ == "__main__":
    main()
