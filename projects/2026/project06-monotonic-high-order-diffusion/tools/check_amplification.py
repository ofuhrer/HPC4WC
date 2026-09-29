"""Checks the measured damping factor against the analytic value.

For a single-eigenmode initial condition the linear filter multiplies the
mode by lambda once per iteration, so after N iterations the field must
equal lambda**N times the initial field. The measured factor is the
least-squares ratio <initial, final> / <initial, initial>.

Usage:
    check_amplification.py IN.dat OUT.dat ORDER ALPHA KX KY NITER [RTOL]

Exits 0 if the relative error is within RTOL, 1 otherwise.

Authors:
    Stefanie Boersig <stefanie.boersig@env.ethz.ch>
    Boaz Ko <boazko@student.ethz.ch>
    Ben Bullinger <ben.bullinger@inf.ethz.ch>
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# pylint: disable=wrong-import-position
from read_field import read_field  # noqa: E402

_DEFAULT_RTOL = 2e-3


def damping_factor(order, alpha, kx, ky, nx, ny):
    """Returns the analytic per-iteration damping factor lambda.

    Args:
        order: Filter order n, even.
        alpha: Diffusion coefficient.
        kx: Mode index along x.
        ky: Mode index along y.
        nx: Grid points along x.
        ny: Grid points along y.

    Returns:
        The eigenvalue of one filter application for that mode.
    """
    symbol = (4 * math.sin(math.pi * kx / nx) ** 2
              + 4 * math.sin(math.pi * ky / ny) ** 2)
    return 1.0 - alpha * symbol ** (order // 2)


def main():
    """Compares the measured damping against the analytic prediction."""
    initial_path, final_path = sys.argv[1], sys.argv[2]
    order, alpha = int(sys.argv[3]), float(sys.argv[4])
    kx, ky, niter = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    rtol = float(sys.argv[8]) if len(sys.argv) > 8 else _DEFAULT_RTOL

    initial, _ = read_field(initial_path)
    final, _ = read_field(final_path)
    ny, nx = initial.shape[-2], initial.shape[-1]

    lam = damping_factor(order, alpha, kx, ky, nx, ny)
    expected = lam ** niter
    measured = float((initial * final).sum()) / float((initial * initial).sum())
    error = abs(measured - expected) / max(abs(expected), 1e-12)

    print(f"order {order}, alpha {alpha:.6g}, mode ({kx},{ky}): "
          f"lambda = {lam:.8f}, expected^N = {expected:.8f}, "
          f"measured = {measured:.8f}, rel err = {error:.2e}")
    if error > rtol:
        print(f"FAIL: relative error {error:.2e} > {rtol}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
