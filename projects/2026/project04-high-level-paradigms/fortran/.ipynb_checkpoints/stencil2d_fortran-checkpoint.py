"""
stencil2d_fortran.py

Python launcher for the compiled stencil2d_fortran.x
binary, built from stencil2d_fortran.F90

"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

# Path to the compiled binary (built via `make` in this same directory).
KERNEL_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stencil2d_fortran.x")


def main():
    parser = argparse.ArgumentParser(description="stencil2d (4th-order diffusion) - Fortran")
    parser.add_argument("--in_field_path", type=str, required=True)
    parser.add_argument("--out_field_path", type=str, required=True)
    parser.add_argument("--num_iter", type=int, required=True)
    parser.add_argument("--num_halo", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.device.lower() != "cpu":
        print(f"error: the fortran kernel only supports --device cpu (got '{args.device}')",
              file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(KERNEL_BIN):
        print(f"error: compiled binary not found at {KERNEL_BIN}\n"
              f"Build it first: cd {os.path.dirname(KERNEL_BIN)} && make",
              file=sys.stderr)
        sys.exit(1)

    field64 = np.load(args.in_field_path)
    nz, ny_full, nx_full = field64.shape
    h = args.num_halo
    ny = ny_full - 2 * h
    nx = nx_full - 2 * h

    # Precision: the Fortran driver uses wp=4 (float32). Downcast here so
    # the raw bytes it reads match its `real(kind=4)` field declaration.
    field32 = np.ascontiguousarray(field64.astype(np.float32))

    with tempfile.TemporaryDirectory() as tmpdir:
        in_raw = os.path.join(tmpdir, "in_field.raw")
        out_raw = os.path.join(tmpdir, "out_field.raw")

        field32.tofile(in_raw)

        #call the binary directly
        cmd = [
            KERNEL_BIN,
            "--nx", str(nx), "--ny", str(ny), "--nz", str(nz),
            "--num_halo", str(h), "--num_iter", str(args.num_iter),
            "--in_field_path", in_raw, "--out_field_path", out_raw,
            "--device", "cpu",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr, end="")
            print(f"error: fortran kernel exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

        out_field32 = np.fromfile(out_raw, dtype=np.float32).reshape(nz, ny_full, nx_full)

    # upcast back to float64 on save so downstream code handling the other
    # frameworks' float64 outputs doesn't need to special-case this dtype
    np.save(args.out_field_path, out_field32.astype(np.float64))

    stdout_lines = result.stdout.strip().splitlines()
    if not stdout_lines:
        print("error: fortran kernel produced no stdout output", file=sys.stderr)
        sys.exit(1)
    print(stdout_lines[-1].strip())


if __name__ == "__main__":
    main()