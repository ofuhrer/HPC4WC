# ******************************************************
#     Program: compare_fields_f90.py
# Description: Comparing two fields written by the Fortran
#              write_field_to_file() routine (m_utils.F90)
# ******************************************************
import click
import numpy as np


def read_field_from_file(filename):
    (rank, nbits, num_halo) = np.fromfile(filename, dtype=np.int32, count=3)
    shape = np.fromfile(filename, dtype=np.int32, count=rank, offset=3 * 4)
    offset = (3 + rank) * 4
    dtype = np.float32 if nbits == 32 else np.float64
    data = np.fromfile(filename, dtype=dtype, offset=offset)
    return np.reshape(data, shape[::-1])


@click.command()
@click.option("--src", type=str, required=True, help="Path to the first field.")
@click.option("--trg", type=str, required=True, help="Path to the second field.")
@click.option("--rtol", type=float, required=False, default=1e-5, help="Relative tolerance.")
@click.option("--atol", type=float, required=False, default=1e-5, help="Absolute tolerance.")
def main(src, trg, rtol=1e-4, atol=1e-5):
    src_f = read_field_from_file(src)
    trg_f = read_field_from_file(trg)

    diff = np.abs(src_f.astype(np.float64) - trg_f.astype(np.float64))
    max_abs_diff = np.max(diff)
    denom = np.abs(trg_f.astype(np.float64))
    max_rel_diff = np.max(diff / np.where(denom > 0, denom, 1.0))
    print(f"max absolute difference: {max_abs_diff:.3e}  (float32 eps ~= 1.2e-7)")
    print(f"max relative difference: {max_rel_diff:.3e}")

    if np.allclose(src_f, trg_f, rtol=rtol, atol=atol, equal_nan=True):
        print(f"HOORAY! '{src}' and '{trg}' are equal (within rtol={rtol}, atol={atol})!")
    else:
        raise click.ClickException(f"{src} and {trg} are not equal (within rtol={rtol}, atol={atol}).")


if __name__ == "__main__":
    main()
