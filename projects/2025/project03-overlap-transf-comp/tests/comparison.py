# ******************************************************
#     Program: compare_fields.py
#      Author: Stefano Ubbiali
#       Email: subbiali@phys.ethz.ch
#        Date: 04.06.2020
# Description: Comparing two NumPy arrays
# ******************************************************
import click
import numpy as np
import struct

# Set numpy to print full arrays
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def read_field_file(filename):
    """Read binary field file written by C++ or Fortran code."""
    with open(filename, 'rb') as f:
        # Read header (6 int32 values)
        header = struct.unpack('6i', f.read(6 * 4))
        three, sixtyfour, halosize, xsize, ysize, zsize = header
        
        # Read the data as doubles
        num_elements = xsize * ysize * zsize
        data_bytes = f.read(num_elements * 8)  # 8 bytes per double
        data = struct.unpack(f'{num_elements}d', data_bytes)
        
        # Reshape to (xsize, ysize, zsize) - matching C++ memory layout
        field = np.array(data).reshape((zsize, ysize, xsize), order='C')
        
    return field, (halosize, xsize, ysize, zsize)


@click.command()
@click.option("--src", type=str, required=True, help="Path to the first field.")
@click.option("--trg", type=str, required=True, help="Path to the second field.")
@click.option(
    "--rtol", type=float, required=False, default=1e-5, help="Relative tolerance."
)
@click.option(
    "--atol", type=float, required=False, default=1e-8, help="Absolute tolerance."
)
def main(src, trg, rtol=1e-5, atol=1e-8):
    try:
        # Try to load as numpy files first
        src_f = np.load(src)
        trg_f = np.load(trg)
    except (ValueError, OSError):
        # If that fails, try to load as binary field files
        try:
            src_f, src_info = read_field_file(src)
            trg_f, trg_info = read_field_file(trg)
            print(f"Loaded binary field files:")
            print(f"  {src}: shape={src_f.shape}, halo={src_info[0]}")
            print(f"  {trg}: shape={trg_f.shape}, halo={trg_info[0]}")
        except Exception as e:
            print(f"Error reading files: {e}")
            return

    if np.allclose(src_f, trg_f, rtol=rtol, atol=atol, equal_nan=True):
        print(f"HOORAY! '{src}' and '{trg}' are equal!")
    else:
        print(f"{src} and {trg} are not equal.")
        print(f"Max absolute difference: {np.max(np.abs(src_f - trg_f))}")
        print(f"Max relative difference: {np.max(np.abs((src_f - trg_f) / (trg_f + 1e-16)))}")


if __name__ == "__main__":
    main()
