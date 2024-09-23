# This is a copy from HPC4WC/day3
# The only changes are: 
# adding a print function for source and target for easier debugging of small input fields
# only looking at source and target data [3:], because I noticed the first three entries contain unspecified values that probably belong to some metadata (although they are usually the same for both the source and the target field)


# ******************************************************
#     Program: compare_fields.py
#      Author: Stefano Ubbiali
#       Email: subbiali@phys.ethz.ch
#        Date: 04.06.2020
# Description: Comparing two NumPy arrays
# ******************************************************
import click
import numpy as np


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
    src_f = np.fromfile(src)
    trg_f = np.fromfile(trg)
    
    print("source: " + str(src_f[3:]))
    
    print("target: " + str(trg_f[3:]))

    if np.allclose(src_f[3:], trg_f[3:], rtol=rtol, atol=atol, equal_nan=True):
        print(f"HOORAY! '{src}' and '{trg}' are equal!")
    else:
        print(f"{src} and {trg} are not equal.")


if __name__ == "__main__":
    main()
