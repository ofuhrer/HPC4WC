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
    src_f = np.load(src)
    trg_f = np.load(trg)

    if np.allclose(src_f, trg_f, rtol=rtol, atol=atol, equal_nan=True):
        print(f"HOORAY! '{src}' and '{trg}' are equal!")
    else:
        print(f"{src} and {trg} are not equal.")


if __name__ == "__main__":
    main()
