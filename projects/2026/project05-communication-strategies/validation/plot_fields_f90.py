# ******************************************************
#     Program: plot_fields_f90.py
# Description: Render the .dat fields produced by the
#              Fortran stencil2d programs (and their
#              difference) as PNGs for visual comparison.
# ******************************************************
import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .compare_fields_f90 import read_field_from_file
else:
    from compare_fields_f90 import read_field_from_file


def plot_field(field, title, filename):
    k = field.shape[0] // 2
    plt.figure()
    im = plt.imshow(field[k, :, :], origin="lower")
    plt.colorbar(im)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def plot_diff(field_a, field_b, title, filename):
    diff = field_a.astype(np.float64) - field_b.astype(np.float64)
    k = diff.shape[0] // 2
    vmax = np.max(np.abs(diff))
    print(f"max |diff| over full volume: {vmax:.3e}  (slice k={k}: {np.max(np.abs(diff[k])):.3e})")
    if vmax == 0:
        vmax = 1e-12
    plt.figure()
    im = plt.imshow(diff[k, :, :], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label=f"diff (vmax={vmax:.2e})")
    plt.title(title)
    plt.savefig(filename)
    plt.close()


@click.command()
@click.option("--in_field", type=str, default="in_field.dat", help="Path to the input field.")
@click.option("--ref_field", type=str, default="out_field_ref.dat", help="Path to the serial reference output field.")
@click.option("--out_field", type=str, default="out_field.dat", help="Path to the MPI output field.")
def main(in_field, ref_field, out_field):
    f_in = read_field_from_file(in_field)
    f_ref = read_field_from_file(ref_field)
    f_out = read_field_from_file(out_field)

    plot_field(f_in, "in_field", "in_field_f90.png")
    plot_field(f_ref, "out_field (serial reference)", "out_field_ref_f90.png")
    plot_field(f_out, "out_field (MPI)", "out_field_f90.png")
    plot_diff(f_out, f_ref, "out_field (MPI) - out_field (serial reference)", "diff_f90.png")

    print("Wrote in_field_f90.png, out_field_ref_f90.png, out_field_f90.png, diff_f90.png")


if __name__ == "__main__":
    main()
