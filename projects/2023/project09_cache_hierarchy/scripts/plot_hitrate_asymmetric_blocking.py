import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pathlib

scriptPath = pathlib.Path(__file__).parent.absolute()
currentPath = pathlib.Path().absolute()

buildFolder = "../build/src"


def main():
    if scriptPath != currentPath:  # failsafe not to delete stuff
        sys.exit("Do not run this from another path!")

    if not os.path.exists(buildFolder):
        sys.exit("Build folder not found!")

    file = "../build/stencil_2D_asymmetric_blocked.csv"
    data = np.genfromtxt(file, delimiter=",")[1:, :]
    table_time = data[:, 2].reshape((6, 6)).transpose()

    file = "../build/stencil_2D_asymmetric_blocked_hits.csv"
    data = np.genfromtxt(file, delimiter=",")[1:, :]
    table_hits = data[:, 3].reshape((6, 6)).transpose()

    heatmap, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    im1 = ax[0].imshow(
        table_time,
        cmap="winter_r",
        extent=[4, 128, 4, 128],
        interpolation="nearest",
        origin="lower",
        aspect="auto",
    )
    ax[0].set_xticks(np.linspace(4 + 128 / 6 / 2, 128 - 128 / 6 / 2, 6))
    ax[0].set_yticks(np.linspace(4 + 128 / 6 / 2, 128 - 128 / 6 / 2, 6))
    ax[0].set_xticklabels([4, 8, 16, 32, 64, 128])
    ax[0].set_yticklabels([4, 8, 16, 32, 64, 128])
    ax[0].set(
        xlabel="Blocking size in $x$-direction", ylabel="Blocking size in $y$-direction"
    )

    im2 = ax[1].imshow(
        table_hits,
        cmap="winter",
        extent=[4, 128, 4, 128],
        interpolation="nearest",
        origin="lower",
        aspect="auto",
    )
    ax[1].set_xticks(np.linspace(4 + 128 / 6 / 2, 128 - 128 / 6 / 2, 6))
    ax[1].set_yticks(np.linspace(4 + 128 / 6 / 2, 128 - 128 / 6 / 2, 6))
    ax[1].set_xticklabels([4, 8, 16, 32, 64, 128])
    ax[1].set_yticklabels([4, 8, 16, 32, 64, 128])
    ax[1].set(
        xlabel="Blocking size in $x$-direction", ylabel="Blocking size in $y$-direction"
    )

    print(table_hits.transpose())

    clb = heatmap.colorbar(im1, ax=ax[0], orientation="vertical")
    clb.set_label("Time [s]", labelpad=-30, y=1.05, rotation=0)

    clb = heatmap.colorbar(im2, ax=ax[1], orientation="vertical")
    clb.set_label("L1 hit rate", labelpad=-30, y=1.05, rotation=0)

    # plt.show()
    f = "../build/stencil_2D_asymmetric_blocked.pdf"
    plt.savefig(f, bbox_inches="tight")
    print("Saved " + f)
    plt.close()


if __name__ == "__main__":
    main()
