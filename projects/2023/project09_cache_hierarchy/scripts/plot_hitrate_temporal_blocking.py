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

    file = "../build/stencil_2D_time.csv"
    file_baseline = "../build/stencil_2D_temporal_blocking_baseline.csv"

    data = np.genfromtxt(file, delimiter=",")[1:, :]
    nx, ny, nz = data[:, 0], data[:, 1], data[:, 2]
    nxnynz = nx * ny * nz
    workingSetSize = nxnynz * 2 * 8 / 1e6  # in MBs
    h1, h2 = data[:, 4], data[:, 5]
    # cm1, cm2 = data[:, 6], data[:, 7]

    data = np.genfromtxt(file_baseline, delimiter=",")[1:, :]
    h1_baseline, h2_baseline = data[:, 4], data[:, 5]
    # cm1_baseline, cm2_baseline = data[:, 6], data[:, 7]

    nxnorm = (nx - nx.min()) / (nx.max() - nx.min())
    nynorm = (ny - ny.min()) / (ny.max() - ny.min())

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=100)

    # plot cache sizes in B
    L1 = (2**15) / 1e6
    L2 = (2**18) / 1e6
    L3 = (2**24) / 1e6
    axs[0].axvline(L1, c="r", lw=1.0, ls=":", label="Cache sizes")
    axs[0].axvline(L2, c="r", lw=1.0, ls=":")
    axs[0].axvline(L3, c="r", lw=1.0, ls=":")
    axs[1].axvline(L1, c="r", lw=1.0, ls=":", label="Cache sizes")
    axs[1].axvline(L2, c="r", lw=1.0, ls=":")
    axs[1].axvline(L3, c="r", lw=1.0, ls=":")

    # plot hit rates for baseline
    axs[0].grid(True, which="both", ls=":", lw=0.5)
    axs[0].scatter(
        workingSetSize,
        h1_baseline,
        marker="*",
        label="Baseline",
        color=plt.get_cmap("winter")(0.5 + 0.5 * nxnorm - 0.5 * nynorm),
    )
    axs[0].scatter(
        workingSetSize,
        h1,
        marker=".",
        label="Temporal Blocking",
        color=plt.get_cmap("winter")(0.5 + 0.5 * nxnorm - 0.5 * nynorm),
    )

    axs[1].grid(True, which="both", ls=":", lw=0.5)
    sp = axs[1].scatter(
        workingSetSize,
        h2_baseline,
        marker="*",
        label="Baseline",
        color=plt.get_cmap("winter")(0.5 + 0.5 * nxnorm - 0.5 * nynorm),
    )
    axs[1].scatter(
        workingSetSize,
        h2,
        marker=".",
        label="Temporal Blocking",
        color=plt.get_cmap("winter")(0.5 + 0.5 * nxnorm - 0.5 * nynorm),
    )

    clb = fig.colorbar(sp, ax=axs, orientation="vertical")
    clb.set_label("XY-Ratio", labelpad=-30, y=1.05, rotation=0)

    # plt.scatter(workingSetSize, h2, marker=".", label="Baseline L2")

    axs[0].set_xlabel("Working set size [MB]")
    axs[0].set_ylabel("Hit rate")
    axs[0].legend()
    axs[0].set_xscale("log")
    axs[0].set_title("Hit rate L1")

    axs[1].set_xlabel("Working set size [MB]")
    axs[1].set_ylabel("Hit rate")
    axs[1].legend()
    axs[1].set_xscale("log")
    axs[1].set_title("Hit rate L2")

    # plt.show()
    f = "../build/hit_rates.pdf"
    plt.savefig(f, bbox_inches="tight")
    print("Saved " + f)
    plt.close()


if __name__ == "__main__":
    main()
