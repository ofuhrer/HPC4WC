import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pathlib
from glob import glob

scriptPath = pathlib.Path(__file__).parent.absolute()
currentPath = pathlib.Path().absolute()

buildFolder = "../build/src"


def main():
    if scriptPath != currentPath:  # failsafe not to delete stuff
        sys.exit("Do not run this from another path!")

    if not os.path.exists(buildFolder):
        sys.exit("Build folder not found!")

    csvfiles = glob(buildFolder + "/*.csv")
    for file in csvfiles:
        if "blocked" in file or "hr.csv" not in file:
            continue

        data = np.genfromtxt(file, delimiter=",")[1:, :]
        nx, ny, nz = data[:, 0], data[:, 1], data[:, 2]
        nxnynz = nx * ny * nz
        workingSetSize = nxnynz * 2 * 8 / 1e6  # in MBs
        h1, h2 = data[:, 4], data[:, 5]

        plt.figure(dpi=100, figsize=(7.25, 4.5))
        plt.rcParams.update({"figure.autolayout": True})

        # plot cache sizes in MB
        L1 = (2**15) / 1e6
        L2 = (2**18) / 1e6
        L3 = (2**24) / 1e6
        plt.axvline(L1, c="r", lw=1.0, ls=":", label="Cache sizes")
        plt.axvline(L2, c="r", lw=1.0, ls=":")
        plt.axvline(L3, c="r", lw=1.0, ls=":")

        # plot hit rates for baseline
        plt.grid(True, which="both", ls=":", lw=0.5)
        plt.scatter(
            workingSetSize,
            h1,
            marker="*",
            s=60,
            color="tab:blue",
            label="Baseline L1",
        )
        plt.scatter(
            workingSetSize,
            h2,
            marker="*",
            s=30,
            color="tab:green",
            label="Baseline L2",
        )

        # find and plot hitrate per grid point for blocked version
        fileBlocked = file[:-7] + "_blocked_hr.csv"
        if fileBlocked in csvfiles:
            data = np.genfromtxt(fileBlocked, delimiter=",")[1:, :]
            nx, ny, nz = data[:, 0], data[:, 1], data[:, 2]
            assert (nxnynz == nx * ny * nz).all()
            h1, h2 = data[:, 4], data[:, 5]

            plt.scatter(
                workingSetSize,
                h1,
                marker=".",
                s=60,
                color="tab:orange",
                label="Spatial blocking L1",
            )
            plt.scatter(
                workingSetSize,
                h2,
                marker=".",
                s=30,
                color="tab:red",
                label="Spatial blocking L2",
            )

        plt.xlabel("Working set size [MB]")
        plt.ylabel("Hit rate")
        plt.legend()

        plt.xscale("log")
        # plt.yscale("log")

        # plt.show()
        f = file[:-3] + "pdf"
        plt.savefig(f)  # bbox_inches="tight"
        print("Saved " + f)
        plt.close()


if __name__ == "__main__":
    main()
