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
        if "blocked" in file or "hr.csv" in file:
            continue

        data = np.genfromtxt(file, delimiter=",")[1:, :]
        nx, ny, nz = data[:, 0], data[:, 1], data[:, 2]
        nxnynz = nx * ny * nz
        workingSetSize = nxnynz * 2 * 8 / 1e6  # in MBs
        runtime = data[:, 3]
        runtimePerGridpoint = runtime / nxnynz * 1e6  # in microseconds

        plt.figure(dpi=100, figsize=(7.25, 4.5))
        plt.rcParams.update({"figure.autolayout": True})

        # plot cache sizes in MB
        L1 = (2**15) / 1e6
        L2 = (2**18) / 1e6
        L3 = (2**24) / 1e6
        plt.axvline(L1, c="r", lw=1.0, ls=":", label="Cache sizes")
        plt.axvline(L2, c="r", lw=1.0, ls=":")
        plt.axvline(L3, c="r", lw=1.0, ls=":")

        # plot run time per grid point for baseline
        plt.grid(True, which="both", ls=":", lw=0.5)
        # nxnorm = (nx - nx.min()) / (nx.max() - nx.min())
        # nynorm = (ny - ny.min()) / (ny.max() - ny.min())
        plt.scatter(
            workingSetSize,
            runtimePerGridpoint,
            marker="*",
            # s=60,
            # color=plt.get_cmap("RdYlBu")(0.5 + 0.5 * nxnorm - 0.5 * nynorm),
            color="tab:blue",
            label="Baseline",
        )

        # find and plot run time per grid point for blocked version
        fileBlocked = file[:-4] + "_blocked.csv"
        if fileBlocked in csvfiles:
            data = np.genfromtxt(fileBlocked, delimiter=",")[1:, :]
            nx, ny, nz = data[:, 0], data[:, 1], data[:, 2]
            assert (nxnynz == nx * ny * nz).all()
            runtime = data[:, 3]
            runtimePerGridpoint = runtime / nxnynz * 1e6
            # nxnorm = (nx - nx.min()) / (nx.max() - nx.min())
            # nynorm = (ny - ny.min()) / (ny.max() - ny.min())
            plt.scatter(
                workingSetSize,
                runtimePerGridpoint,
                marker=".",
                # s=30,
                # color=plt.get_cmap("RdYlGn")(0.5 + 0.5 * nxnorm - 0.5 * nynorm),
                color="tab:orange",
                label="Spatial blocking",
            )

        plt.xlabel("Working set size [MB]")
        plt.ylabel("Runtime per grid point [µs]")
        plt.legend()

        plt.xscale("log")
        plt.yscale("log")

        # plt.show()
        f = file[:-3] + "pdf"
        plt.savefig(f)  # bbox_inches="tight"
        print("Saved " + f)
        plt.close()


if __name__ == "__main__":
    main()
