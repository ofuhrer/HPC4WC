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

        ## plot runtime

        data = np.genfromtxt(file, delimiter=",")[1:, :]
        nx, ny, nz = data[:, 0], data[:, 1], data[:, 2]
        nxnynz = nx * ny * nz
        workingSetSize = nxnynz * 2 * 8 / 1e6  # in MBs
        runtime = data[:, 3]
        runtimePerGridpoint = runtime / nxnynz * 1e6  # in microseconds

        fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=100)

        # plot cache sizes in MB
        L1 = (2**15) / 1e6
        L2 = (2**18) / 1e6
        L3 = (2**24) / 1e6
        ax[0].axvline(L1, c="r", lw=1.0, ls=":", label="Cache sizes")
        ax[0].axvline(L2, c="r", lw=1.0, ls=":")
        ax[0].axvline(L3, c="r", lw=1.0, ls=":")

        # plot run time per grid point for baseline
        ax[0].grid(True, which="both", ls=":", lw=0.5)
        # nxnorm = (nx - nx.min()) / (nx.max() - nx.min())
        # nynorm = (ny - ny.min()) / (ny.max() - ny.min())
        ax[0].scatter(
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
            ax[0].scatter(
                workingSetSize,
                runtimePerGridpoint,
                marker=".",
                # s=30,
                # color=plt.get_cmap("RdYlGn")(0.5 + 0.5 * nxnorm - 0.5 * nynorm),
                color="tab:orange",
                label="Spatial blocking",
            )

        ax[0].set_xlabel("Working set size [MB]")
        ax[0].set_ylabel("Runtime per grid point [µs]")
        ax[0].legend()

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")

        ## plot hitrate

        filehr = file[:-4] + "_hr.csv"
        data = np.genfromtxt(filehr, delimiter=",")[1:, :]
        nx, ny, nz = data[:, 0], data[:, 1], data[:, 2]
        nxnynz = nx * ny * nz
        workingSetSize = nxnynz * 2 * 8 / 1e6  # in MBs
        h1, h2 = data[:, 4], data[:, 5]

        # plot cache sizes in MB
        L1 = (2**15) / 1e6
        L2 = (2**18) / 1e6
        L3 = (2**24) / 1e6
        ax[1].axvline(L1, c="r", lw=1.0, ls=":", label="Cache sizes")
        ax[1].axvline(L2, c="r", lw=1.0, ls=":")
        ax[1].axvline(L3, c="r", lw=1.0, ls=":")

        # plot hit rates for baseline
        ax[1].grid(True, which="both", ls=":", lw=0.5)
        ax[1].scatter(
            workingSetSize,
            h1,
            marker="*",
            s=60,
            color="tab:blue",
            label="Baseline L1",
        )
        ax[1].scatter(
            workingSetSize,
            h2,
            marker="*",
            s=30,
            color="tab:green",
            label="Baseline L2",
        )

        # find and plot run time per grid point for blocked version
        fileBlocked = file[:-4] + "_blocked_hr.csv"
        if fileBlocked in csvfiles:
            data = np.genfromtxt(fileBlocked, delimiter=",")[1:, :]
            nx, ny, nz = data[:, 0], data[:, 1], data[:, 2]
            assert (nxnynz == nx * ny * nz).all()
            h1, h2 = data[:, 4], data[:, 5]

            ax[1].scatter(
                workingSetSize,
                h1,
                marker=".",
                s=60,
                color="tab:orange",
                label="Spatial blocking L1",
            )
            ax[1].scatter(
                workingSetSize,
                h2,
                marker=".",
                s=30,
                color="tab:red",
                label="Spatial blocking L2",
            )

        ax[1].set_xlabel("Working set size [MB]")
        ax[1].set_ylabel("Hit rate")
        ax[1].legend()

        ax[1].set_xscale("log")
        # ax[1].set_yscale("log")

        # plt.show()
        f = file[:-3] + "pdf"
        plt.savefig(f, bbox_inches="tight")
        print("Saved " + f)
        plt.close()


if __name__ == "__main__":
    main()
