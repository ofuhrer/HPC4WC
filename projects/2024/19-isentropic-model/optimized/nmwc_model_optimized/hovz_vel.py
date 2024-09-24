#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Hauser
# Date: March, 2015
# after the matlab version

# in some cases a selection of an apropriate backend is necessary for a plot to show up (uncomment import matplotlib and one of the two backends).
# import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Qt4Agg')

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from nmwc_model_optimized.readsim import readsim
from nmwc_model_optimized.xzplot import plot_dict


def arg_parser():

    usage = "usage: %(prog)s [options] <filename.nc> <x coord>\n\
    Basic: %(prog)s Downslope 1\n\
    Example: %(prog)s -o plot.pdf --vci 5 Downslope 50"

    description = """
    Produces Hovmoeller plots (t, z-plots) of velocity

    See Also
    --------
    hovx_vel
    """

    op = ArgumentParser(
        usage=usage,
        description=description,
        formatter_class=RawDescriptionHelpFormatter,
    )
    # Positional arguments
    op.add_argument(
        "filename",
        metavar="filename",
        nargs=1,
        type=str,
        help="File holding the data from the model",
    )

    op.add_argument(
        "xc", metavar="xc", nargs=1, type=int, help="x coordinate in grid points"
    )

    # Optional arguments
    op.add_argument(
        "-o",
        dest="figname",
        default="hovz_vel.pdf",
        help="Name of the output figure",
        metavar="FILE.pdf",
    )
    op.add_argument(
        "--vci",
        dest="vci",
        default=2,
        metavar="2",
        type=int,
        help="set velocity contouring interval [m/s]",
    )
    op.add_argument(
        "--vlim",
        dest="vlim",
        default=(0, 60),
        nargs=2,
        metavar=("0", "60"),
        help="restrict the velocity contours",
        type=float,
    )
    return op


def plot():

    f, ax = plt.subplots(1)

    data = np.squeeze(var.horizontal_velocity[:, :, xc])

    # Determine min and max integer velocities for plotting routine
    minVel = np.nanmin(data)
    vciDiff = int((var.u00 - minVel) / args.vci + 0.5)
    vMinInt = var.u00 - vciDiff * args.vci
    maxVel = np.nanmax(data)
    vciDiff = int((maxVel - var.u00) / args.vci + 0.5)
    vMaxInt = var.u00 + vciDiff * args.vci

    # Set value range and ticks
    clev = np.arange(vMinInt, vMaxInt + args.vci, args.vci)
    ticks = np.arange(clev[0], clev[-1] + args.vci, args.vci)
    valRange = np.arange(clev[0] - 0.5 * args.vci, clev[-1] + 1.5 * args.vci, args.vci)

    # Set min and max value for color normalization
    distUpMid = clev[-1] + 1.5 * args.vci - var.u00
    distMidDown = var.u00 - clev[0] - 0.5 * args.vci
    maxDist = max(distUpMid, distMidDown)
    vmin = var.u00 - maxDist
    vmax = var.u00 + maxDist

    # Plot
    cs = ax.contourf(
        var.time / 3600.0,
        var.zp[-1, :, xc],
        data.T,
        valRange,
        vmin=vmin,
        vmax=vmax,
        cmap=pd[varname]["cmap"],
    )

    # Add a colorbar
    cb = plt.colorbar(cs, ticks=ticks, spacing="uniform")

    ax.set_ylabel("Height [km]")
    ax.set_xlabel("Time [h]")
    ax.xaxis.set_minor_locator(MultipleLocator(50))

    ax.set_title("Velocity at x = {0}".format(xc))


if __name__ == "__main__":

    op = arg_parser()
    # get command line arguments
    args = op.parse_args()

    xc = args.xc[0]

    varname = "horizontal_velocity"
    var = readsim(args.filename[0], varname)
    pd = plot_dict(args, var, varname)

    plot()

    with plt.rc_context({"savefig.format": "pdf"}):
        plt.savefig(args.figname)

    plt.show()
