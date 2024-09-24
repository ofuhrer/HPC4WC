# -*- coding: utf-8 -*-

"""

Produces xz-plots of the isentrop model output

Input:
simname     Simulation name
time        output frame number

TODO:
 - Add ability to plot multiple variables at once
 - Support reading topo from NetCDF File

***************************************************************
* David Leutwyler, 2015, intitial version                     *
* Marina Dütsch, 2015, animation                              *
* Mathias Hauser, 2015, restructuring                         *
* Christian Zeman, 2018/2019, filled contours, limits         *
***************************************************************

"""

import sys
import numpy as np

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import matplotlib.widgets as widgets
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation

from nmwc_model.readsim import readsim


def arg_parser():

    usage = "usage: %(prog)s [options] <filename.npz>\n\
    Basic: %(prog)s output.npz\n\
    Example: %(prog)s -o plot -v horizontal_velocity -t 5 output.npz"
    description = """
    Creates a very basic plot of the output from the model.

    Variables which can be plotted (use these in the -v option):
    - horizontal_velocity
    - specific_humidity (qv)
    - specific_cloud_liquid_water_content (qc)
    - specific_rain_water_content (qr)
    - cloud_number_density (nc)
    - rain_number_density (nr)
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
        help="File holding the NetCDF data from the model",
    )
    # Optional arguments
    op.add_argument(
        "-o",
        dest="figname",
        default="plot.pdf",
        help="Name of the output figure",
        metavar="FILE.pdf",
    )
    op.add_argument(
        "-v",
        dest="varname",
        default="horizontal_velocity",
        metavar="VAR",
        help="Variable(s) to be plotted",
    )
    op.add_argument(
        "-t",
        dest="time",
        default="-1",
        metavar="timestep",
        help="Output timestep. default=last",
        type=int,
    )
    op.add_argument("-a", dest="anim", action="store_true", help="interactive plot")
    op.add_argument(
        "--zlim",
        dest="zlim",
        default=(0, 10),
        nargs=2,
        metavar=("min", "max"),
        help="set the z-axis window [km]",
        type=float,
    )
    op.add_argument(
        "--tci",
        dest="tci",
        default=2,
        metavar="interval",
        type=int,
        help="set theta contouring interval [K]",
    )
    op.add_argument(
        "--tlim",
        dest="tlim",
        default=(0, 500),
        nargs=2,
        metavar=("min", "max"),
        help="set limits for the theta range [K]",
        type=float,
    )
    # horizontal_velocity
    op.add_argument(
        "--vci",
        dest="vci",
        default=2,
        metavar="interval",
        type=int,
        help="set velocity contouring interval [m/s]",
    )
    op.add_argument(
        "--vlim",
        dest="vlim",
        default=(float("nan"), float("nan")),
        nargs=2,
        metavar=("min", "max"),
        help="set limits for velocity range [m/s]",
        type=float,
    )
    # specific_humidity
    op.add_argument(
        "--qvci",
        dest="qvci",
        default=0.25,
        metavar="interval",
        type=float,
        help="set qv contouring interval [g/kg]",
    )
    op.add_argument(
        "--qvlim",
        dest="qvlim",
        default=(0.0, float("nan")),
        nargs=2,
        metavar=("min", "max"),
        help="set limits for the qv range [g/kg]",
        type=float,
    )
    # specific_cloud_liquid_water_content
    op.add_argument(
        "--qcci",
        dest="qcci",
        default=0.1,
        metavar="interval",
        type=float,
        help="set qc contouring interval [g/kg]",
    )
    op.add_argument(
        "--qclim",
        dest="qclim",
        default=(0.1, float("nan")),
        nargs=2,
        metavar=("min", "max"),
        help="set limits for the qc range [g/kg]",
        type=float,
    )
    # cloud_number_density
    op.add_argument(
        "--ncci",
        dest="ncci",
        default=4 * 10 ** 8,
        metavar="interval",
        type=float,
        help="set nc contouring interval",
    )
    op.add_argument(
        "--nclim",
        dest="nclim",
        default=(10 ** 8, float("nan")),
        nargs=2,
        metavar=("min", "max"),
        type=float,
        help="set range for the nc contours",
    )
    # rain_number_density
    op.add_argument(
        "--nrci",
        dest="nrci",
        default=5 * 10 ** 2,
        type=float,
        metavar="interval",
        help="set nr contouring interval",
    )
    op.add_argument(
        "--nrlim",
        dest="nrlim",
        default=(10 ** 2, float("nan")),
        nargs=2,
        metavar=("min", "max"),
        type=float,
        help="set range for the nr contours",
    )
    # specific_rain_water_content
    op.add_argument(
        "--qrci",
        dest="qrci",
        default=0.005,
        metavar="interval",
        type=float,
        help="set qr contouring interval [g/kg]",
    )
    op.add_argument(
        "--qrlim",
        dest="qrlim",
        default=(0.005, float("nan")),
        nargs=2,
        metavar=("min", "max"),
        help="set limits for the qr range [g/kg]",
        type=float,
    )
    op.add_argument(
        "--totpreclim",
        dest="totpreclim",
        default=(0, float("nan")),
        nargs=2,
        metavar=("min", "max"),
        type=float,
        help="set range for the totprec contours",
    )

    return op


# -------------------------------------------------------------------------


def update_im_back(i, data, ax, fig):
    global timestep
    timestep = timestep - 1
    if timestep < 0:
        timestep = data["time"].shape[0] + timestep
    slider.set_val(var.time[timestep])
    ax.cla()
    im = plot_figure(varnames, var, timestep, False)
    return (im,)


def update_im_forward(i, data, ax, fig):
    global timestep
    timestep = timestep + 1
    if timestep > data["time"].shape[0] - 1:
        timestep = timestep - data["time"].shape[0]
    slider.set_val(var.time[timestep])
    ax.cla()
    im = plot_figure(varnames, var, timestep, False)
    return (im,)


def forward(event):
    """move one timestep forward"""
    global timestep
    if timestep == len(var.time) - 1:
        timestep = 0
    else:
        timestep = timestep + 1
    plot_figure(varnames, var, timestep, False)
    slider.set_val(var.time[timestep])


def back(event):
    """move one timestep backward"""
    global timestep
    if timestep == 0:
        timestep = len(var.time) - 1
    else:
        timestep = timestep - 1
    plot_figure(varnames, var, timestep, False)
    slider.set_val(var.time[timestep])


def start(event):
    """move to the start"""
    global timestep
    timestep = 0
    plot_figure(varnames, var, timestep, False)
    slider.set_val(var.time[timestep])


def end(event):
    """move to the end"""
    global timestep
    timestep = len(var.time) - 1
    plot_figure(varnames, var, timestep, False)
    slider.set_val(var.time[timestep])


def forward_play(event):
    if timestep < len(var.time) - 1:
        ani = FuncAnimation(
            fig,
            update_im_forward,
            frames=range(len(var.time) - 2 - timestep),
            fargs=(var, ax, fig),
            interval=interval,
            repeat=False,
        )
        fig.canvas.draw()


def back_play(event):
    if timestep > 0:
        ani = FuncAnimation(
            fig,
            update_im_back,
            frames=range(timestep - 1),
            fargs=(var, ax, fig),
            interval=interval,
            repeat=False,
        )
        fig.canvas.draw()


# def pause_play(event):
#    global pause
#    pause ^= True


def update(val):
    global timestep
    timestep, = np.where(
        abs(var.time - slider.val) == np.min(abs(var.time - slider.val))
    )[0]
    plot_figure(varnames, var, timestep, False)


def speed(label):
    global interval
    interval = init_interval / int(label[0])


# ------------------------------------------------------------------------------


def get_max_value(values, valMin, valInt, offset=1.0):
    # Raise an error if the value is too small to be plotted
    valMax = np.nanmax(values)
    if valMax < valMin:
        print(
            "The maximum value of the variable is smaller than "
            + "the set minimum value and therefore the field "
            + "cannot be plotted. \r\n\r\n"
            + "The values are probably close to zero. If you "
            + "still want to plot them, try manually setting "
            + "the limits and intervals manually the corresponding"
            + "options (use the -h option to see how).\r\n\r\n"
            + "Max. value of field: "
            + str(valMax)
            + "\r\n"
            + "Lower limit:         "
            + str(valMin)
            + "\r\n"
            + "Interval:            "
            + str(valInt)
        )
        sys.exit()
    # Calculate and return the next best value
    nrInt = int((valMax - valMin) / valInt + offset)
    return valMin + nrInt * valInt


def plot_dict(args, var, varnames):
    # make dictionary for plotting
    # the ifs make it useable from the hovz plots
    if isinstance(varnames, str):
        varnames = [varnames]

    pd = dict()
    v = "horizontal_velocity"
    if v in varnames:
        vMinInt = args.vlim[0]
        vMaxInt = args.vlim[1]
        scale = 1
        values = scale * var.horizontal_velocity

        # If no limits are given, they will be set automatically according
        # to the values in the data.
        if np.isnan(vMinInt):
            minVel = scale * np.nanmin(var.horizontal_velocity)
            vciDiff = int((var.u00 - minVel) / args.vci + 0.5)
            vMinInt = var.u00 - vciDiff * args.vci
        if np.isnan(vMaxInt):
            vMaxInt = get_max_value(values, vMinInt, args.vci, 0.5)

        pd[v] = dict(
            fmt="%2i",
            clev=np.arange(vMinInt, vMaxInt + args.vci, args.vci),
            ci=args.vci,
            cmap=plt.cm.RdBu_r,
            scale=scale,
        )

    v = "specific_humidity"
    if v in varnames:
        vMinInt = args.qvlim[0]
        vMaxInt = args.qvlim[1]
        scale = 1000

        # If no upper limit is given, it will be set automatically
        # according to the values in the data.
        if np.isnan(vMaxInt):
            values = scale * var.specific_humidity
            vMaxInt = get_max_value(values, vMinInt, args.qvci)

        pd[v] = dict(
            fmt="%2i",
            clev=np.arange(vMinInt, vMaxInt + args.qvci, args.qvci),
            ci=args.qvci,
            cmap=plt.cm.Blues,
            scale=scale,
        )

    v = "specific_cloud_liquid_water_content"
    if v in varnames:
        vMinInt = args.qclim[0]
        vMaxInt = args.qclim[1]
        scale = 1000

        # If no upper limit is given, it will be set automatically
        # according to the values in the data.
        if np.isnan(vMaxInt):
            values = scale * var.specific_cloud_liquid_water_content
            vMaxInt = get_max_value(values, vMinInt, args.qcci)

        pd[v] = dict(
            fmt="%1.1f",
            clev=np.arange(vMinInt, vMaxInt + args.qcci, args.qcci),
            ci=args.qcci,
            cmap=plt.cm.Purples,
            scale=scale,
        )

    v = "cloud_number_density"
    if v in varnames:
        vMinInt = args.nclim[0]
        vMaxInt = args.nclim[1]
        scale = 1

        # If no upper limit is given, it will be set automatically
        # according to the values in the data.
        if np.isnan(vMaxInt):
            values = scale * var.cloud_number_density
            vMaxInt = get_max_value(values, vMinInt, args.ncci)

        pd[v] = dict(
            fmt="%1.1e",
            clev=np.arange(vMinInt, vMaxInt + args.ncci, args.ncci),
            ci=args.ncci,
            cmap=plt.cm.Oranges,
            scale=scale,
        )

    v = "rain_number_density"
    if v in varnames:
        vMinInt = args.nrlim[0]
        vMaxInt = args.nrlim[1]
        scale = 1

        # If no upper limit is given, it will be set automatically
        # according to the values in the data.
        if np.isnan(vMaxInt):
            values = scale * var.rain_number_density
            vMaxInt = get_max_value(values, vMinInt, args.nrci)

        pd[v] = dict(
            fmt="%1.1e",
            clev=np.arange(vMinInt, vMaxInt + args.nrci, args.nrci),
            ci=args.nrci,
            cmap=plt.cm.Reds,
            scale=scale,
        )

    v = "specific_rain_water_content"
    if v in varnames:
        vMinInt = args.qrlim[0]
        vMaxInt = args.qrlim[1]
        scale = 1000

        # If no upper limit is given, it will be set automatically
        # according to the values in the data.
        if np.isnan(vMaxInt):
            values = scale * var.specific_rain_water_content
            vMaxInt = get_max_value(values, vMinInt, args.qrci)
            if (vMaxInt - vMinInt) / args.qrci < 1.0:
                vMaxInt += args.qrci

        pd[v] = dict(
            fmt="%1.3f",
            clev=np.arange(vMinInt, vMaxInt + args.qrci, args.qrci),
            ci=args.qrci,
            cmap=plt.cm.Greens,
            scale=scale,
        )

    return pd


# -----------------------------------------------------------------------------


def plot_figure(varnames, var, timestep, plot_cbar):
    plt.sca(ax)
    ax.cla()

    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.set_xlabel("x [km]")
    ax.set_ylabel("Height [km]")

    # Add theta
    if args.tlim[0] == 0:
        tlim_min = var.th00 + var.dth / 2
    else:
        tlim_min = args.tlim[0]

    clev = np.arange(tlim_min, args.tlim[1], args.tci)

    plt.contour(
        var.xp[:, :], var.zp[timestep, :, :],
            var.theta[:, :], clev, colors="grey", linewidths=1
    )

    # Add topography
    plt.plot(var.xp[0, :], var.topo, "-k")
    plt.ylim(args.zlim)

    for varname in varnames:
        # Determine range for values and ticks
        valRange = np.arange(
            pd[varname]["clev"][0],
            pd[varname]["clev"][-1] + pd[varname]["ci"],
            pd[varname]["ci"],
        )
        ticks = np.arange(
            pd[varname]["clev"][0],
            pd[varname]["clev"][-1] + pd[varname]["ci"],
            pd[varname]["ci"],
        )

        vmin = valRange[0]
        vmax = valRange[-1]
        if varname == "horizontal_velocity":
            valRange = np.arange(
                pd[varname]["clev"][0] - 0.5 * pd[varname]["ci"],
                pd[varname]["clev"][-1] + 1.5 * pd[varname]["ci"],
                pd[varname]["ci"],
            )
            distUpMid = pd[varname]["clev"][-1] + 0.5 * args.vci - var.u00
            distMidDown = var.u00 - pd[varname]["clev"][0] - 0.5 * args.vci
            maxDist = max(distUpMid, distMidDown)
            vmin = var.u00 - maxDist
            vmax = var.u00 + maxDist

        # Plot
        cs = ax.contourf(
            var.xp[:, :],
            var.zp[timestep, :, :],
            pd[varname]["scale"] * var[varname][timestep, :, :],
            valRange,
            vmin=vmin,
            vmax=vmax,
            cmap=pd[varname]["cmap"],
        )

        # Add a colorbar if needed
        if plot_cbar:
            cb = plt.colorbar(cs, ticks=ticks, spacing="uniform")

        if varname == "specific_rain_water_content":
            tpi = 0.1
            vMinInt = args.totpreclim[0]
            vMaxInt = args.totpreclim[1]

            # If no upper limit is given, it will be set automatically
            # according to the values in the data.
            if np.isnan(vMaxInt):
                maxTp = np.nanmax(var.accumulated_precipitation[:, :])
                tpDiff = int((maxTp - vMinInt) / tpi + 1.0)
                vMaxInt = vMinInt + tpDiff * tpi

            ax2.cla()
            ax2.set_ylabel("Acum. Rain [mm]")
            #             ax2.set_ylim(args.totpreclim)
            ax2.set_ylim(vMinInt, vMaxInt)
            cs = ax2.plot(
                var.xp[0, :], var.accumulated_precipitation[timestep, :], "b-"
            )


#                           var.accumulated_precipitation[timestep, :], 'b-')

# -----------------------------------------------------------------------------


def plot_setup(varnames, timestep):

    # Create Basic Figure
    if not anim:
        fig = plt.figure(figsize=(6, 5))
    else:
        fig = plt.figure(figsize=(10, 9))

    fig.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.96)

    if not anim:
        if "specific_rain_water_content" in varnames:
            gs = gridspec.GridSpec(3, 1, height_ratios=[20, 1, 10])
        else:
            gs = gridspec.GridSpec(1, 1, height_ratios=[20])
    else:
        timestep = 0
        if "specific_rain_water_content" in varnames:
            gs = gridspec.GridSpec(4, 1, height_ratios=[20, 1, 10, 1])
        else:
            gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1])

    ax = fig.add_subplot(gs[0, 0])

    if "specific_rain_water_content" in varnames:
        ax2 = fig.add_subplot(gs[2, 0])
    else:
        ax2 = None

    if not set(varnames) <= set(pd.keys()):
        msg = "specified variable not supported, " "please check: {0} -h".format(
            sys.argv[0]
        )
        sys.exit(msg)

    return fig, ax, ax2, timestep


if __name__ == "__main__":
    op = arg_parser()
    # get command line arguments
    args = op.parse_args()
    # ds = Dataset(args.filename[0], 'r')
    timestep = args.time
    varnames = args.varname.split(",")
    anim = bool(args.anim)

    # load data
    var = readsim(args.filename[0], varnames)

    # get plotting settings for different varnames
    pd = plot_dict(args, var, varnames)

    # setup of the figure
    fig, ax, ax2, timestep = plot_setup(varnames, timestep)

    # plot
    plot_figure(varnames, var, timestep, True)
    # plt.show()

    if anim:
        global interval
        init_interval = 300
        interval = init_interval

        bend = widgets.Button(plt.axes([0.735, 0.06, 0.09, 0.065]), "end")
        bforward_play = widgets.Button(plt.axes([0.635, 0.06, 0.09, 0.065]), ">>")
        bforward = widgets.Button(plt.axes([0.535, 0.06, 0.09, 0.065]), ">")
        # bpause = widgets.Button(plt.axes([0.455, 0.06, 0.09, 0.065]),'||')
        bback = widgets.Button(plt.axes([0.435, 0.06, 0.09, 0.065]), "<")
        bback_play = widgets.Button(plt.axes([0.335, 0.06, 0.09, 0.065]), "<<")
        bstart = widgets.Button(plt.axes([0.235, 0.06, 0.09, 0.065]), "start")
        slider = widgets.Slider(
            plt.axes([0.235, 0.01, 0.59, 0.03]),
            "Time",
            var.time[0],
            var.time[-1],
            valinit=0.0,
        )
        # radio = widgets.RadioButtons(plt.axes([0.845, 0.01, 0.125, 0.115]), ('1x', '2x', '3x'), active=0)

        bend.on_clicked(end)
        bforward.on_clicked(forward)
        bforward_play.on_clicked(forward_play)
        # bpause.on_clicked(pause_play)
        bback.on_clicked(back)
        bback_play.on_clicked(back_play)
        bstart.on_clicked(start)
        slider.on_changed(update)
        # radio.on_clicked(speed)
        plt.show()

    if anim == False:
        with plt.rc_context({"savefig.format": "pdf"}):
            plt.savefig(args.figname)
        # else:
        plt.show()
