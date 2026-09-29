# Imports
import math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

plt.rc('font', size=10)         #controls default text size
plt.rc('axes', titlesize=10)    #fontsize of the title
plt.rc('axes', labelsize=10)    #fontsize of the x and y labels
plt.rc('xtick', labelsize=10)   #fontsize of the x tick labels
plt.rc('ytick', labelsize=10)   #fontsize of the y tick labels
plt.rc('legend', fontsize=8)    #fontsize of the legend

def resolution_vis(filelist, dtypes, nx, nums, dt = 1e-3, n_halo = 3):   # plots the same resolution at different numerical precisions

    # Setting up the figure
    fig = plt.figure(figsize=(150 / 25.4, 150 / 25.4), constrained_layout=True)
    gs = fig.add_gridspec(ncols=2, nrows=3, width_ratios=[1, 1], hspace=0.1, wspace=0.1)
    ax_tf = fig.add_subplot(gs[0, 0])
    ax_mf = fig.add_subplot(gs[1, 0], sharex = ax_tf, sharey = ax_tf)
    ax_bf = fig.add_subplot(gs[2, 0], sharex = ax_tf, sharey = ax_tf)
    ax_tl = fig.add_subplot(gs[0, 1])
    ax_ml = fig.add_subplot(gs[1, 1], sharex = ax_tl, sharey = ax_tl)
    ax_bl = fig.add_subplot(gs[2, 1], sharex = ax_tl, sharey = ax_tl)

    ax_tf.set_aspect(1, adjustable='box')
    ax_mf.set_aspect(1, adjustable='box')
    ax_bf.set_aspect(1, adjustable = 'box')

    ax_tf.axis('off')
    ax_mf.axis('off')
    ax_bf.axis('off')

    ax_tf.set_xlim([0, nx])
    ax_tf.set_ylim([0, nx])
    ax_tl.set_xlim([0, 1])
    ax_tl.set_ylim([-1, 1])

    ax_tl.spines['left'].set_visible(False)
    ax_tl.spines['bottom'].set_visible(False)
    ax_tl.spines['top'].set_visible(False)

    ax_ml.spines['left'].set_visible(False)
    ax_ml.spines['bottom'].set_visible(False)
    ax_ml.spines['top'].set_visible(False)

    ax_bl.spines['left'].set_visible(False)
    ax_bl.spines['top'].set_visible(False)

    ax_tl.set_ylabel(mode)
    ax_ml.set_ylabel(mode)
    ax_bl.set_ylabel(mode)

    ax_tl.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False)

    ax_tl.tick_params(
        axis='y',
        which='both',
        left=False,
        right=True,
        labelleft=False,
        labelright=True)

    ax_ml.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False)

    ax_ml.tick_params(
        axis='y',
        which='both',
        left=False,
        right=True,
        labelleft=False,
        labelright=True)

    ax_bl.tick_params(
        axis='x',
        which='both',
        bottom=True,
        top=False,
        labelbottom=True,
        labeltop=False)

    ax_bl.tick_params(
        axis='y',
        which='both',
        left=False,
        right=True,
        labelleft=False,
        labelright=True)

    ax_tl.yaxis.set_label_position('right')
    ax_ml.yaxis.set_label_position('right')
    ax_bl.yaxis.set_label_position('right')

    ax_bl.set_xlabel('Y-Position')

    header_size = 24                                                # to skip metdadata
    phys_field_size = (nx + 2 * n_halo) * (nx + 2 * n_halo) * nz    # actual size to be read (in bytes)

    frames = [0, math.floor(nums / 2), nums-1]                        # number of frames to display
    mapax = [ax_tf, ax_mf, ax_bf]                                   # list of axes for plotting
    linax = [ax_tl, ax_ml, ax_bl]                                   # list of axes for plotting

    labels = ['16 bytes', '8 bytes', '4 bytes']                     # list of pertaining labels for plotting

    for i, file in enumerate(filelist):

        dtype = dtypes[i]                                           # set dtype

        field_size = phys_field_size * np.dtype(dtype).itemsize     # get actual field size

        for j, frame in enumerate(frames):
            with open(file, "rb") as o_file:
                # if this is the first frame, only skip header
                if j == 0:
                    data = np.fromfile(o_file, dtype=dtype, count=phys_field_size, sep="", offset=header_size).reshape(nx + 2 * n_halo, nx + 2 * n_halo)

                # else skip unneeded frames and a header
                else:
                    data = np.fromfile(o_file, dtype=dtype, count=phys_field_size, sep="", offset= header_size + (frame-1) * (header_size + field_size)).reshape(nx + 2 * n_halo, nx + 2 * n_halo)

                # crop the snapshot to not include halo
                inner = data[n_halo:n_halo + nx, n_halo:n_halo + nx].astype(dtype)

            if i == 0:
                mapax[j].plot([math.floor(nx / 2), math.floor(nx / 2)], [0, nx], 'k--', linewidth=1)                 # only show longdouble field
                tocolorbar = mapax[j].imshow(inner, cmap='bwr', vmin=-1, vmax=1)                                            # only show longdouble field
                plt.colorbar(tocolorbar, location='left', label=mode)                                                       # only show longdouble field
            idx = math.floor(nx/2)
            linax[j].plot(np.linspace(start=0, stop=1, num=nx), inner[:, idx], label = labels[i], linewidth=0.6)      # plot the vertical line
            linax[j].set_title(f'Simulation Time: {frame * dt:.4f} s')

    # print legend
    ax_tl.legend(loc='lower left')
    plt.savefig(f'figures/overview_nx_{nx}_' + mode + '_tend_' + f'{frame * dt:.4f}' + '.svg', format='svg', dpi=1000)

if __name__ == '__main__':

    # call signature: "python visualisation.py --nx $nx --ny $ny --nz $nz --Tend $Tend --nums $nums --mode u"

    # read from command line args
    nx = int(sys.argv[2])
    ny = int(sys.argv[4])
    nz = int(sys.argv[6])
    Tend = int(sys.argv[8])
    nums = int(sys.argv[10])
    mode = str(sys.argv[12])

    # first let's plot the lines for resolution
    for res in [int(nx), int(nx/2), int(nx/4)]:

        filelist = ['output_data/' + mode + f'_out_field_nx_{res}_ny_{res}_nz_{nz}_' + prec + '.dat' for prec in ['longdouble', 'double', 'single']]
        dtypes = [np.longdouble, np.float64, np.float32]
        resolution_vis(filelist=filelist, dtypes=dtypes, nums=nums, nx=res, dt=Tend/(nums-1))