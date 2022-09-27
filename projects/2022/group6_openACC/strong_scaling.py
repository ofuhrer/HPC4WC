# script to plot the runtimes of the original cpu code and the different gpu codes
import numpy as np
import matplotlib.pyplot as plt
import os as os
from glob import glob

# get filenames
#os.chdir("/users/class193/HPC4WC/projects/2022/group06_openACC/performance/")
cpu_file = sorted(glob("data/result_orig_0128.txt"))
gpu_files = sorted(glob("data/result_acc*_0128.txt"))
result_files = cpu_file + gpu_files

# define the version array (x axis of the plot)
version = []
for file in result_files:
    version.append(file.split("_")[1])

# define the runtime array (y axis of the plot)
runtime = []
for file in result_files:
    lines = open(file).readlines()
    exec(" ".join(lines[(-3):None]))
    runtime.append(data[0, (-1)])

# plot the data in a bar plot (runtime against code version)
x_loc = np.arange(len(version))
bar_width = 0.7

titlesize = 30
labelsize = 24
ticklabelsize = 20

fig = plt.figure(figsize = (24, 12))    # width, height

ax1 = fig.add_subplot(1, 1, 1)          # nrow, ncol, nplot
ax1.bar(x_loc[0], runtime[0], width=bar_width, color="dodgerblue", label="CPU")
ax1.bar(x_loc[1:None], runtime[1:None], width=bar_width, color="forestgreen", label="GPU")
ax1.set_title("runtime against code version", fontsize=titlesize, pad=30)
ax1.set_ylabel("runtime (s)", fontsize=labelsize, labelpad=15)
ax1.set_xticks(x_loc, version)
ax1.tick_params(axis="x", pad=15, labelsize=labelsize)
ax1.tick_params(axis="y", labelsize=ticklabelsize)
ax1.legend(loc="best", fontsize=labelsize)

fig.savefig("plots/strong_scaling.pdf", format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.5)


