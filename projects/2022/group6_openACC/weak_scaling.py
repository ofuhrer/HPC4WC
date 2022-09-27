# script to plot the runtimes of the original cpu code and the best gpu code
import numpy as np
import matplotlib.pyplot as plt
import os as os
from glob import glob

# identify the best GPU version
gpu_vers = sorted(glob("data/result_acc*_0128.txt"))
top_ver = gpu_vers[-1].split("_")[1]

# get filenames
#os.chdir("/users/class193/HPC4WC/projects/2022/group06_openACC/performance/")
cpu_files = sorted(glob("data/result_orig_*.txt"))
gpu_files = sorted(glob(f"data/result_{top_ver}_*.txt"))

# define the domain array (x axis of the plot)
domain = []
for file in cpu_files:
    domain.append(file.split("_")[2].split(".")[0])
for index in range(0, len(domain)):
    domain[index] = str(int(domain[index])) + r" $\times$ " + str(int(domain[index]))

# define the runtime arrays (y axis of the plot)
cpu_runtime = []
gpu_runtime = []
for file in cpu_files:
    lines = open(file).readlines()
    exec(" ".join(lines[(-3):None]))
    cpu_runtime.append(data[0, (-1)])
for file in gpu_files:
    lines = open(file).readlines()
    exec(" ".join(lines[(-3):None]))
    gpu_runtime.append(data[0, (-1)])

# plot the data in a bar plot (runtime against domain size)
x_loc = np.arange(len(domain))
bar_width = 0.7/2

titlesize = 30
labelsize = 24
ticklabelsize = 20

fig = plt.figure(figsize = (24, 12))    # width, height

ax1 = fig.add_subplot(1, 1, 1)          # nrow, ncol, nplot
ax1.bar(x_loc - (bar_width/2), cpu_runtime, width=bar_width, color="dodgerblue", label="CPU")
ax1.bar(x_loc + (bar_width/2), gpu_runtime, width=bar_width, color="forestgreen", label="GPU")
ax1.set_title("runtime against domain size", fontsize=titlesize, pad=30)
ax1.set_ylabel("runtime (s)", fontsize=labelsize, labelpad=15)
ax1.set_yscale("log")
ax1.set_xticks(x_loc, domain, fontsize=labelsize)
ax1.tick_params(axis="x", pad=15)
ax1.tick_params(axis="y", labelsize=ticklabelsize)
ax1.legend(loc="best", fontsize=labelsize)

fig.savefig("plots/weak_scaling.pdf", format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.5)


