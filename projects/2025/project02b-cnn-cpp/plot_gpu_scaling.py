import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# ---------------------------
# Style
# ---------------------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.grid": True,
})

os.makedirs("figs/GPU", exist_ok=True)

# Load data
strong_df = pd.read_csv("logs/results_strong.tsv", sep="\t")
weak_df = pd.read_csv("logs/results_weak.tsv", sep="\t")

# Format x-axis as powers of 2
gpu_ticks = [1, 2, 4, 8]
gpu_labels = [r"$2^0$", r"$2^1$", r"$2^2$", r"$2^3$"]

# ---------------------------
# Strong Scaling
# ---------------------------
plt.figure()
plt.plot(strong_df["GPUs"], strong_df["Time(s)"], marker="o", linewidth=2)
plt.xlabel("Number of GPUs")
plt.ylabel("Time (s)")
# plt.title("Strong Scaling: Time vs GPUs")
plt.xticks(gpu_ticks, gpu_labels)
plt.tight_layout()
plt.savefig("figs/GPU/strong_time.png", dpi=300)

plt.figure()
plt.plot(strong_df["GPUs"], strong_df["Speedup"], marker="o", linewidth=2)
plt.xlabel("Number of GPUs")
plt.ylabel("Speedup")
# plt.title("Strong Scaling: Speedup vs GPUs")
plt.xticks(gpu_ticks, gpu_labels)
plt.tight_layout()
plt.savefig("figs/GPU/strong_speedup.png", dpi=300)

plt.figure()
plt.plot(strong_df["GPUs"], strong_df["Efficiency(%)"], marker="o", linewidth=2)
plt.xlabel("Number of GPUs")
plt.ylabel("Efficiency (\%)")
# plt.title("Strong Scaling: Efficiency vs GPUs")
plt.xticks(gpu_ticks, gpu_labels)
plt.tight_layout()
plt.savefig("figs/GPU/strong_efficiency.png", dpi=300)

# ---------------------------
# Weak Scaling
# ---------------------------
plt.figure()
plt.plot(weak_df["GPUs"], weak_df["Time(s)"], marker="o", linewidth=2)
plt.xlabel("Number of GPUs")
plt.ylabel("Time (s)")
# plt.title("Weak Scaling: Time vs GPUs")
plt.xticks(gpu_ticks, gpu_labels)
plt.tight_layout()
plt.savefig("figs/GPU/weak_time.png", dpi=300)

plt.figure()
plt.plot(weak_df["GPUs"], weak_df["Speedup"], marker="o", linewidth=2)
plt.xlabel("Number of GPUs")
plt.ylabel("Speedup")
# plt.title("Weak Scaling: Speedup vs GPUs")
plt.xticks(gpu_ticks, gpu_labels)
plt.tight_layout()
plt.savefig("figs/GPU/weak_speedup.png", dpi=300)

plt.figure()
plt.plot(weak_df["GPUs"], weak_df["Efficiency(%)"], marker="o", linewidth=2)
plt.xlabel("Number of GPUs")
plt.ylabel("Efficiency (\%)")
# plt.title("Weak Scaling: Efficiency vs GPUs")
plt.xticks(gpu_ticks, gpu_labels)
plt.tight_layout()
plt.savefig("figs/GPU/weak_efficiency.png", dpi=300)
