#!/usr/bin/env python3
"""
Plot strong-scaling results for CPU multi-thread vs CUDA GPU variant.

Produces three figures saved to ./figs:
 - scaling_time.png
 - scaling_speedup.png
 - scaling_efficiency.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------
# Style
# ---------------------------
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.grid": True,
})

os.makedirs("figs", exist_ok=True)

# ---------------------------
# Display name mapping 
# ---------------------------
DISPLAY_NAME = {
    "opt":               "CPU Naive: Parallelized + Optimized",
    "opt_alt":           "CPU Naive: Simple Parallelization",
    "stencil_opt":       "CPU Stencil: Parallelized + Optimized",
    "stencil_opt_alt":   "CPU Stencil: Simple parallelization",
}

DESIRED_CPU_ORDER = [
    "opt_alt",
    "opt",
    "stencil_opt_alt",
    "stencil_opt",
]

# ---------------------------
# Data (extracted from provided results)
# ---------------------------
xpu_counts = np.array([1, 2, 4, 8])
xpu_labels = [r"$2^0$", r"$2^1$", r"$2^2$", r"$2^3$"]

# GPU strong-scaling (GPUs: Time(s))
cuda_gpu_times = {
    1: 9.156343,
    2: 5.594964,
    4: 4.101600,
    8: 2.961734,
}

# CPU variants: 
cpu_variants_times = {
    "opt": {
        1: 20.1658724180015,
        2: 12.3119377344847,
        4: 8.14340847748099,
        8: 6.15199356450466,
    },
    "opt_alt": {
        1: 20.5579035745177,
        2: 14.400869887002,
        4: 9.16809933900368,
        8: 6.94303813349688,
    },
    "stencil_opt": {
        1: 7.04705964197638,
        2: 6.29154288303107,
        4: 5.37645771951065,
        8: 5.29214795047301,
    },
    "stencil_opt_alt": {
        1: 7.80255457400926,
        2: 6.73089460900519,
        4: 5.62021182349417,
        8: 5.43463719051215,
    },
}

# Add CUDA GPU under explicit label
cpu_variants_times["CUDA GPU"] = cuda_gpu_times

# ---------------------------
# Build DataFrame
# ---------------------------
def build_df_from_variant(name, times_dict):
    rows = []
    for x in xpu_counts:
        t = times_dict.get(int(x), np.nan)
        rows.append({"xpu": int(x), "time": float(t) if not np.isnan(t) else np.nan, "variant": name})
    return pd.DataFrame(rows)

df_list = [build_df_from_variant(v, times) for v, times in cpu_variants_times.items()]
df = pd.concat(df_list, ignore_index=True)

# Remove variants with <= 1 valid datapoint
valid_counts = df.dropna(subset=["time"]).groupby("variant").size()
variants_to_keep = valid_counts[valid_counts > 1].index.tolist()
df = df[df["variant"].isin(variants_to_keep)].reset_index(drop=True)

# Map variant names to display names
def map_display_name(v):
    if v == "CUDA GPU":
        return "CUDA GPU"
    return DISPLAY_NAME.get(v, f"CPU {v}")

df["variant"] = df["variant"].map(map_display_name)

# Compute baseline (time at xpu=1) per variant and derived metrics
baselines = df[df["xpu"] == 1].set_index("variant")["time"].to_dict()
df["baseline_time"] = df["variant"].map(baselines)
df["speedup"] = df["baseline_time"] / df["time"]
df.loc[df["time"].isna(), "speedup"] = np.nan
df["efficiency_pct"] = (df["speedup"] / df["xpu"]) * 100.0

# ---------------------------
# Prepare ordered list for plotting
# ---------------------------
# Translate DESIRED_CPU_ORDER keys 
ordered_variants = []
for key in DESIRED_CPU_ORDER:
    disp = map_display_name(key)
    if disp in df["variant"].unique():
        ordered_variants.append(disp)

# Add CUDA GPU
if "CUDA GPU" in df["variant"].unique():
    ordered_variants.append("CUDA GPU")

# Append any remaining variants 
remaining = [v for v in df["variant"].unique() if v not in ordered_variants]
ordered_variants.extend(sorted(remaining))

# ---------------------------
# Plot: Time vs xPU (using ordered variants)
# ---------------------------
plt.figure(figsize=(7, 4.5))
for variant in ordered_variants:
    group = df[df["variant"] == variant]
    valid = group.dropna(subset=["time"])
    if valid.empty:
        continue
    plt.plot(valid["xpu"], valid["time"], marker="o", linewidth=2, label=variant)

plt.xlabel("xPU Count")
plt.ylabel("Time (s)")
# plt.title("Strong Scaling: Time vs xPU Count")
plt.xticks(xpu_counts, xpu_labels)
plt.legend(ncol=1, loc="upper right")  # unchanged
plt.tight_layout()
plt.savefig("figs/scaling_time.png", dpi=300)

# ---------------------------
# Plot: Speedup vs xPU
# ---------------------------
plt.figure(figsize=(7, 4.5))
for variant in ordered_variants:
    group = df[df["variant"] == variant]
    valid = group.dropna(subset=["speedup"])
    if valid.empty:
        continue
    plt.plot(valid["xpu"], valid["speedup"], marker="o", linewidth=2, label=variant)

plt.xlabel("xPU Count")
plt.ylabel("Speedup")
# plt.title("Strong Scaling: Speedup vs xPU Count")
plt.xticks(xpu_counts, xpu_labels)
plt.legend(ncol=1, loc="upper left")  # moved to upper left
plt.tight_layout()
plt.savefig("figs/scaling_speedup.png", dpi=300)

# ---------------------------
# Plot: Efficiency (%) vs xPU
# ---------------------------
plt.figure(figsize=(7, 4.5))
for variant in ordered_variants:
    group = df[df["variant"] == variant]
    valid = group.dropna(subset=["efficiency_pct"])
    if valid.empty:
        continue
    plt.plot(valid["xpu"], valid["efficiency_pct"], marker="o", linewidth=2, label=variant)

plt.xlabel("xPU Count")
plt.ylabel("Efficiency (\%)")
# plt.title("Strong Scaling: Efficiency vs xPU Count")
plt.xticks(xpu_counts, xpu_labels)
plt.ylim(0, None)
plt.legend(ncol=1, loc="upper right")  # set to upper right
plt.tight_layout()
plt.savefig("figs/scaling_efficiency.png", dpi=300)

# Save CSV summary
df.to_csv("logs/xpu_scaling_summary.csv", index=False)
