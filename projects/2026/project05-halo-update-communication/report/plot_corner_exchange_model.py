"""Draw the illustrative communication model used in Section 3.3.

The values are deliberately dimensionless assumptions, not benchmark data.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

RANKS = np.arange(4)
COLORS = {
    "pack": "#4C78A8",
    "unpack": "#72B7B2",
    "post": "#F2CF5B",
    "transfer": "#E45756",
    "dependency": "#B8B8B8",
}


def add_segment(ax, bottom, values, color, hatch=None):
    values = np.asarray(values, dtype=float)
    ax.barh(RANKS, values, left=bottom, height=0.68, color=color,
            edgecolor="white", linewidth=0.7, hatch=hatch)
    return bottom + values


fig, axes = plt.subplots(4, 1, figsize=(10.5, 10.8), sharex=True)

# The two panels assign every rank exactly the same total packing, posting,
# and intrinsic data-transfer time. Rank 1's first transfer is deliberately
# the longest. That delays its second post, leaving rank 0 without a matching
# send after rank 0 has already posted its own second exchange.
first_transfer = np.array([0.6, 1.8, 0.9, 1.0])
second_transfer = np.array([1.4, 0.7, 1.2, 1.0])
intrinsic_transfer = first_transfer + second_transfer

# Sequential baseline: top/bottom must finish before left/right is packed.
bottom = np.zeros(4)
bottom = add_segment(axes[0], bottom, [0.6] * 4, COLORS["pack"])
bottom = add_segment(axes[0], bottom, [0.175] * 4, COLORS["post"])
bottom = add_segment(axes[0], bottom, first_transfer, COLORS["transfer"])
bottom = add_segment(axes[0], bottom, [0.25] * 4, COLORS["unpack"])
bottom = add_segment(axes[0], bottom, [0.6] * 4, COLORS["pack"])
second_post_start = bottom.copy()
bottom = add_segment(axes[0], bottom, [0.175] * 4, COLORS["post"], "//")
# Rank 0's gap ends exactly when rank 1 starts its second post.
dependency_stall = np.array([
    max(0.0, second_post_start[1] - bottom[0]), 0.0, 0.0, 0.0
])
stall_base = bottom.copy()
bottom = add_segment(axes[0], bottom, dependency_stall, COLORS["dependency"])
bottom = add_segment(axes[0], bottom, second_transfer, COLORS["transfer"], "//")
bottom = add_segment(axes[0], bottom, [0.25] * 4, COLORS["unpack"], "//")
baseline_max = bottom.max()

# Explicit corners allow one synchronization after all messages are posted.
bottom = np.zeros(4)
bottom = add_segment(axes[1], bottom, [1.2] * 4, COLORS["pack"])
bottom = add_segment(axes[1], bottom, [0.35] * 4, COLORS["post"])
bottom = add_segment(axes[1], bottom, intrinsic_transfer, COLORS["transfer"])
bottom = add_segment(axes[1], bottom, [0.5] * 4, COLORS["unpack"])
corners_max = bottom.max()

# Pipelined corners: only part of posting is hidden behind packing.
bottom = np.zeros(4)
bottom = add_segment(axes[2], bottom, [1.2] * 4, COLORS["pack"])
post_overlap = 0.20
axes[2].barh(RANKS, [post_overlap] * 4,
             left=[1.2 - post_overlap] * 4, height=0.44,
             color=COLORS["post"], edgecolor="white", linewidth=0.7,
             hatch="xx")
bottom = add_segment(axes[2], bottom, [0.35 - post_overlap] * 4,
                     COLORS["post"])
bottom = add_segment(axes[2], bottom, intrinsic_transfer, COLORS["transfer"])
bottom = add_segment(axes[2], bottom, [0.5] * 4, COLORS["unpack"])
pipelined_max = bottom.max()

# Waitany corners: part of unpacking overlaps transfer; the remainder is
# serial and therefore still extends the timeline.
bottom = np.zeros(4)
bottom = add_segment(axes[3], bottom, [1.2] * 4, COLORS["pack"])
bottom = add_segment(axes[3], bottom, [0.35] * 4, COLORS["post"])
transfer_base = bottom.copy()
bottom = add_segment(axes[3], bottom, intrinsic_transfer, COLORS["transfer"])
unpack_overlap = 0.30
axes[3].barh(RANKS, [unpack_overlap] * 4,
             left=transfer_base + intrinsic_transfer - unpack_overlap,
             height=0.44, color=COLORS["unpack"], edgecolor="white",
             linewidth=0.7, hatch="xx")
bottom = add_segment(axes[3], bottom, [0.5 - unpack_overlap] * 4,
                     COLORS["unpack"])
waitany_max = bottom.max()

titles = [
    "Default: sequential edge exchange",
    "Explicit corners: one exchange phase",
    "Pipelined corners: post overlaps pack",
    "Waitany corners: unpack overlaps transfer",
]
maxima = [baseline_max, corners_max, pipelined_max, waitany_max]
for ax, title, maximum in zip(axes, titles, maxima):
    ax.set_title(title, fontsize=10.5)
    ax.set_yticks(RANKS)
    ax.set_yticklabels([f"Rank {rank}" for rank in RANKS])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.axvline(maximum, color="#333333", linestyle="--", linewidth=1.2)
    ax.text(maximum - 0.04, 3.55, "slowest-rank time", rotation=90,
            ha="right", va="bottom", fontsize=8.0, color="#333333")

axes[0].annotate("rank 1's long first transfer\ndelays its second post;\nrank 0 cannot receive yet",
                 xy=(stall_base[0] + dependency_stall[0] * 0.55, 0),
                 xytext=(3.25, 2.0), fontsize=8.2, ha="center",
                 bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                       "edgecolor": "none", "alpha": 0.9},
                 arrowprops={"arrowstyle": "->", "color": "#555555"})
axes[2].annotate("partly overlapped", xy=(1.1, 1),
                 xytext=(1.7, 2.1), fontsize=8.0, ha="center",
                 arrowprops={"arrowstyle": "->", "color": "#555555"})
axes[3].annotate("partly overlapped", xy=(3.85, 1),
                 xytext=(3.0, 2.1), fontsize=8.0, ha="center",
                 arrowprops={"arrowstyle": "->", "color": "#555555"})

axes[3].set_xlabel("Illustrative halo-update time (arbitrary units)")
axes[0].set_xlim(0, 5.8)
legend = [
    Patch(facecolor=COLORS["pack"], label="pack"),
    Patch(facecolor=COLORS["post"], label="post receives and sends"),
    Patch(facecolor=COLORS["transfer"], label="data transfer"),
    Patch(facecolor=COLORS["dependency"], label="matching send not posted"),
    Patch(facecolor=COLORS["unpack"], label="unpack"),
]
fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False,
           fontsize=8.5)
fig.suptitle("Conceptual model of corner-exchange strategies",
             fontsize=12)
fig.text(0.5, 0.075,
         "Narrow hatched regions show the overlapped fraction; the serial remainder still extends the bar.",
         ha="center", fontsize=8.5, color="#444444")
fig.tight_layout(rect=(0, 0.13, 1, 0.96))
fig.savefig("figures/corner_exchange_timing_model.png", dpi=220,
            bbox_inches="tight")
