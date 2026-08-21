"""Regenerates every figure, into figures/ and benchmarks/zalesak/.

Each figure is drawn from the medians recorded in the benchmarks result
notes, which are quoted as literals here so the figures are reproducible
without rerunning the benchmarks. `fig_overshoot` and `fig_zalesak` additionally
run the filter locally, so `make VERSION=filter` must have been run.

Usage:
    cd src && make VERSION=filter && python3 ../tools/make_figures.py

Figures are written to ../figures; set FIGDIR to override.

Authors:
    Stefanie Boersig <stefanie.boersig@env.ethz.ch>
    Boaz Ko <boazko@student.ethz.ch>
    Ben Bullinger <ben.bullinger@inf.ethz.ch>
"""

import math
import os
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
FIGURES = os.environ.get("FIGDIR", os.path.join(HERE, "..", "figures"))
os.makedirs(FIGURES, exist_ok=True)
sys.path.insert(0, HERE)
from read_field import read_field  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})


def run(args, tag, binary="./stencil2d-filter.x"):
    """Runs a filter binary and returns the output field.

    Args:
        args: Command-line arguments for the binary.
        tag: Suffix for the saved output file.
        binary: Path to the executable, relative to src/.

    Returns:
        The interior of the resulting field.
    """
    subprocess.run(
        ["mpirun", "-n", "1", binary, "--nx", "128", "--ny", "128",
         "--nz", "4", "--num_iter", "100"] + args,
        cwd=SRC, check=True, stdout=subprocess.DEVNULL)
    os.replace(os.path.join(SRC, "out_field.dat"),
               os.path.join(SRC, f"fig_{tag}.dat"))


def fig_overshoot():
    """Plots a cross-section showing the unlimited filter leaving [0, 1]."""
    run(["--order", "4"], "unlim")
    os.replace(os.path.join(SRC, "in_field.dat"),
               os.path.join(SRC, "fig_ic.dat"))
    run(["--order", "4", "--limiter"], "lim")
    run(["--order", "2"], "o2")

    ic, _ = read_field(os.path.join(SRC, "fig_ic.dat"))
    un, _ = read_field(os.path.join(SRC, "fig_unlim.dat"))
    li, _ = read_field(os.path.join(SRC, "fig_lim.dat"))
    o2, _ = read_field(os.path.join(SRC, "fig_o2.dat"))
    k, j = 2, 64  # mid k-level, row through the box center
    x = np.arange(ic.shape[-1])
    u, lim = un[k, j], li[k, j]

    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=200)

    # bounds of the initial data
    ax.axhline(0, color="0.35", lw=0.5)
    ax.axhline(1, color="0.35", lw=0.5)

    # context: initial condition and the smeared 2nd-order alternative
    h_ic, = ax.plot(x, ic[k, j], color="0.15", lw=0.7)
    h_o2, = ax.plot(x, o2[k, j], color="0.65", lw=0.9, ls=(0, (2, 1.2)))

    # bound violations of the unlimited filter: filled red lobes
    ax.fill_between(x, u, 1.0, where=(u > 1.0), color="C3", alpha=0.55, lw=0)
    ax.fill_between(x, u, 0.0, where=(u < 0.0), color="C3", alpha=0.55, lw=0)

    # unlimited below, limited drawn on top: red appears only where they differ
    h_un, = ax.plot(x, u, color="C3", lw=1.2)
    h_li, = ax.plot(x, lim, color="C0", lw=1.4)

    ax.annotate("overshoot", xy=(35.5, 1.075), xytext=(46, 1.06),
                fontsize=6, style="italic", color="C3", va="center",
                arrowprops=dict(arrowstyle="-", color="C3", lw=0.5))
    ax.annotate("undershoot", xy=(28.5, -0.045), xytext=(37, -0.075),
                fontsize=6, style="italic", color="C3", va="center",
                arrowprops=dict(arrowstyle="-", color="C3", lw=0.5))

    ax.set_xlim(22, 106)
    ax.set_ylim(-0.12, 1.14)
    ax.set_xlabel("$x$ (gridpoints)", fontsize=8)
    # y-axis title horizontal above the axis (house style: no rotated text)
    ax.set_title(r"$\phi$", loc="left", fontsize=8, pad=3)
    ax.tick_params(labelsize=7)
    ax.legend([h_ic, h_o2, h_un, h_li],
              ["initial (box)", "$n{=}2$, 100 steps",
               "$n{=}4$ unlimited", "$n{=}4$ limited"],
              fontsize=6, loc="center", frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig-overshoot.pdf"))

    for tag in ["ic", "unlim", "lim", "o2"]:
        os.remove(os.path.join(SRC, f"fig_{tag}.dat"))
    print("fig-overshoot.pdf: unlimited bounds", u.min(), u.max(),
          "| limited bounds", lim.min(), lim.max())


def fig_tradeoff():
    """Plots the accuracy-cost tradeoff across filter orders."""
    orders = [2, 4, 6, 8]
    # Residual damping of the (pi/4, pi/4) mode per step.
    L = 2 * 4 * math.sin(math.pi / 8) ** 2
    resid = {n: (1 / 8 ** (n // 2)) * L ** (n // 2) for n in orders}

    # Medians from benchmarks/orders/results.md
    scan_off = {2: 33.5, 4: 63.0, 6: 82.3, 8: 100.0}
    scan_lim = {2: 55.5, 4: 88.8, 6: 108.3, 8: 119.7}
    # Anchors at nx=512, n=4; other orders scaled by the order scan.
    t_off = {n: 276.6 * scan_off[n] / scan_off[4] for n in orders}
    t_lim = {n: 484.5 * scan_lim[n] / scan_lim[4] for n in orders}

    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=200)
    for tdict, lbl, c, m in [(t_off, "unlimited", "C3", "o"),
                             (t_lim, "limited (monotone)", "C0", "s")]:
        xs = [tdict[n] for n in orders]
        ys = [resid[n] for n in orders]
        ax.plot(xs, ys, "-" + m, color=c, ms=3.5, lw=0.9, label=lbl)
        # Interior labels go upper-right; n=8 drops below its line.
        for n in orders:
            if n == orders[-1]:
                xo, ha = ((4, -9), "left") if c == "C3" else ((-4, -9), "right")
            else:
                xo, ha = (4, 3), "left"
            ax.annotate(f"$n{{=}}{n}$", (tdict[n], resid[n]),
                        textcoords="offset points", xytext=xo, ha=ha,
                        fontsize=6, color=c)
    ax.margins(x=0.07, y=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("runtime per 1024 iterations [ms]")
    # y-axis title horizontal above the axis (house style: no rotated text)
    ax.set_title("damping of resolved $8\\Delta x$ wave per step "
                 "($1-\\lambda$)",
                 loc="left", fontsize=7.5, pad=3)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig-tradeoff.pdf"))
    print("fig-tradeoff.pdf: residual damping",
          {n: f"{resid[n]:.2e}" for n in orders})


def fig_roofline():
    """Plots the roofline placement of the order-4 variants."""
    data = [
        ("naive",   "off",   1.125, 641, 0.32, (9, -2),  "left",   "center"),
        ("naive",   "IF",    1.224, 723, 0.38, (0, 8),   "center", "bottom"),
        ("Zalesak", "",      2.873, 2102, 0.31, (-4, 1),  "right",  "center"),
        ("k-block", "off",   0.382,  87, 2.4,  (-4, -5), "right",  "top"),
        ("inline",  "off",   0.311,  78, 2.7,  (-5, 0),  "right",  "center"),
        ("k-block", "merge", 0.509,  83, 3.3,  (0, 11),  "center", "bottom"),
        ("inline",  "merge", 0.492,  62, 4.5,  (10, 0),  "left",   "center"),
        ("k-block", "IF",    0.767,  83, 3.3,  (-7, -4), "right",  "center"),
        ("inline",  "IF",    0.954,  63, 4.4,  (3, -2),  "left",   "center"),
    ]
    BW = 512.0      # GB/s, nominal LPDDR5X per 72-core Grace socket
    PEAK = 3571.2   # SP Gflop/s, one 72-core Grace socket
    color = {"off": "C3", "IF": "C0", "merge": "C2", "": "C1"}
    marker = {"naive": "o", "k-block": "s", "inline": "^", "Zalesak": "D"}
    # the best usable (limited/monotone) configuration is starred
    best = ("inline", "merge")

    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=200)
    xr = np.logspace(np.log10(0.16), np.log10(12), 100)
    ax.plot(xr, np.minimum(PEAK, BW * xr), color="0.3", lw=0.9)
    ax.text(0.5, 700, "DRAM roof\n512 GB/s", fontsize=6, color="0.3",
            ha="left", va="bottom")
    ax.text(11, 3800, "SP peak 3571 Gflop/s", fontsize=6, color="0.3",
            ha="right", va="bottom")
    for var, mode, t, gb, ai, off, ha, va in data:
        gflops = ai * gb / t  # flops = AI x bytes
        if (var, mode) == best:
            # Star marks the best configuration, in its mode colour.
            ax.plot(ai, gflops, "*", color=color[mode], ms=14, mew=0.8,
                    mec="black", zorder=5)
        else:
            ax.plot(ai, gflops, marker[var], color=color[mode], ms=4,
                    mew=0.5, mec="black")
        ax.annotate(f"{var} {mode}".strip(), (ai, gflops),
                    textcoords="offset points", xytext=off, ha=ha, va=va,
                    fontsize=5.2, color=color[mode])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.16, 12)  # extra left room for the Zalesak label
    ax.set_ylim(100, 6000)
    ax.set_xlabel("arithmetic intensity [flop/B]")
    # y-axis title horizontal above the axis (house style: no rotated text)
    ax.set_title("attained performance [Gflop/s]",
                 loc="left", fontsize=7.5, pad=3)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig-roofline.pdf"))
    print("fig-roofline.pdf: points",
          {f"{v}/{m}": round(ai * gb / t) for v, m, t, gb, ai, *_ in data})


def fig_domainscan():
    """Plots array-demotion speedup against problem size."""
    sizes = [128, 256, 512, 1024]
    naive = [62.4, 145.2, 556.1, 4644.9]
    kblock = [58.8, 109.4, 334.0, 1659.6]
    inline = [55.5,  98.1, 276.6, 1400.5]
    sp_i = [n / v for n, v in zip(naive, inline)]
    sp_k = [n / v for n, v in zip(naive, kblock)]

    fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=200)
    ax.axhline(1.0, color="0.6", lw=0.6, ls=(0, (3, 2)))
    ax.plot(sizes, sp_i, "-^", color="C2", ms=4.5, lw=1.0,
            label="inline (scalar fluxes)")
    ax.plot(sizes, sp_k, "-s", color="C0", ms=4, lw=1.0,
            label="k-block (2D planes)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlim(110, 1180)
    ax.set_ylim(0.9, 3.6)
    ax.set_xlabel("domain size (per horizontal dimension)")
    # y-axis title horizontal above the axis (house style: no rotated text)
    ax.set_title("speedup over naive 3D-temporary layout",
                 loc="left", fontsize=7.5, pad=3)
    # regime labels kept out of the crowded left end -> in the caption
    ax.annotate("cache-resident\n($\\approx$1$\\times$)", (128, 1.09),
                textcoords="offset points", xytext=(6, 14), fontsize=5.5,
                color="0.35", ha="left",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))
    ax.text(1024, 3.42, "DRAM-resident", fontsize=5.5, color="0.35",
            ha="right", va="top")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig-domainscan.pdf"))
    print("fig-domainscan.pdf: inline speedup", [round(s, 2) for s in sp_i],
          "| k-block", [round(s, 2) for s in sp_k])


def fig_gt4py():
    """Plots GT4Py CPU and GPU backends against hand-tuned Fortran."""
    labels = ["$n{=}4$\noff", "$n{=}4$\nlim", "$n{=}8$\noff", "$n{=}8$\nlim"]
    fort = [0.455, 0.966, 0.679, 1.195]
    gcpu = [1.87, 2.48, 3.86, 3.96]
    ggpu = [0.0142, 0.0144, 0.0167, 0.0158]
    x = np.arange(len(labels))
    w = 0.27

    fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=200)
    ax.bar(x - w, fort, w, color="0.4", label="Fortran (1 Grace core)")
    ax.bar(x, gcpu, w, color="C0", label="GT4Py, CPU (1 core)")
    ax.bar(x + w, ggpu, w, color="C3", label="GT4Py, GPU (1 H100)")
    ax.set_yscale("log")
    ax.set_ylim(0.008, 22)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    # y-axis title horizontal above the axis (house style: no rotated text)
    ax.set_title("time per 128 iterations [s]", loc="left", fontsize=7.5, pad=3)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=5.6, frameon=False, loc="upper left", ncol=1,
              handlelength=1.2, borderpad=0.2, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig-gt4py.pdf"))
    print("fig-gt4py.pdf: gpu limiter cost n4 off/lim",
          ggpu[0], ggpu[1], "| cpu", gcpu[0], gcpu[1])


def fig_zalesak():
    """Plots the Zalesak and simple-limiter contours and shape metrics."""
    run(["--order", "4", "--limiter"], "simple")
    run(["--order", "4"], "zalesak", binary="./stencil2d-filter-zalesak.x")
    si, _ = read_field(os.path.join(SRC, "fig_simple.dat"))
    za, _ = read_field(os.path.join(SRC, "fig_zalesak.dat"))
    k = 2
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.9), dpi=200, sharey=True)
    panels = [(axs[0], za, "Zalesak"), (axs[1], si, "simple limiter")]
    for ax, f, title in panels:
        ax.contour(f[k], levels=np.linspace(0.1, 0.9, 9),
                   colors="k", linewidths=0.4)
        ax.set_title(title, fontsize=7)
        ax.set_aspect("equal")
        ax.set_xlim(24, 104)
        ax.set_ylim(24, 104)
        ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "benchmarks", "zalesak",
                             "fig-zalesak.pdf"))
    # Half-widths: centre to the phi=0.5 crossing, axis and diagonal.
    c = 63.5

    def halfwidth(f, dy, dx):
        ts = np.arange(0, 60, 0.25)
        ys, xs = c + ts * dy, c + ts * dx
        vals = f[np.round(ys).astype(int), np.round(xs).astype(int)]
        below = np.nonzero(vals < 0.5)[0]
        return ts[below[0]] if len(below) else float("nan")

    for name, f in [("zalesak", za[k]), ("simple", si[k])]:
        ax_w = halfwidth(f, 0, 1)
        dg_w = halfwidth(f, 2 ** -0.5, 2 ** -0.5)
        print(f"  {name}: halfwidth axis {ax_w:.2f} diag {dg_w:.2f} "
              f"ratio {dg_w / ax_w:.3f}")
    print("  max |simple - zalesak| =", float(np.abs(si - za).max()),
          "| L2 =", float(np.sqrt(((si - za) ** 2).mean())))
    for tag in ["simple", "zalesak"]:
        os.remove(os.path.join(SRC, f"fig_{tag}.dat"))
    os.remove(os.path.join(SRC, "in_field.dat"))
    print("fig-zalesak.pdf (benchmarks/zalesak/): bounds zalesak",
          za.min(), za.max(),
          "| simple", si.min(), si.max())


if __name__ == "__main__":
    fig_overshoot()
    fig_tradeoff()
    fig_roofline()
    fig_domainscan()
    fig_gt4py()
    fig_zalesak()
