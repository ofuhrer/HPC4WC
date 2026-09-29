#!/usr/bin/env python3
# benchmarking/plot_full_cpu.py
# Recreate all CPU plots from the latest full benchmark run (no execution, just CSV -> figures).

import os
import sys
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import patheffects as pe
from pathlib import Path

# ---------------------------------------------
# Config
# ---------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarking" / "results"
DPI = 400  # as requested

# Consistent sizes across paired plots
FIGSIZE_SE = (6.0, 4.0)   # Speedup & Efficiency
FIGSIZE_BT = (7.0, 4.0)   # Best end-to-end & Compute-time-with-IQR

# Human-friendly display names (output-only labels)
DISPLAY_NAME = {
    "naive":             "Naive - serial",
    "stencil":           "Stencil - serial",
    "opt":               "Naive — parallelized + optimized",
    "opt_alt":           "Naive — simple parallelization",
    "stencil_opt":       "Stencil — parallelized + optimized",
    "stencil_opt_alt":   "Stencil — simple parallelization",
}

# ---------------------------------------------
# Helpers
# ---------------------------------------------
def pretty(name: str) -> str:
    return DISPLAY_NAME.get(name, name)

def find_latest_full_cpu_dir() -> Path:
    if not RESULTS_DIR.is_dir():
        sys.exit(f"Missing directory: {RESULTS_DIR}")
    # candidates look like 'YYYYMMDD_HHMM_full_cpu'
    dirs = [p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name.endswith("_full_cpu")]
    if not dirs:
        sys.exit(f"No '*_full_cpu' result directories found in {RESULTS_DIR}")
    dirs.sort(key=lambda p: p.name)
    return dirs[-1]

def summarize(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = df.groupby(["model", "threads"])[value_col]
    stats = g.agg(
        median="median",
        q1=lambda x: np.percentile(x.dropna(), 25) if len(x.dropna()) else np.nan,
        q3=lambda x: np.percentile(x.dropna(), 75) if len(x.dropna()) else np.nan,
    ).reset_index()

    outs = []
    for m, sub in stats.groupby("model"):
        sub = sub.sort_values("threads").copy()
        if not len(sub):
            continue
        base = sub["median"].iloc[0]
        sub["speedup"] = base / sub["median"]
        sub["efficiency"] = sub["speedup"] / sub["threads"]
        outs.append(sub)
    if not outs:
        return pd.DataFrame(columns=["model","threads","median","q1","q3","speedup","efficiency"])
    return pd.concat(outs, ignore_index=True)

def load_or_compute_summaries(run_dir: Path):
    raw_csv = run_dir / "results.csv"
    if not raw_csv.is_file():
        sys.exit(f"Missing CSV: {raw_csv}")

    df = pd.read_csv(raw_csv)
    comp_path = run_dir / "summary_compute.csv"
    totl_path = run_dir / "summary_total.csv"

    comp = pd.read_csv(comp_path) if comp_path.is_file() else summarize(df.dropna(subset=["compute_time_s"]), "compute_time_s")
    totl = pd.read_csv(totl_path) if totl_path.is_file() else summarize(df.dropna(subset=["total_time_s"]), "total_time_s")

    return df, comp, totl

def ensure_outdir(base_out: Path) -> Path:
    base_out.mkdir(parents=True, exist_ok=True)
    return base_out

# -------- helpers for nicer ticks/limits & label normalization --------
def _fmt_nice_ticks(x, _):
    if x == 0:
        return "0"
    if abs(x) >= 100:
        s = f"{x:.0f}"
    elif abs(x) >= 10:
        s = f"{x:.1f}"
    else:
        s = f"{x:.2f}"
    return s.rstrip("0").rstrip(".")

def _nice_ceiling(val):
    if val <= 1: step = 0.1
    elif val <= 2: step = 0.2
    elif val <= 5: step = 0.5
    elif val <= 10: step = 1.0
    else: step = 2.0
    return float(np.ceil(val / step) * step)

def _normalize_label(label: str) -> str:
    return re.sub(r"\s*[-—]\s*", " — ", label).strip()

# ---------------------------------------------
# Main plotting
# ---------------------------------------------
def main():
    run_dir = find_latest_full_cpu_dir()
    outdir = ensure_outdir(ROOT / "benchmarking" / "figs" / run_dir.name)

    print(f"[plot] Using run directory : {run_dir}")
    print(f"[plot] Saving figures to  : {outdir}")

    # Load data
    df, comp_stats, totl_stats = load_or_compute_summaries(run_dir)

    # Identify parallel models heuristically: those observed with >1 thread
    par_models = []
    for m, sub in comp_stats.groupby("model"):
        if sub["threads"].max() > 1:
            par_models.append(m)
    comp_par = comp_stats[comp_stats["model"].isin(par_models)].copy()

    # ---- 1) compute time vs threads with IQR error bars + lines ----
    if not comp_par.empty:
        fig, ax = plt.subplots()
        fig.set_size_inches(*FIGSIZE_BT)
        for m, sub in comp_par.groupby("model"):
            sub = sub.sort_values("threads")
            y = sub["median"].values
            yerr = np.vstack([
                np.maximum(0.0, y - sub["q1"].values),
                np.maximum(0.0, sub["q3"].values - y)
            ])
            ax.errorbar(
                sub["threads"], y, yerr=yerr,
                fmt="o-", markersize=3, capsize=4, capthick=1.2,
                elinewidth=1.2, alpha=1.0,
                label=pretty(m), zorder=3
            )
        ax.set_xlabel("Threads")
        ax.set_ylabel("Compute Time [s]")
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend()
        fig.savefig(outdir / "time_compute_iqr.png", bbox_inches="tight", dpi=DPI)

    # ---- 2) speedup vs threads — lines (no legend) ----
    fig, ax = plt.subplots()
    fig.set_size_inches(*FIGSIZE_SE)
    xs_all = sorted(comp_par["threads"].unique()) if not comp_par.empty else []
    max_speed = 1.0
    if not comp_par.empty:
        for m, sub in comp_par.groupby("model"):
            sub = sub.sort_values("threads")
            ax.plot(sub["threads"], sub["speedup"], marker="o", markersize=3, linestyle="-")
        max_speed = comp_par["speedup"].replace([np.inf, -np.inf], np.nan).max()
    ymax = max(1.0, _nice_ceiling(float(max_speed) * 1.10 if np.isfinite(max_speed) else 1.0))
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_nice_ticks))
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.grid(True, alpha=0.3)
    fig.savefig(outdir / "speedup_compute.png", bbox_inches="tight", dpi=DPI)

    # ---- 3) efficiency vs threads — lines (legend kept) ----
    fig, ax = plt.subplots()
    fig.set_size_inches(*FIGSIZE_SE)
    if not comp_par.empty:
        for m, sub in comp_par.groupby("model"):
            sub = sub.sort_values("threads")
            ax.plot(sub["threads"], sub["efficiency"], marker="o", markersize=3, linestyle="-", label=pretty(m))
    ax.set_xlabel("Threads")
    ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(outdir / "efficiency_compute.png", bbox_inches="tight", dpi=DPI)

    # ---- 4) best end-to-end (total) time per model — full box ----
    if not totl_stats.empty:
        best_idx = totl_stats.groupby("model")["median"].idxmin()
        best_total = totl_stats.loc[best_idx].sort_values("median").copy()
        best_total["pretty"] = best_total["model"].map(pretty)
        xlabels = [_normalize_label(lbl) for lbl in best_total["pretty"]]

        fig, ax = plt.subplots()
        fig.set_size_inches(*FIGSIZE_BT)
        bars = ax.bar(xlabels, best_total["median"])

        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(True)

        ax.set_ylabel("Best Total Time [s]")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_nice_ticks))
        plt.xticks(rotation=20, ha="right")

        for rect, t in zip(bars, best_total["threads"].astype(int)):
            h = rect.get_height()
            x = rect.get_x() + rect.get_width() / 2
            ax.text(
                x, h * 0.85, f"T={t}",
                ha="center", va="center", color="white", fontsize=9, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")]
            )

        fig.savefig(outdir / "best_total_per_model.png", bbox_inches="tight", dpi=DPI)
    else:
        print("[plot] Warning: total-time summary empty; skipping best_total_per_model plot.")

    print("[plot] Done.")

if __name__ == "__main__":
    main()
