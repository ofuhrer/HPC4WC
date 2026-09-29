#!/usr/bin/env python3
# benchmarking/fullbenchmark_cpu.py
import argparse, csv, json, os, re, shlex, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import patheffects as pe

# Resolve repo root so the script works from any CWD
ROOT = Path(__file__).resolve().parent.parent

# ------------------- configuration -------------------
MODELS = [
    # name, binary path, is_parallel
    ("naive",               "build-cpu/test_mnist",                 False),
    ("stencil",             "build-cpu/test_mnist_stencil",         False),
    ("opt",                 "build-cpu/test_mnist_opt",              True),
    ("opt_alt",             "build-cpu/test_mnist_opt_alt",          True),
    ("stencil_opt",         "build-cpu/test_mnist_stencil_opt",      True),
    ("stencil_opt_alt",     "build-cpu/test_mnist_stencil_alt",      True),
]

# Human-friendly display names (output-only labels)
DISPLAY_NAME = {
    "naive":             "Naive - serial",
    "stencil":           "Stencil - serial",
    "opt":               "Naive — parallelized + optimized",
    "opt_alt":           "Naive — simple parallelization",
    "stencil_opt":       "Stencil — parallelized + optimized",
    "stencil_opt_alt":   "Stencil — simple parallelization",
}

DEFAULT_THREADS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]  # clamped to available cores/Slurm cpus-per-task
WARMUPS = 1
REPEATS = 10
DATA_REQUIRED = ["data/mnist/images.npy", "data/mnist/labels.npy"]

# Plot config to match plot_full_cpu.py
DPI = 400
FIGSIZE_SE = (6.0, 4.0)   # Speedup & Efficiency
FIGSIZE_BT = (7.0, 4.0)   # Best end-to-end & Compute-time-with-IQR

# test binaries print:  RESULT compute_time_sec=<float> samples=<int>
TIME_RE = re.compile(r"compute_time_sec\s*=\s*([0-9]*\.?[0-9]+)")
ACC_RE  = re.compile(r"accuracy\s*=\s*([0-9]*\.?[0-9]+)")

# -----------------------------------------------------

def sh(msg): print(msg, flush=True)

def clamp_threads(tlist):
    avail = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or (os.cpu_count() or 1)
    return [t for t in tlist if t <= max(1, avail)]

def libtorch_fallback(env):
    torch_dir = env.get("Torch_DIR")
    try:
        if torch_dir:
            lib = Path(torch_dir).parents[2] / "lib"
            if lib.is_dir():
                env["LD_LIBRARY_PATH"] = str(lib) + (":" + env["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in env else "")
        else:
            home_lib = Path.home() / "libtorch-cpu" / "lib"
            if home_lib.is_dir():
                env["LD_LIBRARY_PATH"] = str(home_lib) + (":" + env["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in env else "")
    except Exception:
        pass
    return env

def set_env_for_run(threads, samples):
    env = os.environ.copy()
    # OpenMP and math libs
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "false"
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    # test control
    env["MNIST_SAMPLES"] = str(samples)
    env["MNIST_QUIET"] = "1"
    # runtime fallback (optional)
    env = libtorch_fallback(env)
    return env

def ensure_prereqs():
    for name, path, _ in MODELS:
        p = ROOT / path
        if not (p.is_file() and os.access(p, os.X_OK)):
            sys.exit(f"Missing binary: {path} for model '{name}' (looked at: {p})")
    for rel in DATA_REQUIRED:
        p = ROOT / rel
        if not p.is_file():
            sys.exit(f"Missing dataset file: {rel} (looked at: {p})")

def try_parse_compute_time(stdout_text):
    m = TIME_RE.search(stdout_text)
    return float(m.group(1)) if m else None

def try_parse_accuracy(stdout_text):
    m = ACC_RE.search(stdout_text)
    return float(m.group(1)) if m else None

def run_once(bin_relpath, threads, samples):
    env = set_env_for_run(threads, samples)
    bin_exec = ROOT / bin_relpath
    t0 = time.monotonic()
    try:
        proc = subprocess.run([str(bin_exec)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, check=True, cwd=str(ROOT))
        wall = time.monotonic() - t0
        comp = try_parse_compute_time(proc.stdout)
        acc  = try_parse_accuracy(proc.stdout)
        return True, wall, comp, acc
    except subprocess.CalledProcessError:
        wall = time.monotonic() - t0
        return False, wall, None, None

def warmup(bin_relpath, threads, samples):
    for _ in range(WARMUPS):
        run_once(bin_relpath, threads, samples)

def collect_meta():
    def git(cmd):
        try:
            return subprocess.check_output(shlex.split(cmd), text=True, cwd=str(ROOT)).strip()
        except Exception:
            return "unknown"
    def cmd_out(txt):
        try:
            return subprocess.check_output(txt, shell=True, text=True, cwd=str(ROOT)).strip()
        except Exception:
            return ""
    meta = dict(
        git_commit=git("git rev-parse HEAD"),
        git_status=git("git status --porcelain"),
        hostname=os.uname().nodename if hasattr(os, "uname") else "",
        start_iso=datetime.now().isoformat(timespec="seconds"),
        cpu_count=os.cpu_count(),
        slurm_cpus=os.environ.get("SLURM_CPUS_PER_TASK", ""),
        omp=dict(
            OMP_PROC_BIND=os.environ.get("OMP_PROC_BIND", ""),
            OMP_PLACES=os.environ.get("OMP_PLACES", ""),
        ),
        lscpu=cmd_out("LC_ALL=C lscpu | sed -n '1,20p'"),
    )
    return meta

def summarize(df, value_col):
    """
    Returns per-model, per-thread stats with median, q1, q3, plus speedup and efficiency.
    Baseline is the median at the minimum observed thread count for each model.
    """
    # Ensure threads is integer-typed for robust grouping/sorting
    df = df.copy()
    df["threads"] = df["threads"].astype(int)

    g = df.groupby(["model", "threads"])[value_col]
    stats = g.agg(
        median="median",
        q1=lambda x: np.percentile(x.dropna(), 25) if len(x.dropna()) else np.nan,
        q3=lambda x: np.percentile(x.dropna(), 75) if len(x.dropna()) else np.nan
    ).reset_index()

    outs = []
    for m, sub in stats.groupby("model", as_index=False):
        # sort and choose baseline strictly at the smallest threads present
        sub = sub.sort_values("threads").reset_index(drop=True)
        if sub.empty:
            continue

        # baseline = median at threads == min(threads)
        tmin = int(sub["threads"].min())
        base_row = sub.loc[sub["threads"] == tmin]
        base = float(base_row["median"].iloc[0]) if not base_row["median"].isna().iloc[0] else np.nan

        # Compute speedup/efficiency safely
        with np.errstate(divide="ignore", invalid="ignore"):
            speedup = base / sub["median"] if np.isfinite(base) else np.nan
            efficiency = speedup / sub["threads"]

        sub["speedup"] = speedup
        sub["efficiency"] = efficiency
        outs.append(sub)

    if not outs:
        return pd.DataFrame(columns=["model","threads","median","q1","q3","speedup","efficiency"])

    return pd.concat(outs, ignore_index=True)


def pretty(name: str) -> str:
    """Map internal model name to human-friendly label for output/plots."""
    return DISPLAY_NAME.get(name, name)

# -------- helpers for nicer ticks/limits (used only in plots 2 & 4) --------
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
    # Ensure consistent " — " dash with single spaces around it
    return re.sub(r"\s*[-—]\s*", " — ", label).strip()

def main():
    parser = argparse.ArgumentParser(description="Full CPU benchmark (statistically meaningful)")
    parser.add_argument("--threads", nargs="*", type=int, default=DEFAULT_THREADS,
                        help="Thread sweep for parallel models")
    parser.add_argument("--samples", type=int, default=int(os.environ.get("MNIST_SAMPLES", "50000")),
                        help="MNIST samples per run")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help="Repeats per (model,threads)")
    parser.add_argument("--outdir", default=None,
                        help="Optional explicit output directory")
    args = parser.parse_args()

    ensure_prereqs()

    threads = clamp_threads(args.threads)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    base = Path(__file__).resolve().parent / "results" / f"{ts}_full_cpu"
    if args.outdir:
        base = Path(args.outdir)
    base.mkdir(parents=True, exist_ok=True)

    # metadata
    meta = collect_meta()
    meta.update(dict(kind="full", warmups=WARMUPS, repeats=args.repeats, samples=args.samples, threads=threads))
    (base / "meta.json").write_text(json.dumps(meta, indent=2))

    rows = []  # model, threads, run_id, total_time_s, compute_time_s, accuracy

    sh("Full CPU benchmark")
    sh(f"- samples per run : {args.samples}")
    sh(f"- repeats per point: {args.repeats}")
    sh(f"- thread sweep     : {threads}\n")

    for name, bin_path, is_par in MODELS:
        tlist = [1] if not is_par else threads
        sh(f"== {pretty(name)} ==")
        for t in tlist:
            warmup(bin_path, t, args.samples)
            for r in range(args.repeats):
                ok, wall, comp, acc = run_once(bin_path, t, args.samples)
                rows.append(dict(
                    model=name, threads=t, run_id=r,
                    total_time_s=(wall if ok else np.nan),
                    compute_time_s=(comp if ok and comp is not None else np.nan),
                    accuracy=(acc if ok and acc is not None else np.nan),
                ))
                # nicer aligned terminal line
                if ok:
                    comp_txt = f"{comp:.3f}s" if comp == comp else "n/a"
                    acc_txt  = f"{acc:.3f}"   if acc  == acc  else "n/a"
                    sh(f"  T={t:>2d}  run {r+1:02d}/{args.repeats}    wall={wall:>7.3f}s    compute={comp_txt:>8}    acc={acc_txt:>6}")
                else:
                    sh(f"  T={t:>2d}  run {r+1:02d}/{args.repeats}    FAILED")
        sh("")

    # write raw CSV (keeps internal model names for reproducibility)
    raw_csv = base / "results.csv"
    with raw_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "threads", "run_id", "total_time_s", "compute_time_s", "accuracy"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    df = pd.read_csv(raw_csv)

    # compute summaries
    comp_stats = summarize(df.dropna(subset=["compute_time_s"]), "compute_time_s")
    totl_stats = summarize(df.dropna(subset=["total_time_s"]), "total_time_s")
    comp_stats.to_csv(base / "summary_compute.csv", index=False)
    totl_stats.to_csv(base / "summary_total.csv", index=False)

    # --------- plots (adapted to match plot_full_cpu.py) ----------
    # 1) compute time vs threads with IQR error bars + dots (parallel models)
    par_models = [m for m, _, p in MODELS if p]
    comp_par = comp_stats[comp_stats.model.isin(par_models)]
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
                elinewidth=1.2, alpha=1.0, label=pretty(m), zorder=3  # thicker & opaque for overlap
            )
        ax.set_xlabel("Threads")
        ax.set_ylabel("Compute Time [s]")
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend()
        fig.savefig(base / "time_compute_iqr.png", bbox_inches="tight", dpi=DPI)

    # 2) speedup vs threads — dots only, no legend
    fig, ax = plt.subplots()
    fig.set_size_inches(*FIGSIZE_SE)
    if not comp_par.empty:
        for m, sub in comp_par.groupby("model"):
            sub = sub.sort_values("threads")
            ax.plot(sub["threads"], sub["speedup"], marker="o", markersize=3, linestyle="None")
        max_speed = comp_par["speedup"].replace([np.inf, -np.inf], np.nan).max()
    else:
        max_speed = 1.0
    ymax = max(1.0, _nice_ceiling(float(max_speed) * 1.10 if np.isfinite(max_speed) else 1.0))
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_nice_ticks))
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.grid(True, alpha=0.3)
    fig.savefig(base / "speedup_compute.png", bbox_inches="tight", dpi=DPI)

    # 3) efficiency vs threads — dots only, legend kept
    fig, ax = plt.subplots()
    fig.set_size_inches(*FIGSIZE_SE)
    if not comp_par.empty:
        for m, sub in comp_par.groupby("model"):
            sub = sub.sort_values("threads")
            ax.plot(sub["threads"], sub["efficiency"], marker="o", markersize=3, linestyle="None", label=pretty(m))
    ax.set_xlabel("Threads")
    ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(base / "efficiency_compute.png", bbox_inches="tight", dpi=DPI)

    # 4) best end-to-end (total) time per model — full box and in-bar T
    best_total = totl_stats.loc[totl_stats.groupby("model")["median"].idxmin()].sort_values("median").copy()
    best_total["pretty"] = best_total["model"].map(pretty)
    xlabels = [_normalize_label(lbl) for lbl in best_total["pretty"]]

    fig, ax = plt.subplots()
    fig.set_size_inches(*FIGSIZE_BT)
    bars = ax.bar(xlabels, best_total["median"])

    # full box (all spines visible)
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

    fig.savefig(base / "best_total_per_model.png", bbox_inches="tight", dpi=DPI)

    # Print short terminal summary with pretty names
    sh("\nBest end-to-end (total) time per model:")
    for _, row in best_total.iterrows():
        sh(f"{row['pretty']:>35s}  {row['median']:>7.3f} s  at T={int(row['threads'])}")

    sh(f"\nSaved results to: {base}")
    sh("Done.")

if __name__ == "__main__":
    main()
