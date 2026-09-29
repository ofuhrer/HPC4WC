# ******************************************************
#     Program: analyze_benchmark_results.py
# Description: Aggregate result_*.py files written by
#              experiment_scaling_all_strategies.sh and
#              experiment_mpi_leftright_scaling.sh (including legacy results).
#              Write raw per-run and repeat-averaged CSVs, then PNG plots:
#                1) average communication vs computation time, with one
#                   panel per rank count (requires a timer breakdown)
#                2) relative communication, packing, sending, and waiting
#                   time, separately for each scaling mode; one panel per
#                   rank count >= 8, normalized to the slowest strategy,
#                   with the fastest outlined (requires the relevant timers)
#                3) time vs rank count for each scaling mode
#                4) strong-scaling speedup relative to the smallest rank
#                   count, with an ideal-scaling reference
#                5) weak-scaling efficiency relative to the smallest rank
#                   count, with an ideal reference of 1
#
#              --metric total_time (default) uses wall time for plots 3-5;
#              --metric comm uses average communication time and prefixes
#              their filenames with "comm_". For comm, weak-scaling
#              efficiency is a communication-time ratio. Rows missing the
#              selected timer are omitted. Scaling series require a fixed
#              global size (strong) or fixed local size (weak).
#
#              --results-dir is repeatable to combine datasets. For example,
#              regenerate the report's strong-scaling corner plots with:
#
#                python3 analyze_benchmark_results.py \
#                    --results-dir results/corners/results_strong_1024_detailed_timing \
#                    --results-dir results/corners/results_corners_strong_1024_timed_n96 \
#                    --results-dir results/corners/results_mpi_leftright_strong_1024_n96 \
#                    --series-from strategy --mode strong \
#                    --csv-out plots/corners/plots_strong_1024_detailed_timing/runs.csv \
#                    --agg-csv-out plots/corners/plots_strong_1024_detailed_timing/aggregated.csv \
#                    --out-dir plots/corners/plots_strong_1024_detailed_timing
#
#              By default, series are "<experiment>/<strategy>" so cases
#              from different directories stay separate. Use
#              --series-from strategy to merge same-named strategies.
#              Repeats are averaged within each configuration; different
#              problem sizes remain separate. Filter with repeatable
#              --experiment, --strategy, --mode, and --ranks options.
#              --out-dir selects the PNG directory (default: current directory);
#              --csv-out and --agg-csv-out independently select the CSV paths.
# ******************************************************
import csv
import fnmatch
import glob
import math
import os
from collections import defaultdict

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAW_FIELDS = [
    "series",
    "experiment",
    "strategy",
    "mode",
    "rep",
    "nodes",
    "ranks",
    "cores_per_rank",
    "nx",
    "ny",
    "nz",
    "num_iter",
    "local_nx",
    "local_ny",
    "total_time",
    "max_comm",
    "max_compute",
    "avg_comm",
    "avg_compute",
    "avg_pack",
    "avg_send",
    "avg_wait",
    # provenance -- what code, what command, when. Empty for pre-lib_experiment
    # runs whose provenance could not be recovered.
    "binary",
    "command",
    "env_overrides",
    "srun_extra",
    "app_extra",
    "git_commit",
    "git_branch",
    "git_dirty",
    "slurm_jobid",
    "t_start",
    "t_start_approx",
    "wall_s",
    "metadata_source",
    "source_file",
]

AGG_FIELDS = [
    "series",
    "experiment",
    "strategy",
    "mode",
    "nodes",
    "ranks",
    "cores_per_rank",
    "nx",
    "ny",
    "nz",
    "num_iter",
    "local_nx",
    "local_ny",
    "n_reps",
    "total_time_mean",
    "total_time_std",
    "avg_comm_mean",
    "avg_comm_std",
    "avg_compute_mean",
    "avg_compute_std",
    "avg_pack_mean",
    "avg_pack_std",
    "avg_send_mean",
    "avg_send_std",
    "avg_wait_mean",
    "avg_wait_std",
    "avg_actual_comm_mean",
    "avg_actual_comm_std",
    "git_commit",
    "git_dirty",
    "command",
]


def load_result_file(path, series_from="experiment"):
    """Each result_*.py is real Python: 'data = np.array([[...]])' plus a
    metadata trailer appended by lib_experiment.sh's run_case() -- strategy,
    nodes, ranks, cores_per_rank, mode, rep, and the provenance fields
    (binary, command, env_overrides, git_commit, slurm_jobid, t_start,
    wall_s, ...).

    Files written before lib_experiment.sh existed carry only the first six
    keys; every provenance field falls back to "" so old and new runs can
    sit in the same CSV. Missing provenance does not prevent plotting.

    Not every stencil2d version prints the same number of columns: the
    default (stencil2d-mpi.F90) and the three corners variants print 13 --
    [ranks, nx, ny, nz, num_iter, total_time, max_comm, max_compute,
    avg_comm, avg_compute, avg_pack, avg_send, avg_wait], where
    avg_pack + avg_send + avg_wait sums back to avg_comm (pack folds in both
    packing and unpacking; send is time issuing Irecv/Isend; wait is time
    blocked in Waitall/Waitany). Other full timer versions (comm_strats_timer)
    print 10, without that sub-breakdown. Older total-time-only versions
    print 6 -- [ranks, nx, ny, nz, num_iter, total_time]. Missing fields
    become NaN so all three kinds can live in the same CSV.
    """
    ns = {"np": np}
    with open(path) as f:
        exec(f.read(), ns)  # noqa: S102 -- trusted, self-generated files

    row = list(np.asarray(ns["data"])[0])
    ranks_field, nx, ny, nz, num_iter, total_time = row[:6]
    if len(row) >= 10:
        max_comm, max_compute, avg_comm, avg_compute = row[6:10]
    else:
        max_comm = max_compute = avg_comm = avg_compute = float("nan")
    if len(row) >= 13:
        avg_pack, avg_send, avg_wait = row[10:13]
    else:
        avg_pack = avg_send = avg_wait = float("nan")

    strategy = ns.get("strategy", "unknown")
    # The experiment defaults to the containing directory, which is what the
    # pre-lib_experiment files have to fall back on.
    experiment = ns.get("experiment") or os.path.basename(os.path.dirname(os.path.abspath(path)))

    def opt_int(key):
        v = ns.get(key)
        return int(v) if v not in (None, "") else -1

    def opt_float(key):
        v = ns.get(key)
        return float(v) if v not in (None, "") else float("nan")

    return {
        "series": strategy if series_from == "strategy" else f"{experiment}/{strategy}",
        "experiment": experiment,
        "strategy": strategy,
        "mode": ns.get("mode", "strong"),
        "rep": int(ns.get("rep", 1)),
        "nodes": int(ns.get("nodes", -1)),
        "ranks": int(ns.get("ranks", ranks_field)),
        "cores_per_rank": int(ns.get("cores_per_rank", -1)),
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "num_iter": int(num_iter),
        "local_nx": opt_int("local_nx"),
        "local_ny": opt_int("local_ny"),
        "total_time": float(total_time),
        "max_comm": float(max_comm),
        "max_compute": float(max_compute),
        "avg_comm": float(avg_comm),
        "avg_compute": float(avg_compute),
        "avg_pack": float(avg_pack),
        "avg_send": float(avg_send),
        "avg_wait": float(avg_wait),
        "binary": ns.get("binary", ""),
        "command": ns.get("command", ""),
        "env_overrides": ns.get("env_overrides", ""),
        "srun_extra": ns.get("srun_extra", ""),
        "app_extra": ns.get("app_extra", ""),
        "git_commit": ns.get("git_commit", ""),
        "git_branch": ns.get("git_branch", ""),
        "git_dirty": ns.get("git_dirty", ""),
        "slurm_jobid": ns.get("slurm_jobid", ""),
        "t_start": ns.get("t_start", ""),
        "t_start_approx": opt_int("t_start_approx"),
        "wall_s": opt_float("wall_s"),
        "metadata_source": ns.get("metadata_source", "legacy"),
        "source_file": path,
    }


def write_raw_csv(rows, csv_path):
    rows = sorted(rows, key=lambda r: (r["mode"], r["series"], r["ranks"], r["nx"], r["rep"]))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def aggregate(rows):
    """Average over repeats only -- i.e. over runs that are identical in
    every respect except the rep index. Std is NaN with a single rep;
    that's expected, not a bug.

    The problem size is PART OF THE KEY. Without it, two runs at the same
    rank count but different local domains (say the 64x64 ceiling runs and
    a 128x128 weak-scaling sweep) would silently average into one
    meaningless row -- which is exactly what happens as soon as you merge
    several results_*/ directories.
    """
    groups = defaultdict(list)
    for r in rows:
        key = (
            r["series"],
            r["mode"],
            r["nodes"],
            r["ranks"],
            r["cores_per_rank"],
            r["nx"],
            r["ny"],
            r["nz"],
            r["num_iter"],
        )
        groups[key].append(r)

    agg_rows = []
    for (series, mode, nodes, ranks, cores, _nx, _ny, _nz, _ni), group in groups.items():
        total_times = [g["total_time"] for g in group]
        comms = [g["avg_comm"] for g in group if not math.isnan(g["avg_comm"])]
        computes = [g["avg_compute"] for g in group if not math.isnan(g["avg_compute"])]
        packs = [g["avg_pack"] for g in group if not math.isnan(g["avg_pack"])]
        sends = [g["avg_send"] for g in group if not math.isnan(g["avg_send"])]
        waits = [g["avg_wait"] for g in group if not math.isnan(g["avg_wait"])]
        # "actual communication" = everything that isn't local buffer
        # marshalling -- send (posting Irecv/Isend) + wait (blocked in
        # Waitall/Waitany). Computed per run before averaging, not as a
        # difference of means, so its std reflects real run-to-run variance.
        actual_comms = [
            g["avg_send"] + g["avg_wait"]
            for g in group
            if not math.isnan(g["avg_send"]) and not math.isnan(g["avg_wait"])
        ]

        commits = sorted({g["git_commit"] for g in group if g["git_commit"]})
        agg_rows.append(
            {
                "series": series,
                "experiment": group[0]["experiment"],
                "strategy": group[0]["strategy"],
                "mode": mode,
                "nodes": nodes,
                "ranks": ranks,
                "cores_per_rank": cores,
                "nx": group[0]["nx"],
                "ny": group[0]["ny"],
                "nz": group[0]["nz"],
                "num_iter": group[0]["num_iter"],
                "local_nx": group[0]["local_nx"],
                "local_ny": group[0]["local_ny"],
                "git_commit": ",".join(commits),
                "git_dirty": group[0]["git_dirty"],
                "command": group[0]["command"],
                "n_reps": len(group),
                "total_time_mean": float(np.mean(total_times)),
                "total_time_std": float(np.std(total_times)) if len(total_times) > 1 else float("nan"),
                "avg_comm_mean": float(np.mean(comms)) if comms else float("nan"),
                "avg_comm_std": float(np.std(comms)) if len(comms) > 1 else float("nan"),
                "avg_compute_mean": float(np.mean(computes)) if computes else float("nan"),
                "avg_compute_std": float(np.std(computes)) if len(computes) > 1 else float("nan"),
                "avg_pack_mean": float(np.mean(packs)) if packs else float("nan"),
                "avg_pack_std": float(np.std(packs)) if len(packs) > 1 else float("nan"),
                "avg_send_mean": float(np.mean(sends)) if sends else float("nan"),
                "avg_send_std": float(np.std(sends)) if len(sends) > 1 else float("nan"),
                "avg_wait_mean": float(np.mean(waits)) if waits else float("nan"),
                "avg_wait_std": float(np.std(waits)) if len(waits) > 1 else float("nan"),
                "avg_actual_comm_mean": float(np.mean(actual_comms)) if actual_comms else float("nan"),
                "avg_actual_comm_std": float(np.std(actual_comms)) if len(actual_comms) > 1 else float("nan"),
            }
        )
    return sorted(agg_rows, key=lambda r: (r["mode"], r["series"], r["ranks"], r["nx"]))


def select_rows(rows, experiments=(), strategies=(), modes=(), ranks=()):
    """Select run records before aggregation.

    Experiment and strategy selectors use shell-style patterns, making e.g.
    ``--strategy 'corners*'`` useful without relying on filename conventions.
    Repeated selectors are ORed within a field and fields are ANDed together.
    """
    return [
        row
        for row in rows
        if (not experiments or any(fnmatch.fnmatchcase(row["experiment"], p) for p in experiments))
        and (not strategies or any(fnmatch.fnmatchcase(row["strategy"], p) for p in strategies))
        and (not modes or row["mode"] in modes)
        and (not ranks or row["ranks"] in ranks)
    ]


def write_agg_csv(agg_rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        writer.writeheader()
        writer.writerows(agg_rows)


def plot_comm_vs_compute(agg_rows, out_path):
    """Only meaningful for strategies with the full 10-col timer output."""
    rows = [r for r in agg_rows if not math.isnan(r["avg_comm_mean"])]
    if not rows:
        print("Skipping comm-vs-compute plot: no rows with a comm/compute breakdown.")
        return

    strategies = sorted(set(r["series"] for r in rows))
    rank_counts = sorted(set(r["ranks"] for r in rows))

    ncols = len(rank_counts)
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    x = np.arange(len(strategies))
    width = 0.35

    for ax, ranks in zip(axes, rank_counts):
        comm, comm_err, compute, compute_err = [], [], [], []
        for s in strategies:
            match = [r for r in rows if r["series"] == s and r["ranks"] == ranks]
            comm.append(match[0]["avg_comm_mean"] if match else np.nan)
            comm_err.append(match[0]["avg_comm_std"] if match else 0)
            compute.append(match[0]["avg_compute_mean"] if match else np.nan)
            compute_err.append(match[0]["avg_compute_std"] if match else 0)

        ax.bar(x - width / 2, comm, width, yerr=comm_err, capsize=3, label="avg comm")
        ax.bar(x + width / 2, compute, width, yerr=compute_err, capsize=3, label="avg compute")
        ax.set_title(f"ranks = {ranks}")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.set_ylabel("time [s]")
        ax.legend()

    fig.suptitle("Average communication vs. computation time per strategy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# Which pair of aggregate columns a plot draws from, selected by --metric.
# "comm" isolates communication cost -- it only exists for runs with the
# full 10-column timer output, so rows lacking avg_comm_mean are dropped
# before plotting rather than plotted as a false zero.
METRICS = {
    "total_time": dict(mean="total_time_mean", std="total_time_std", ylabel="total time [s]", noun="Total wall time"),
    "comm": dict(mean="avg_comm_mean", std="avg_comm_std", ylabel="avg communication time [s]", noun="Average communication time"),
}


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot_comm_relative_bars(agg_rows, out_path, mode):
    """One subplot per rank count actually run, each a bar chart of avg comm
    time normalized to the slowest strategy AT THAT RANK COUNT (slowest bar
    = 1.0). This answers "is one strategy best across every rank config" in
    a way the overlaid line plot can't: line-plot series sit within a few
    percent of each other in absolute time, so their error bars stack and
    hide which is ahead; per-rank relative bars make the ranking at each
    point unambiguous even when the absolute gap is small, and the fastest
    bar (not necessarily rank 0 in x) is called out with an outline.
    """
    rows = [r for r in agg_rows if r["mode"] == mode and not math.isnan(r["avg_comm_mean"])]
    if not rows:
        print(f"Skipping comm-relative-bars plot ({mode}): no rows with a comm/compute breakdown.")
        return

    strategies = sorted(set(r["series"] for r in rows))
    rank_counts = sorted(set(r["ranks"] for r in rows if r["ranks"] >= 8))
    if not rank_counts:
        print(f"Skipping {timer}-relative-bars plot ({mode}): no rank counts >= 8.")
        return

    ncols = len(rank_counts)
    fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 4.5), sharey=True, squeeze=False)
    axes = axes[0]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(len(strategies))

    for ax, ranks in zip(axes, rank_counts):
        means, stds, n_reps = [], [], []
        for s in strategies:
            match = [r for r in rows if r["series"] == s and r["ranks"] == ranks]
            means.append(match[0]["avg_comm_mean"] if match else np.nan)
            stds.append(match[0]["avg_comm_std"] if match else np.nan)
            n_reps.append(match[0]["n_reps"] if match else 0)
        means = np.array(means)
        valid = ~np.isnan(means)
        if not valid.any():
            ax.set_title(f"ranks = {ranks}\n(no data)")
            continue
        slowest = np.nanmax(means)
        rel = means / slowest
        rel_err = [0 if math.isnan(e) else e / slowest for e in stds]
        bar_colors = [colors[i % len(colors)] for i in range(len(strategies))]
        bars = ax.bar(x, rel, yerr=rel_err, capsize=3, color=bar_colors)

        fastest_idx = int(np.nanargmin(means))
        bars[fastest_idx].set_edgecolor("black")
        bars[fastest_idx].set_linewidth(2.2)

        for b, m, n in zip(bars, means, n_reps):
            if not np.isnan(m):
                ax.text(b.get_x() + b.get_width() / 2, 0.02, f"{m:.3g}s, n={n}", rotation=90,
                        ha="center", va="bottom", fontsize=7, color="white")

        ax.set_title(f"ranks = {ranks}")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.3, linewidth=1)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("avg comm time, relative to slowest strategy")
    fig.suptitle(
        f"Communication time per strategy, relative to the slowest at each rank count ({mode} scaling)\n"
        "black outline = fastest strategy at that rank count; bar label = actual seconds and number of runs"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timer_relative_bars(agg_rows, out_path, mode, timer):
    """Plot one detailed communication timer relative to its slowest
    strategy at each rank count, using the same layout as the communication
    relative-bar plot.
    """
    timer_columns = {
        "packing": ("avg_pack_mean", "avg_pack_std"),
        "sending": ("avg_send_mean", "avg_send_std"),
        "waiting": ("avg_wait_mean", "avg_wait_std"),
    }
    mean_col, std_col = timer_columns[timer]
    rows = [r for r in agg_rows if r["mode"] == mode and not math.isnan(r[mean_col])]
    if not rows:
        print(f"Skipping {timer}-relative-bars plot ({mode}): no detailed timer data.")
        return

    strategies = sorted(set(r["series"] for r in rows))
    rank_counts = sorted(set(r["ranks"] for r in rows if r["ranks"] >= 8))
    if not rank_counts:
        print(f"Skipping {timer}-relative-bars plot ({mode}): no rank counts >= 8.")
        return
    fig, axes = plt.subplots(
        1, len(rank_counts), figsize=(2.6 * len(rank_counts), 4.5),
        sharey=True, squeeze=False,
    )
    axes = axes[0]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(len(strategies))

    for ax, ranks in zip(axes, rank_counts):
        means, stds, n_reps = [], [], []
        for strategy in strategies:
            match = [r for r in rows if r["series"] == strategy and r["ranks"] == ranks]
            means.append(match[0][mean_col] if match else np.nan)
            stds.append(match[0][std_col] if match else np.nan)
            n_reps.append(match[0]["n_reps"] if match else 0)
        means = np.asarray(means)
        if not (~np.isnan(means)).any():
            continue
        slowest = np.nanmax(means)
        rel = means / slowest
        rel_err = [0 if math.isnan(e) else e / slowest for e in stds]
        bars = ax.bar(
            x, rel, yerr=rel_err, capsize=3,
            color=[colors[i % len(colors)] for i in range(len(strategies))],
        )
        fastest_idx = int(np.nanargmin(means))
        bars[fastest_idx].set_edgecolor("black")
        bars[fastest_idx].set_linewidth(2.2)
        for bar, mean, n in zip(bars, means, n_reps):
            if not np.isnan(mean):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, 0.02, f"{mean:.3g}s, n={n}",
                    rotation=90, ha="center", va="bottom", fontsize=7,
                    color="white",
                )
        ax.set_title(f"ranks = {ranks}")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.3, linewidth=1)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(f"avg {timer} time, relative to slowest strategy")
    fig.suptitle(
        f"{timer.capitalize()} time per strategy, relative to the slowest at each rank count ({mode} scaling)\n"
        "black outline = fastest strategy at that rank count; bar label = actual seconds and number of runs"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_total_time_scaling(agg_rows, out_path, mode, metric="total_time"):
    m = METRICS[metric]
    rows = [r for r in agg_rows if r["mode"] == mode and not math.isnan(r[m["mean"]])]
    if not rows:
        return
    strategies = sorted(set(r["series"] for r in rows))

    fig, ax = plt.subplots(figsize=(7, 5))
    # Series often sit within their own error bars of each other (e.g. comm
    # time across corners/pipelined/waitany/mpi_leftright), so markers and
    # error bars would otherwise stack exactly on top of one another and be
    # unreadable as anything but one blob. Nudge each series a few percent
    # apart in log-x space (dodge) and give it its own marker shape -- the
    # dodge is purely visual, real x stays at the true rank count via the
    # unshifted gridlines/ticks.
    n = len(strategies)
    spread = 0.28  # total dodge width, in log2(ranks) units
    for i, s in enumerate(strategies):
        series_rows = [r for r in rows if r["series"] == s]
        problem_sizes = {(r["nx"], r["ny"], r["nz"]) for r in series_rows}
        local_sizes = {(r["local_nx"], r["local_ny"], r["nz"]) for r in series_rows}
        valid = len(problem_sizes) == 1 if mode == "strong" else len(local_sizes) == 1
        if not valid:
            print(f"Skipping incompatible {mode}-scaling series {s!r}: problem-size convention is not constant.")
            continue
        pts = sorted(
            [(r["ranks"], r[m["mean"]], r[m["std"]]) for r in series_rows],
            key=lambda t: t[0],
        )
        if not pts:
            continue
        xs, ys, errs = zip(*pts)
        errs = [0 if math.isnan(e) else e for e in errs]
        offset = spread * (i - (n - 1) / 2) / max(n - 1, 1)
        xs_dodged = [x * (2 ** offset) for x in xs]
        ax.errorbar(
            xs_dodged, ys, yerr=errs, marker=MARKERS[i % len(MARKERS)],
            markersize=6, capsize=3, alpha=0.85, label=s,
        )

    ax.set_xlabel("ranks")
    ax.set_ylabel(m["ylabel"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    # set_xscale resets the tick locator/formatter, so the real (undodged)
    # rank counts as plain-number ticks must be set after it, not before.
    ax.set_xticks(sorted(set(r["ranks"] for r in rows)))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_title(f"{m['noun']} vs. rank count ({mode} scaling)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_strong_scaling_speedup(agg_rows, out_path, metric="total_time"):
    m = METRICS[metric]
    rows = [r for r in agg_rows if r["mode"] == "strong" and not math.isnan(r[m["mean"]])]
    if not rows:
        return
    strategies = sorted(set(r["series"] for r in rows))

    fig, ax = plt.subplots(figsize=(7, 5))
    for s in strategies:
        series_rows = [r for r in rows if r["series"] == s]
        if len({(r["nx"], r["ny"], r["nz"]) for r in series_rows}) != 1:
            continue
        pts = sorted([(r["ranks"], r[m["mean"]]) for r in series_rows], key=lambda t: t[0])
        if len(pts) < 2:
            continue
        base_ranks, base_time = pts[0]
        xs = [p[0] for p in pts]
        speedup = [base_time / p[1] for p in pts]
        ax.plot(xs, speedup, marker="o", label=s)

    if strategies:
        all_ranks = sorted(set(r["ranks"] for r in rows))
        base = all_ranks[0]
        ax.plot(all_ranks, [r / base for r in all_ranks], "k--", alpha=0.5, label="ideal")

    ax.set_xlabel("ranks")
    ax.set_ylabel(f"speedup (relative to smallest rank count)")
    ax.set_xscale("log", base=2)
    ax.set_title(f"Strong-scaling speedup vs. ideal ({m['noun'].lower()})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weak_scaling_efficiency(agg_rows, out_path, metric="total_time"):
    """Note: for the comm metric this is 'comm-time efficiency', not the
    standard parallel-efficiency definition -- it's still base/observed
    against the smallest rank count, just applied to avg_comm instead of
    total_time, which is what makes it comparable to the strong-scaling
    speedup plot's use of the same metric."""
    m = METRICS[metric]
    rows = [r for r in agg_rows if r["mode"] == "weak" and not math.isnan(r[m["mean"]])]
    if not rows:
        return
    strategies = sorted(set(r["series"] for r in rows))

    fig, ax = plt.subplots(figsize=(7, 5))
    for s in strategies:
        series_rows = [r for r in rows if r["series"] == s]
        if len({(r["local_nx"], r["local_ny"], r["nz"]) for r in series_rows}) != 1:
            continue
        pts = sorted([(r["ranks"], r[m["mean"]]) for r in series_rows], key=lambda t: t[0])
        if len(pts) < 2:
            continue
        base_ranks, base_time = pts[0]
        xs = [p[0] for p in pts]
        efficiency = [base_time / p[1] for p in pts]  # ideal weak scaling: flat at 1.0
        ax.plot(xs, efficiency, marker="o", label=s)

    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="ideal (flat)")
    ax.set_xlabel("ranks")
    ylabel = "parallel efficiency" if metric == "total_time" else f"{m['noun'].lower()} efficiency"
    ax.set_ylabel(f"{ylabel} (relative to smallest rank count)")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.15)
    title = "Weak-scaling parallel efficiency" if metric == "total_time" else f"Weak-scaling {m['noun'].lower()} efficiency"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@click.command()
@click.option(
    "--results-dir",
    "results_dirs",
    type=str,
    multiple=True,
    default=("results",),
    help="Directory containing result_*.py files. Repeat to merge several experiments into one comparison.",
)
@click.option(
    "--series-from",
    type=click.Choice(["experiment", "strategy"]),
    default="experiment",
    help=(
        "How to name a plot series. 'experiment' (default) gives '<dir>/<label>', which keeps the "
        "same-named cases of different experiments apart; 'strategy' uses the bare label, which "
        "deliberately merges them."
    ),
)
@click.option("--experiment", "experiments", multiple=True, help="Select experiment names (shell pattern; repeatable).")
@click.option("--strategy", "strategies", multiple=True, help="Select strategy names (shell pattern; repeatable).")
@click.option("--mode", "modes", type=click.Choice(["strong", "weak"]), multiple=True, help="Select scaling mode (repeatable).")
@click.option("--ranks", "rank_counts", type=int, multiple=True, help="Select a rank count (repeatable).")
@click.option("--csv-out", type=str, default="comm_strategy_results.csv", help="Path to write the raw per-run CSV.")
@click.option("--agg-csv-out", type=str, default="comm_strategy_results_agg.csv", help="Path to write the rep-averaged CSV.")
@click.option("--out-dir", type=str, default=".", help="Directory to write plot PNGs into.")
@click.option(
    "--metric",
    type=click.Choice(["total_time", "comm"]),
    default="total_time",
    help=(
        "Which timing column the scaling/speedup/efficiency plots use. 'total_time' (default) is "
        "wall time; 'comm' isolates avg communication time and drops rows without a 10-column timer "
        "breakdown (the aggregate CSV always contains both columns regardless of this choice)."
    ),
)
def main(results_dirs, series_from, experiments, strategies, modes, rank_counts, csv_out, agg_csv_out, out_dir, metric):
    paths = []
    for d in results_dirs:
        found = sorted(glob.glob(os.path.join(d, "result_*.py")))
        if not found:
            raise click.ClickException(f"No result_*.py files found in {d}/")
        print(f"{d}: {len(found)} result files")
        paths.extend(found)

    rows = [load_result_file(p, series_from=series_from) for p in paths]
    rows = select_rows(rows, experiments, strategies, modes, rank_counts)
    if not rows:
        raise click.ClickException("No run records matched the selection filters.")
    os.makedirs(out_dir, exist_ok=True)
    for output in (csv_out, agg_csv_out):
        parent = os.path.dirname(os.path.abspath(output))
        os.makedirs(parent, exist_ok=True)
    rows = write_raw_csv(rows, csv_out)
    print(f"Wrote {len(rows)} raw rows to {csv_out}")

    # Non-overlapping implementations should partition the timed work into
    # communication and computation. The small remainder is timer/loop overhead.
    bad_sums = []
    for r in rows:
        if math.isnan(r["avg_comm"]) or math.isnan(r["avg_compute"]):
            continue
        remainder = r["total_time"] - r["avg_comm"] - r["avg_compute"]
        tolerance = max(1.0e-6, 0.02 * r["total_time"])
        if abs(remainder) > tolerance:
            bad_sums.append((r, remainder))
    if bad_sums:
        print(f"\n!! {len(bad_sums)} run(s) failed total ~= avg_comm + avg_compute (2% tolerance):")
        for r, remainder in bad_sums:
            print(f"   {r['source_file']}: residual={remainder:.4e} s")

    agg_rows = aggregate(rows)
    write_agg_csv(agg_rows, agg_csv_out)
    print(f"Wrote {len(agg_rows)} aggregated rows to {agg_csv_out}")

    width = max((len(r["series"]) for r in agg_rows), default=16)
    for r in agg_rows:
        std_note = f" (+/- {r['total_time_std']:.2e})" if not math.isnan(r["total_time_std"]) else ""
        dirty = " [DIRTY TREE]" if str(r["git_dirty"]) == "1" else ""
        print(
            f"  [{r['mode']:>6s}] {r['series']:>{width}s}  ranks={r['ranks']:<4d} "
            f"{r['nx']}x{r['ny']}x{r['nz']}  "
            f"total={r['total_time_mean']:.4e}{std_note}  n_reps={r['n_reps']}{dirty}"
        )

    missing = [r for r in rows if not r["command"]]
    if missing:
        print(
            f"\n!! {len(missing)} of {len(rows)} runs have no recorded command/commit "
            f"Provenance fields remain empty where unavailable; plotting continues."
        )

    plot_comm_vs_compute(agg_rows, os.path.join(out_dir, "comm_vs_compute_avg.png"))

    # File names carry the metric only when it isn't the default, so a
    # total_time run and a comm run can share one --out-dir without clobbering.
    metric_prefix = "" if metric == "total_time" else "comm_"

    modes_present = sorted(set(r["mode"] for r in agg_rows))
    for mode in modes_present:
        plot_comm_relative_bars(agg_rows, os.path.join(out_dir, f"comm_relative_bars_{mode}.png"), mode)
        for timer in ("packing", "sending", "waiting"):
            plot_timer_relative_bars(
                agg_rows, os.path.join(out_dir, f"{timer}_relative_bars_{mode}.png"), mode, timer
            )
    for mode in modes_present:
        plot_total_time_scaling(
            agg_rows, os.path.join(out_dir, f"{metric_prefix}total_time_scaling_{mode}.png"), mode, metric=metric
        )

    if "strong" in modes_present:
        plot_strong_scaling_speedup(
            agg_rows, os.path.join(out_dir, f"{metric_prefix}strong_scaling_speedup.png"), metric=metric
        )
    if "weak" in modes_present:
        plot_weak_scaling_efficiency(
            agg_rows, os.path.join(out_dir, f"{metric_prefix}weak_scaling_efficiency.png"), metric=metric
        )

    print(f"Wrote plots to {out_dir}/")
    print(
        "\nNote: 'total_time_mean' in the aggregated CSV is the number to diff against a "
        "comm/compute-overlap version once you have it -- everything else is strategy-specific."
    )


if __name__ == "__main__":
    main()
