from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchmarks.plot_runtime import (
    BASELINE_COLOR,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    GT4PY_COLOR,
    YASMIN_COLOR,
    _load_pyplot,
)

PROBLEMS = (
    ("laplacian", "Laplacian"),
    ("advection_diffusion", "Advection–Diffusion"),
)


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    color: str
    marker: str
    zorder: int


SERIES_SPECS = {
    "yasmin_openmp": SeriesSpec(
        label="Yasmin-OpenMP",
        color=YASMIN_COLOR,
        marker="^",
        zorder=3,
    ),
    "gt4py_cpu": SeriesSpec(
        label="GT4Py-CPU",
        color=GT4PY_COLOR,
        marker="o",
        zorder=2,
    ),
    "cpp_openmp": SeriesSpec(
        label="C++-OpenMP",
        color=BASELINE_COLOR,
        marker="v",
        zorder=1,
    ),
}


# Draw bottom-to-top so Yasmin remains visible when curves overlap.
SERIES_ORDER = (
    "cpp_openmp",
    "gt4py_cpu",
    "yasmin_openmp",
)

# Display top-to-bottom in the legend.
LEGEND_ORDER = (
    "yasmin_openmp",
    "gt4py_cpu",
    "cpp_openmp",
)


def load_scaling_data(
    data_dir: Path,
    problem: str,
    mode: Literal["strong", "weak"] = "strong",
) -> tuple[int, int, dict[str, list[tuple[int, float]]]]:
    data: dict[str, list[tuple[int, float]]] = {}
    sizes: dict[int, tuple[int, int]] = {}
    seen: set[tuple[str, int]] = set()

    path = data_dir / f"{problem}_{mode}_scaling.csv"

    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            implementation = row["implementation"].strip()
            nx = int(row["nx"])
            ny = int(row["ny"])
            threads = int(row["threads"])
            runtime = float(row["runtime_ms"])

            if implementation not in SERIES_SPECS:
                raise ValueError(f"Unexpected scaling implementation: {implementation}")

            if row["correct"] != "true":
                raise ValueError(
                    f"Incorrect result for {implementation}, {threads} threads"
                )

            if nx < 2 or ny < 2 or threads < 1 or runtime <= 0:
                raise ValueError(
                    "Scaling requires positive sizes, thread counts and runtimes"
                )

            if (implementation, threads) in seen:
                raise ValueError(
                    f"Duplicate result for {implementation}, {threads} threads"
                )

            seen.add((implementation, threads))

            size = (nx, ny)

            if threads in sizes and sizes[threads] != size:
                raise ValueError(
                    "Implementations must use the same size at each thread count"
                )

            sizes[threads] = size
            data.setdefault(implementation, []).append((threads, runtime))

    if 1 not in sizes:
        raise ValueError("Missing one-thread grid size")

    base_nx, base_ny = sizes[1]

    for threads, size in sizes.items():
        expected = (
            (base_nx, base_ny * threads) if mode == "weak" else (base_nx, base_ny)
        )

        if size != expected:
            raise ValueError(
                f"Unexpected grid size for {mode} scaling at {threads} threads"
            )

    for implementation, points in data.items():
        points.sort()

        if points[0][0] != 1:
            raise ValueError(f"Missing one-thread baseline for {implementation}")

    return base_nx, base_ny, data


def _plot_scaling_axis(
    *,
    axis: Any,
    data: dict[str, list[tuple[int, float]]],
    title: str,
    mode: Literal["strong", "weak"],
) -> None:
    all_threads = sorted(
        {threads for points in data.values() for threads, _runtime in points}
    )

    for implementation in SERIES_ORDER:
        points = data.get(implementation)
        if points is None:
            continue

        spec = SERIES_SPECS[implementation]

        threads = [count for count, _runtime in points]
        runtime = [value for _threads, value in points]

        scaling = [
            runtime[0] / value * (100 if mode == "weak" else 1) for value in runtime
        ]

        axis.plot(
            threads,
            scaling,
            marker=spec.marker,
            color=spec.color,
            linewidth=2,
            label=spec.label,
            zorder=spec.zorder,
        )

    ideal = (
        [100.0] * len(all_threads)
        if mode == "weak"
        else [float(thread) for thread in all_threads]
    )

    axis.plot(
        all_threads,
        ideal,
        "--",
        color="0.35",
        linewidth=1.5,
        label="Ideal",
        zorder=0,
    )

    axis.set_title(title)
    axis.set_xlabel("Threads")
    axis.set_ylabel("Efficiency (%)" if mode == "weak" else "Speedup")

    if mode == "weak":
        axis.set_xscale("log", base=2)

    axis.set_xticks(all_threads)
    axis.set_xticklabels([str(thread) for thread in all_threads])

    axis.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    handles, labels = axis.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles, strict=True))

    legend_labels = [
        SERIES_SPECS[implementation].label for implementation in LEGEND_ORDER
    ] + ["Ideal"]

    axis.legend(
        [handle_by_label[label] for label in legend_labels],
        legend_labels,
        loc="lower left" if mode == "weak" else "upper left",
        frameon=True,
    )


def plot_scaling(
    *,
    data_dir: Path,
    output_dir: Path,
    mode: Literal["strong", "weak"] = "strong",
) -> Path:
    data_by_problem = {
        problem: load_scaling_data(
            data_dir,
            problem,
            mode,
        )[2]
        for problem, _title in PROBLEMS
    }

    plt = _load_pyplot()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    for axis, (problem, title) in zip(
        axes,
        PROBLEMS,
        strict=True,
    ):
        _plot_scaling_axis(
            axis=axis,
            data=data_by_problem[problem],
            title=title,
            mode=mode,
        )

    figure.subplots_adjust(
        wspace=0.28,
    )

    output_path = output_dir / f"{mode}_scaling.png"

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CPU scaling.")
    parser.add_argument(
        "--mode",
        choices=("strong", "weak"),
        default="strong",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    output_path = plot_scaling(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
    )

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
