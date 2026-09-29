from __future__ import annotations

import argparse
import csv
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATA_DIR = Path("benchmarks/outputs/raw/csv")
DEFAULT_OUTPUT_DIR = Path("benchmarks/outputs/plots")

PROBLEMS = (
    ("laplacian", "Laplacian"),
    ("advection_diffusion", "Advection–Diffusion"),
)

YASMIN_COLOR = "#148714"
GT4PY_COLOR = "#a23b72"
BASELINE_COLOR = "#D09A44"

BENCHMARK_SETUP = (
    "Alps (CSCS)\nNVIDIA Grace, 72 cores\nGCC 14.3.0\nPython 3.14.3\nOpenMP 4.5"
)


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    color: str
    marker: str
    zorder: int
    alpha: float


SERIES_SPECS = {
    "yasmin_numpy": SeriesSpec(
        label="Yasmin-NumPy",
        color=YASMIN_COLOR,
        marker="o",
        zorder=3,
        alpha=0.45,
    ),
    "gt4py_numpy": SeriesSpec(
        label="GT4Py-NumPy",
        color=GT4PY_COLOR,
        marker="o",
        zorder=2,
        alpha=0.45,
    ),
    "numpy": SeriesSpec(
        label="NumPy",
        color=BASELINE_COLOR,
        marker="o",
        zorder=1,
        alpha=0.45,
    ),
    "yasmin_cpp": SeriesSpec(
        label="Yasmin-C++",
        color=YASMIN_COLOR,
        marker="s",
        zorder=3,
        alpha=0.70,
    ),
    "gt4py_cpu_single": SeriesSpec(
        label="GT4Py-CPU",
        color=GT4PY_COLOR,
        marker="s",
        zorder=2,
        alpha=0.70,
    ),
    "cpp": SeriesSpec(
        label="C++",
        color=BASELINE_COLOR,
        marker="s",
        zorder=1,
        alpha=0.70,
    ),
    "yasmin_openmp": SeriesSpec(
        label="Yasmin-OpenMP",
        color=YASMIN_COLOR,
        marker="^",
        zorder=3,
        alpha=1.0,
    ),
    "gt4py_cpu_multi": SeriesSpec(
        label="GT4Py-CPU",
        color=GT4PY_COLOR,
        marker="^",
        zorder=2,
        alpha=1.0,
    ),
    "cpp_openmp": SeriesSpec(
        label="C++-OpenMP",
        color=BASELINE_COLOR,
        marker="^",
        zorder=1,
        alpha=1.0,
    ),
}


# Draw bottom-to-top so Yasmin remains visible when curves overlap.
SERIES_ORDER = (
    "numpy",
    "gt4py_numpy",
    "yasmin_numpy",
    "cpp",
    "gt4py_cpu_single",
    "yasmin_cpp",
    "cpp_openmp",
    "gt4py_cpu_multi",
    "yasmin_openmp",
)


# Display Yasmin first in each legend block.
LEGEND_SERIES_BY_GROUP = {
    "NumPy": (
        "yasmin_numpy",
        "gt4py_numpy",
        "numpy",
    ),
    "Single-Core": (
        "yasmin_cpp",
        "gt4py_cpu_single",
        "cpp",
    ),
    "Multi-Core (8 threads)": (
        "yasmin_openmp",
        "gt4py_cpu_multi",
        "cpp_openmp",
    ),
}


def find_problem_csvs(data_dir: Path, problem: str) -> list[Path]:
    csv_files: list[Path] = []

    for path in sorted(data_dir.glob(f"{problem}_*.csv")):
        if "scaling" in path.name:
            continue

        size_part = path.stem[len(problem) + 1 :]
        if size_part.isdigit():
            csv_files.append(path)

    return csv_files


def _series_key(row: dict[str, str]) -> str:
    implementation = row["implementation"].strip()

    if implementation != "gt4py_cpu":
        return implementation

    threads_text = row["threads"].strip()
    if not threads_text:
        raise ValueError("GT4Py CPU runtime row is missing a thread count")

    threads = int(threads_text)

    if threads == 1:
        return "gt4py_cpu_single"

    return "gt4py_cpu_multi"


def load_runtime_data(
    *,
    data_dir: Path,
    problem: str,
) -> dict[str, list[tuple[int, float]]]:
    runtime_by_series: dict[str, list[tuple[int, float]]] = {}

    for csv_path in find_problem_csvs(data_dir, problem):
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["correct"] != "true":
                    continue

                series = _series_key(row)
                nx = int(row["nx"])
                runtime_ms = float(row["runtime_ms"])

                runtime_by_series.setdefault(series, []).append((nx, runtime_ms))

    for values in runtime_by_series.values():
        values.sort(key=lambda item: item[0])

    return runtime_by_series


def _style_box(axis: Any) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_facecolor("white")

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("0.8")
        spine.set_linewidth(0.8)


def plot_runtime(
    *,
    data_dir: Path,
    output_dir: Path,
) -> Path:
    runtime_by_problem = {
        problem: load_runtime_data(
            data_dir=data_dir,
            problem=problem,
        )
        for problem, _title in PROBLEMS
    }

    for problem, runtime_by_series in runtime_by_problem.items():
        if not runtime_by_series:
            raise FileNotFoundError(
                f"No correct runtime CSV data found for {problem!r} in {data_dir}"
            )

        unknown_series = set(runtime_by_series) - set(SERIES_SPECS)
        if unknown_series:
            unknown = ", ".join(sorted(unknown_series))
            raise ValueError(f"Unknown runtime series for {problem!r}: {unknown}")

    plt = _load_pyplot()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    handles_by_series: dict[str, Any] = {}

    for axis, (problem, title) in zip(
        axes,
        PROBLEMS,
        strict=True,
    ):
        runtime_by_series = runtime_by_problem[problem]

        for series in SERIES_ORDER:
            points = runtime_by_series.get(series)
            if points is None:
                continue

            spec = SERIES_SPECS[series]

            nx_values = [nx for nx, _runtime_ms in points]
            runtime_values = [runtime_ms for _nx, runtime_ms in points]

            (line,) = axis.plot(
                nx_values,
                runtime_values,
                marker=spec.marker,
                linewidth=2,
                color=spec.color,
                alpha=spec.alpha,
                zorder=spec.zorder,
            )

            handles_by_series.setdefault(
                series,
                line,
            )

        axis.set_title(title)
        axis.set_xlabel("Grid Size (N×N)")
        axis.set_ylabel("Runtime (ms)")
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.4,
        )

    figure.subplots_adjust(
        right=0.78,
        wspace=0.28,
    )

    box_x = 0.82
    box_width = 0.15
    box_height = 0.18

    box_y_positions = {
        "NumPy": 0.74,
        "Single-Core": 0.53,
        "Multi-Core (8 threads)": 0.32,
        "setup": 0.11,
    }

    for group, series_names in LEGEND_SERIES_BY_GROUP.items():
        box_axis = figure.add_axes(
            [
                box_x,
                box_y_positions[group],
                box_width,
                box_height,
            ]
        )
        _style_box(box_axis)

        available_series = [
            series for series in series_names if series in handles_by_series
        ]

        handles = [handles_by_series[series] for series in available_series]
        labels = [SERIES_SPECS[series].label for series in available_series]

        box_axis.legend(
            handles,
            labels,
            title=group,
            loc="center left",
            bbox_to_anchor=(0.08, 0.5),
            frameon=False,
            borderaxespad=0.0,
            alignment="left",
        )

    setup_axis = figure.add_axes(
        [
            box_x,
            box_y_positions["setup"],
            box_width,
            box_height,
        ]
    )
    _style_box(setup_axis)

    setup_axis.text(
        0.08,
        0.5,
        BENCHMARK_SETUP,
        ha="left",
        va="center",
        fontsize=8,
        family="monospace",
        linespacing=1.3,
        transform=setup_axis.transAxes,
    )

    output_path = output_dir / "runtime.png"

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def _load_pyplot() -> Any:
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        return importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Plotting requires matplotlib. Install the benchmark extras first."
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark runtime results.")
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

    output_path = plot_runtime(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
