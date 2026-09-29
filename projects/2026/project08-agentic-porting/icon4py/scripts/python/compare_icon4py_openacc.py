#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --only-group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the performance of ICON4Py and ICON on a set of stencils and plot the results."""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from typing import Any

import typer


type VariantDescriptor = tuple[str, dict[str, Any]]

cli = typer.Typer()

log = logging.getLogger(__name__)

experiment = "mch_icon-ch1_medium"
target = "GH200 1-rank"

openacc_backend = "openacc"
output_filename = "bench_blueline_stencil_compute"

# the default 'file_prefix' assumes that the json files are in the script folder
file_prefix = pathlib.Path(__file__).parent
openacc_input = file_prefix / "../data/bencher=mch_icon-ch1_medium_stencils=0.939946=OPENACC.json"
gt4py_input = {}
gt4py_metrics = ["compute"]  # here we can add other metrics, e.g. 'total'
gt4py_unmatched_ncalls_threshold = (
    2  # ignore unmatched icon4py stencils if less than this threshold
)

# Mapping from fortran stencil to gt4py stencil variants. The mapped value contains,
# besides the gt4py stencil name, a dictionary of static arguments that should be
# matched in the gt4py timer report. If the value is `None`, we do not check the
# static arguments and assume the stencil name is the same.
fortran_to_icon4py: dict[str, VariantDescriptor | None] = {
    "apply_diffusion_to_theta_and_exner": None,
    "apply_diffusion_to_vn": None,
    "apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence": None,
    "apply_divergence_damping_and_update_vn": None,
    "boundary_halo_cleanup": None,
    "compute_dwdz_and_boundary_update_rho_theta_w": None,
    "calculate_diagnostic_quantities_for_turbulence": None,
    "calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools": None,
    "calculate_nabla2_and_smag_coefficients_for_vn": None,
    "compute_advection_in_horizontal_momentum_equation": (
        "compute_advection_in_horizontal_momentum",
        {},
    ),
    "compute_advection_in_vertical_momentum_equation": (
        "compute_advection_in_corrector_vertical_momentum",
        {},
    ),
    "compute_averaged_vn_and_fluxes_and_prepare_tracer_advection": (
        "compute_averaged_vn_and_fluxes",
        {
            "at_first_substep": False,
            "prepare_advection": True,
        },
    ),
    "compute_averaged_vn_and_fluxes_and_prepare_tracer_advection_first": (
        "compute_averaged_vn_and_fluxes",
        {
            "at_first_substep": True,
            "prepare_advection": True,
        },
    ),
    "compute_advection_in_predictor_vertical_momentum": (
        "compute_advection_in_predictor_vertical_momentum",
        {
            "skip_compute_predictor_vertical_advection": False,
        },
    ),
    "compute_advection_in_predictor_vertical_momentum_skip": (
        "compute_advection_in_predictor_vertical_momentum",
        {
            "skip_compute_predictor_vertical_advection": True,
        },
    ),
    "compute_diagnostics_from_normal_wind": (
        "compute_diagnostics_from_normal_wind",
        {"skip_compute_predictor_vertical_advection": False},
    ),
    "compute_diagnostics_from_normal_wind_skip": (
        "compute_diagnostics_from_normal_wind",
        {"skip_compute_predictor_vertical_advection": True},
    ),
    "compute_horizontal_velocity_quantities_and_fluxes": None,
    "compute_perturbed_quantities_and_interpolation": None,
    "compute_rho_theta_pgrad_and_update_vn": (
        "compute_rho_theta_pgrad_and_update_vn",
        {},
    ),
    "compute_interpolation_and_nonhydro_buoy": (
        "compute_interpolation_and_nonhydro_buoy",
        {},
    ),
    "update_mass_flux_weighted": None,
    "update_mass_flux_weighted_first": None,
    "vertically_implicit_solver_at_corrector_step": (
        "vertically_implicit_solver_at_corrector_step",
        {
            "at_first_substep": False,
            "at_last_substep": False,
        },
    ),
    "vertically_implicit_solver_at_corrector_step_first": (
        "vertically_implicit_solver_at_corrector_step",
        {
            "at_first_substep": True,
            "at_last_substep": False,
        },
    ),
    "vertically_implicit_solver_at_corrector_step_last": (
        "vertically_implicit_solver_at_corrector_step",
        {
            "at_first_substep": False,
            "at_last_substep": True,
        },
    ),
    "vertically_implicit_solver_at_predictor_step": (
        "vertically_implicit_solver_at_predictor_step",
        {
            "at_first_substep": False,
        },
    ),
    "vertically_implicit_solver_at_predictor_step_first": (
        "vertically_implicit_solver_at_predictor_step",
        {
            "at_first_substep": True,
        },
    ),
}


# Pre-process the 'fortran_to_icon4py' mapping and create a list of variants
# for each icon4py stencil
icon4py_stencils: dict[str, list[VariantDescriptor]] = {}
for fortran_stencil, v in fortran_to_icon4py.items():
    if v is None:
        # expect same stencil name in fortran and icon4py
        icon4py_stencil = fortran_stencil
        variant = (fortran_stencil, {})
    else:
        icon4py_stencil, desc = v
        variant = (fortran_stencil, desc)
    if icon4py_stencil in icon4py_stencils:
        icon4py_stencils[icon4py_stencil].append(variant)
    else:
        icon4py_stencils[icon4py_stencil] = [variant]


def load_openacc_log(filename: pathlib.Path) -> dict:
    log.info(f"Loading openacc data from {filename}")
    with filename.open("r") as f:
        j = json.load(f)

    data = {}
    count = {}
    for stencil, meas in j.items():
        if stencil in fortran_to_icon4py:
            data[stencil] = meas["latency_total"]["value"] / 1000.0  # milliseconds to seconds
            count[stencil] = meas["num_calls"]["value"]
        else:
            log.warning(f"skipping openacc meas for {stencil}")
    return data, count


def load_gt4py_timers(filename: pathlib.Path, metric: str) -> tuple[dict, dict]:
    import numpy as np  # noqa: PLC0415 [import-outside-top-level]

    log.info(f"Loading icon4py data from {filename}")
    with filename.open("r") as f:
        j = json.load(f)

    data = {}
    unmatched_data = {}
    for v in j.values():
        stencil_metadata = v["metadata"]
        stencil = stencil_metadata["name"]
        if metric not in v["metrics"]:
            log.debug(f"no meas for icon4py stencil {stencil_metadata}")
        else:
            metric_data = v["metrics"][metric]
            # we replace the first measurement with the median value
            metric_data[0] = np.median(metric_data)

            if stencil in icon4py_stencils:
                fortran_names = [
                    k
                    for k, desc in icon4py_stencils[stencil]
                    # all static args specified in the fortran stencil variant must match
                    if all(
                        stencil_metadata["static_args"].get(key) == value
                        for key, value in desc.items()
                    )
                    or len(desc) == 0
                ]
                if len(fortran_names) != 1:
                    raise ValueError(
                        f"Could not find a match for icon4py stencil {stencil_metadata}"
                    )
                fortran_name = fortran_names[0]
                if fortran_name in data:
                    raise ValueError(f"Double entry for fortran stencil {fortran_name}.")
                data[fortran_name] = metric_data
            else:
                log.warning(f"Unmatched icon4py meas for {stencil}")
                if stencil in unmatched_data:
                    raise NotImplementedError("Cannot handle stencil variant in unmatched data.")
                if len(metric_data) >= gt4py_unmatched_ncalls_threshold:
                    unmatched_data[stencil] = metric_data

    assert "update_mass_flux_weighted" in data
    update_mass_flux_weighted_original = data.pop("update_mass_flux_weighted")
    data["update_mass_flux_weighted"] = [
        update_mass_flux_weighted_original[i]
        for i in range(0, len(update_mass_flux_weighted_original))
        if i % 5 != 0
    ]  # take all BUT every fifth measurement which corresponds to the first substep of the 5 substeps per ICON timestep ("update_mass_flux_weighted_first")

    # Merge some stencils into 'update_mass_flux_weighted_first'
    assert "update_mass_flux_weighted_first" not in data
    data["update_mass_flux_weighted_first"] = [  # name matches icon-exclaim timer
        a + b
        for a, b in zip(
            update_mass_flux_weighted_original[
                ::5
            ],  # take ONLY every fifth measurement which corresponds to the first substep of the 5 substeps per ICON timestep
            unmatched_data.pop("init_cell_kdim_field_with_zero_wp"),
            strict=True,
        )
    ]

    # Merge 'compute_hydrostatic_correction_term' stencil into 'compute_rho_theta_pgrad_and_update_vn'
    assert "compute_hydrostatic_correction_term" in unmatched_data
    data["compute_rho_theta_pgrad_and_update_vn"] = [
        a + b
        for a, b in zip(
            data["compute_rho_theta_pgrad_and_update_vn"],
            unmatched_data.pop("compute_hydrostatic_correction_term"),
            strict=True,
        )
    ]

    # Merge some stencils into 'boundary_halo_cleanup'
    assert "boundary_halo_cleanup" not in data
    data["boundary_halo_cleanup"] = [
        a + b
        for a, b in zip(
            unmatched_data.pop("compute_exner_from_rhotheta"),
            unmatched_data.pop("update_theta_and_exner_in_halo"),
            strict=True,
        )
    ]

    # Merge some stencils into 'compute_dwdz_and_boundary_update_rho_theta_w'
    assert "compute_dwdz_and_boundary_update_rho_theta_w" not in data
    data["compute_dwdz_and_boundary_update_rho_theta_w"] = [  # name matches icon-exclaim timer
        a + b
        for a, b in zip(
            unmatched_data.pop("stencils_61_62"),
            unmatched_data.pop("compute_dwdz_for_divergence_damping"),
            strict=True,
        )
    ]

    diff = set(fortran_to_icon4py.keys()) - set(data.keys())
    if len(diff) != 0:
        print(f"Missing icon4py meas for these stencils: {diff} in file {filename}.")

    return data, unmatched_data


@cli.command(help=__doc__)
def compare_icon4py_openacc():
    import matplotlib.pyplot as plt  # noqa: PLC0415 [import-outside-top-level]
    import numpy as np  # noqa: PLC0415 [import-outside-top-level]

    openacc_meas, openacc_count = load_openacc_log(openacc_input)

    # Sort stencil names in descendent order of openacc total time.
    stencil_names: list[str] = [
        v[0] for v in sorted(openacc_meas.items(), key=lambda x: x[1], reverse=True)
    ]

    # Collect the names of unmatched stencils
    unmatched_stencil_names: list[str] | None = None

    backends: list[str] = [openacc_backend]
    data: dict[str, list[float]] = {
        openacc_backend: [openacc_meas[stencil] for stencil in stencil_names]
    }
    unmatched_data: dict[str, list[float]] = {}
    for backend, filename in gt4py_input.items():
        for metric in gt4py_metrics:
            # create a unique name for the combination of backend and metric
            label = f"{backend}_{metric}" if len(gt4py_metrics) > 1 else backend
            backends.append(label)
            gt4py_meas, unmatched_gt4py_meas = load_gt4py_timers(filename, metric)

            values = []
            for stencil in stencil_names:
                tvalues = gt4py_meas[stencil]
                if len(tvalues) != openacc_count[stencil]:
                    log.error(
                        f"Mismatch number of calls on {stencil} {openacc_backend}={openacc_count[stencil]} {label}={len(tvalues)}."
                    )
                values.append(np.sum(tvalues))
            data[label] = values

            # handle stencils that do not have a correspondant fortran stencil
            if unmatched_stencil_names is None:
                unmatched_stencil_names = sorted(list(unmatched_gt4py_meas.keys()))
            elif len(unmatched_stencil_names) != len(unmatched_gt4py_meas):
                raise ValueError("List of unmatched stencils with different length.")
            unmatched_data[label] = [
                np.sum(unmatched_gt4py_meas[stencil]) for stencil in unmatched_stencil_names
            ]

    # Combine all bar plots in a single plot
    fig, ax = plt.subplots(figsize=(22, 64))
    fig.subplots_adjust(left=0.3, right=0.98)
    bar_width = 8.0
    spacing = 20.0  # Additional spacing between stencil names
    gap = spacing * 4
    index = np.arange(len(stencil_names)) * (bar_width * len(backends) + spacing)
    extended_index = (
        np.arange(len(unmatched_stencil_names)) * (bar_width * (len(backends) - 1) + spacing)
        + index[-1]
        + bar_width
        + gap
    )

    # Define base RGB colors for different backends
    base_colors = [
        (0.1, 0.2, 0.5),  # Example RGB color 1
        (0.2, 0.6, 0.3),  # Example RGB color 2
        (0.8, 0.4, 0.1),  # Example RGB color 3
        (0.5, 0.1, 0.7),  # Example RGB color 4
        (0.3, 0.3, 0.3),  # Example RGB color 5
        (0.4, 0.4, 0.4),  # Example RGB color 6
    ]

    if len(base_colors) < len(backends):
        raise ValueError("Not enough base colors defined for the different backends.")

    for i, backend in enumerate(backends):
        color = base_colors[i]
        values = data[backend]
        ax.barh(index + i * bar_width, width=values, height=bar_width, label=backend, color=color)
        if i > 0:
            # Only annotate bars for gt4py backends
            ratios = [
                openacc_meas[stencil] / val
                for stencil, val in zip(stencil_names, values, strict=True)
            ]
            for k, (val, ratio) in enumerate(zip(values, ratios)):
                ax.text(
                    val + 0.02,  # Position slightly above the bar
                    index[k] + (i - 0.3) * bar_width,
                    f"{ratio:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=0,
                    color=color,
                )
            # Plot also unmatched icon4py stencils
            ax.barh(
                extended_index + (i - 1) * bar_width,
                unmatched_data[backend],
                bar_width,
                color=color,
            )

    ax.set_title(f"Backend comparison on {experiment} ({target})")
    ax.set_xlabel("Total compute time [s]")
    ax.set_ylabel(f"Stencil name (speedup w.r.t. {openacc_backend} next to the bars)")
    ax.set_yticks(
        np.concatenate(
            [
                index + (bar_width * len(backends)) / 2,
                extended_index + (bar_width * (len(backends) - 1)) / 2,
            ]
        )
        - bar_width / 2
    )
    ax.set_yticklabels(stencil_names + unmatched_stencil_names, rotation=0)

    # Add a horizontal line to separate unmatched stencils
    ax.axhline(y=(extended_index[0] - gap / 2), color="gray", linestyle="--", linewidth=1.5)

    ax.legend(loc="upper right")

    # Save the plot to a file
    output_dir = pathlib.Path.cwd() / "plots"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{output_filename}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    print("")
    print(f"Plot figure saved to {output_file}")


if __name__ == "__main__":
    sys.exit(cli())
