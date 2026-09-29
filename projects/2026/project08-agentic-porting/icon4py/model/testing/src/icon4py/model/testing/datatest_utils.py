# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import pathlib
import urllib.parse

import gt4py.next.typing as gtx_typing

from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.standalone_driver import config as driver_config
from icon4py.model.testing import data_handling, definitions, serialbox


logger = logging.getLogger(__name__)


def get_process_properties_for_run(
    run_instance: decomposition.RunType,
) -> decomposition.ProcessProperties:
    return decomposition.get_process_properties(run_instance)


def get_experiment_name_with_version(
    experiment_description: definitions.ExperimentDescription,
) -> str:
    """Generate experiment name with version suffix."""
    return f"{experiment_description.name}_v{experiment_description.version:02d}"


def get_ranked_experiment_name_with_version(
    experiment_description: definitions.ExperimentDescription, comm_size: int
) -> str:
    """Generate ranked experiment name with version suffix."""
    return f"mpitask{comm_size}_{get_experiment_name_with_version(experiment_description)}"


def get_experiment_archive_filename(
    experiment_description: definitions.ExperimentDescription, comm_size: int
) -> str:
    """Generate ranked archive filename for an experiment."""
    return f"{get_ranked_experiment_name_with_version(experiment_description, comm_size)}.tar.gz"


def get_experiment_archive_url(root_url: str, filepath: str) -> str:
    """Build a download URL for experiment archive from root URL."""
    return f"{root_url}/{urllib.parse.quote(filepath)}"


def get_grid_archive_filename(grid: definitions.GridDescription) -> str:
    return f"{grid.name}.tar.gz"


def get_grid_filename(grid: definitions.GridDescription) -> str:
    return f"{grid.name}.nc"


def get_grid_filepath(grid: definitions.GridDescription) -> pathlib.Path:
    return definitions.grids_path().joinpath(grid.name, get_grid_filename(grid))


def get_grid_archive_url(root_url: str, grid: definitions.GridDescription) -> str:
    """Build a download URL for a grid archive from root URL."""
    filepath = f"{definitions.GRID_DATA_DIR}/{get_grid_archive_filename(grid)}"
    return f"{root_url}/{urllib.parse.quote(filepath)}"


def get_muphys_archive_url(root_url: str, experiment_type: str, experiment_name: str) -> str:
    """Build a download URL for a muphys archive from root URL."""
    filepath = f"{definitions.MUPHYS_DATA_DIR}/{experiment_type}/{experiment_name}.tar.gz"
    return f"{root_url}/{urllib.parse.quote(filepath)}"


def get_path_for_experiment(
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
) -> pathlib.Path:
    """Get the path to an experiment root directory."""

    experiment_dir = get_ranked_experiment_name_with_version(
        experiment_description,
        process_props.comm_size,
    )
    return definitions.serialized_data_path() / experiment_dir


def get_datapath_for_experiment(
    experiment_description: definitions.ExperimentDescription,
    process_props: decomposition.ProcessProperties,
) -> pathlib.Path:
    """Get the path to serialized data for an experiment."""

    experiment_path = get_path_for_experiment(
        experiment_description,
        process_props,
    )
    return experiment_path.joinpath(definitions.SERIALIZED_DATA_SUBDIR)


def create_icon_serial_data_provider(
    datapath: pathlib.Path,
    rank: int,
    backend: gtx_typing.Backend | None,
) -> serialbox.IconSerialDataProvider:
    return serialbox.IconSerialDataProvider(
        backend=backend,
        fname_prefix="icon_pydycore",
        path=str(datapath),
        mpi_rank=rank,
        do_print=True,
    )


def download_experiment(
    experiment_description: definitions.ExperimentDescription,
    processor_props: decomposition.ProcessProperties,
) -> None:
    """Download data and config for an experiment--if not already present."""
    comm_size = processor_props.comm_size
    root_url = definitions.TESTDATA_ROOT_URL
    archive_filename = get_experiment_archive_filename(experiment_description, comm_size)
    archive_path = definitions.EXPERIMENT_DATA_DIR + "/" + archive_filename
    uri = get_experiment_archive_url(root_url, archive_path)
    destination_path = get_datapath_for_experiment(experiment_description, processor_props)
    data_handling.download_test_data(destination_path.parent, uri)


def create_experiment_configuration(
    experiment_description: definitions.ExperimentDescription,
    processor_props: decomposition.ProcessProperties,
) -> driver_config.ExperimentConfig:
    experiment_path = get_path_for_experiment(experiment_description, processor_props)
    return driver_config.read_experiment_config_from_fortran(experiment_path)
