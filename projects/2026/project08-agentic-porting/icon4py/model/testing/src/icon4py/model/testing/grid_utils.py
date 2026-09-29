# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib

import gt4py.next.typing as gtx_typing

from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    grid_manager as gm,
    gridfile,
    vertical as v_grid,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import config as driver_config
from icon4py.model.testing import config, data_handling, datatest_utils as dt_utils, definitions


grid_geometries: dict[str, geometry.GridGeometry] = {}


def get_grid_manager_from_experiment(
    experiment: definitions.Experiment,
    keep_skip_values: bool,
    allocator: gtx_typing.Allocator,
) -> gm.GridManager:
    return get_grid_manager_from_identifier(
        experiment.grid,
        num_levels=experiment.config.vertical_grid.num_levels,
        keep_skip_values=keep_skip_values,
        allocator=allocator,
    )


def get_grid_manager_from_identifier(
    grid: definitions.GridDescription,
    num_levels: int,
    keep_skip_values: bool,
    allocator: gtx_typing.Allocator,
) -> gm.GridManager:
    grid_file = _download_grid_file(grid)
    return get_grid_manager(
        grid_file, num_levels=num_levels, keep_skip_values=keep_skip_values, allocator=allocator
    )


def get_grid_manager(
    filename: pathlib.Path,
    num_levels: int,
    keep_skip_values: bool,
    allocator: gtx_typing.Allocator,
) -> gm.GridManager:
    """
    Construct a GridManager instance for an ICON grid file.

    Args:
        filename: full path to the file
        num_levels: number of vertical levels, needed for IconGrid construction but independent from grid file
        keep_skip_values: whether to keep skip values
        backend: the gt4py Backend we are running on
    """
    manager = gm.GridManager(
        grid_file=filename,
        config=v_grid.VerticalGridConfig(num_levels=num_levels),
        offset_transformation=gridfile.ToZeroBasedIndexTransformation(),
    )
    manager(allocator=allocator, keep_skip_values=keep_skip_values)
    return manager


def _download_grid_file(grid: definitions.GridDescription) -> pathlib.Path:
    full_name = dt_utils.get_grid_filepath(grid)
    grid_directory = full_name.parent
    grid_directory.mkdir(parents=True, exist_ok=True)
    if config.ENABLE_GRID_DOWNLOAD:
        uri = dt_utils.get_grid_archive_url(definitions.TESTDATA_ROOT_URL, grid)
        data_handling.download_and_extract(
            uri,
            grid_directory,
        )
    else:
        # If grid download is disabled, we check if the file exists
        # without locking. We assume the location is managed by the user
        # and avoid locking shared directories (e.g. on CI).
        if not full_name.exists():
            raise FileNotFoundError(
                f"Grid file {full_name} does not exist and grid download is disabled."
            )
    return full_name


def get_grid_geometry(
    backend: gtx_typing.Backend | None,
    grid: definitions.GridDescription,
    experiment_config: driver_config.ExperimentConfig,
) -> geometry.GridGeometry:
    register_name = "_".join(
        (
            grid.name,
            data_alloc.backend_name(backend),
            str(experiment_config.geometry.use_analytical_means),
        )
    )

    def _construct_grid_geometry() -> geometry.GridGeometry:
        gm = get_grid_manager_from_identifier(
            grid,
            num_levels=experiment_config.vertical_grid.num_levels,
            keep_skip_values=True,
            allocator=model_backends.get_allocator(backend),
        )
        return geometry.GridGeometry(
            grid=gm.grid,
            decomposition_info=gm.decomposition_info,
            backend=backend,
            coordinates=gm.coordinates,
            extra_fields=gm.geometry_fields,
            metadata=geometry_attrs.attrs,
            config=experiment_config.geometry,
            process_props=decomposition.SingleNodeProcessProperties(),
            exchange=decomposition.single_node_exchange,
        )

    if not grid_geometries.get(register_name):
        grid_geometries[register_name] = _construct_grid_geometry()
    return grid_geometries[register_name]
