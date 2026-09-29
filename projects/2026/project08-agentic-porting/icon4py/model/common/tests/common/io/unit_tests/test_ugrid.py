# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import xarray as xa

from icon4py.model.common.io.ugrid import (
    FILL_VALUE,
    IconUGridPatcher,
    IconUGridWriter,
    extract_horizontal_coordinates,
    load_data_file,
)
from icon4py.model.testing import datatest_utils, definitions, grid_utils

from ...fixtures import test_path


def grid_files():
    grids = [
        definitions.Grids.R02B04_GLOBAL,
        definitions.Grids.MCH_CH_R04B09_DSL,
    ]

    for grid in grids:
        yield datatest_utils.get_grid_filepath(grid)


@pytest.mark.parametrize("file", grid_files())
def test_convert_to_ugrid(file):
    with load_data_file(file) as ds:
        patch = IconUGridPatcher()
        uxds = patch(ds, validate=True)
        assert uxds.attrs["title"] == "ICON grid description"
        assert uxds.attrs["ellipsoid_name"] == "Sphere"
        assert uxds["mesh"].attrs["cf_role"] == "mesh_topology"
        assert uxds["mesh"].attrs["topology_dimension"] == 2
        assert uxds["mesh"].attrs["face_dimension"] == "cell"
        assert uxds["mesh"].attrs["edge_dimension"] == "edge"
        assert uxds["mesh"].attrs["node_dimension"] == "vertex"
        assert uxds["mesh"].attrs["node_coordinates"] == "vlon vlat"
        assert uxds["mesh"].attrs["face_node_connectivity"] == "vertex_of_cell"


@pytest.mark.parametrize("file", grid_files())
def test_icon_ugrid_writer_writes_ugrid_file(file, test_path):
    output_dir = test_path.joinpath("output")
    output_dir.mkdir(0o755, exist_ok=True)
    writer = IconUGridWriter(file, output_dir)
    writer(validate=False)
    fname = output_dir.iterdir().__next__().name
    assert fname == file.stem + "_ugrid.nc"


@pytest.mark.parametrize("file", grid_files())
def test_icon_ugrid_patch_index_transformation(file):
    with load_data_file(file) as ds:
        patch = IconUGridPatcher()
        uxds = patch(ds)
        for name in patch.index_lists:
            assert_start_index(uxds, name)
            assert uxds[name].dtype == "int32"


@pytest.mark.parametrize("file", grid_files())
def test_icon_ugrid_patch_transposed_index_lists(file):
    with load_data_file(file) as ds:
        patch = IconUGridPatcher()
        uxds = patch(ds)
        horizontal_dims = ("cell", "edge", "vertex")
        horizontal_sizes = list(uxds.sizes[k] for k in horizontal_dims)
        for name in patch.connectivities:
            assert len(uxds[name].shape) == 2
            assert uxds[name].shape[0] > uxds[name].shape[1]
            assert uxds[name].dims[0] in horizontal_dims
            assert uxds[name].shape[0] in horizontal_sizes


@pytest.mark.parametrize("file", grid_files())
def test_icon_ugrid_patch_fill_value(file):
    with load_data_file(file) as ds:
        patch = IconUGridPatcher()
        uxds = patch(ds)
        patch._set_fill_value(uxds)
        for name in patch.connectivities:
            assert uxds[name].attrs["_FillValue"] == FILL_VALUE


def assert_start_index(uxds: xa.Dataset, name: str):
    assert uxds[name].attrs["start_index"] == 0
    assert np.min(np.nonzero(uxds[name].data > FILL_VALUE)) == 0


@pytest.mark.parametrize("file", grid_files())
def test_extract_horizontal_coordinates(file):
    with load_data_file(file) as ds:
        dim_sizes = ds.sizes
        coords = extract_horizontal_coordinates(ds)
        # TODO(halungge): fix:
        #  - 'long_name', 'standard_name' of attributes fx cell center latitudes
        # - 'units' of lat, lon are conventionally in degrees not in radians as ICON provides themconvert to degrees_north, degrees_east..
        for k in ("cell", "edge", "vertex"):
            assert k in coords
            assert coords[k][0].shape[0] == dim_sizes[k]
