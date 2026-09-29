# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import xarray as xa

import icon4py.model.common.grid.simple as simple_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.io import ugrid, utils
from icon4py.model.common.states import data, model
from icon4py.model.common.utils import data_allocation as data_alloc


def test_data_array_has_ugrid_and_cf_attributes():
    grid = simple_grid.simple_grid()
    buffer = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
    data_array = utils.to_data_array(buffer, data.PROGNOSTIC_CF_ATTRIBUTES["air_density"])
    assert data_array.attrs["units"] == "kg m-3"
    assert data_array.attrs["standard_name"] == "air_density"
    assert data_array.attrs["icon_var_name"] == "rho"
    assert data_array.attrs["long_name"].startswith("density")
    ## attributes for mapping to ugrid
    assert data_array.attrs["location"] == "face"
    assert data_array.attrs["coordinates"] == "clon clat"
    assert data_array.attrs["mesh"] == ugrid.MESH


def test_type_check_for_datafields():
    grid = simple_grid.simple_grid()
    field = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=gtx.int32)
    model_field = model.ModelField(data=field, attrs=data.DIAGNOSTIC_CF_ATTRIBUTES["eastward_wind"])
    assert not isinstance(field, model.ModelField)
    assert isinstance(model_field, model.DataField)
    dataarray = utils.to_data_array(field, attrs=model_field.attrs)
    assert isinstance(dataarray, xa.DataArray)
    assert isinstance(dataarray, model.DataField)
