# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.update_mass_volume_flux import update_mass_volume_flux
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def update_mass_volume_flux_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    z_contr_w_fl_l: np.ndarray,
    rho_ic: np.ndarray,
    vwind_impl_wgt: np.ndarray,
    w: np.ndarray,
    mass_flx_ic: np.ndarray,
    vol_flx_ic: np.ndarray,
    r_nsubsteps: float,
) -> tuple[np.ndarray, np.ndarray]:
    vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=1)
    z_a = r_nsubsteps * (z_contr_w_fl_l + rho_ic * vwind_impl_wgt * w)
    mass_flx_ic = mass_flx_ic + z_a
    vol_flx_ic = vol_flx_ic + z_a / rho_ic
    return (mass_flx_ic, vol_flx_ic)


class TestUpdateMassVolumeFlux(StencilTest):
    PROGRAM = update_mass_volume_flux
    OUTPUTS = (
        "mass_flx_ic",
        "vol_flx_ic",
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        z_contr_w_fl_l: np.ndarray,
        rho_ic: np.ndarray,
        vwind_impl_wgt: np.ndarray,
        w: np.ndarray,
        mass_flx_ic: np.ndarray,
        vol_flx_ic: np.ndarray,
        r_nsubsteps: float,
        **kwargs: Any,
    ) -> dict:
        (mass_flx_ic, vol_flx_ic) = update_mass_volume_flux_numpy(
            connectivities=connectivities,
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            vwind_impl_wgt=vwind_impl_wgt,
            w=w,
            mass_flx_ic=mass_flx_ic,
            vol_flx_ic=vol_flx_ic,
            r_nsubsteps=r_nsubsteps,
        )
        return dict(mass_flx_ic=mass_flx_ic, vol_flx_ic=vol_flx_ic)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        z_contr_w_fl_l = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        vwind_impl_wgt = data_alloc.random_field(grid, dims.CellDim, dtype=ta.wpfloat)
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        mass_flx_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        vol_flx_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        r_nsubsteps = 7.0

        return dict(
            z_contr_w_fl_l=z_contr_w_fl_l,
            rho_ic=rho_ic,
            vwind_impl_wgt=vwind_impl_wgt,
            w=w,
            mass_flx_ic=mass_flx_ic,
            vol_flx_ic=vol_flx_ic,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
