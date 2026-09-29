# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Final

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_results_for_thermodynamic_variables import (
    compute_results_for_thermodynamic_variables,
)
from icon4py.model.common import constants, dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def compute_results_for_thermodynamic_variables_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    z_rho_expl: np.ndarray,
    vwind_impl_wgt: np.ndarray,
    inv_ddqz_z_full: np.ndarray,
    rho_ic: np.ndarray,
    w: np.ndarray,
    z_exner_expl: np.ndarray,
    exner_ref_mc: np.ndarray,
    z_alpha: np.ndarray,
    z_beta: np.ndarray,
    rho_now: np.ndarray,
    theta_v_now: np.ndarray,
    exner_now: np.ndarray,
    dtime: float,
) -> tuple[np.ndarray, ...]:
    rho_ic_offset_1 = rho_ic[:, 1:]
    w_offset_0 = w[:, :-1]
    w_offset_1 = w[:, 1:]
    z_alpha_offset_1 = z_alpha[:, 1:]
    vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=1)
    rho_new = z_rho_expl - vwind_impl_wgt * dtime * inv_ddqz_z_full * (
        rho_ic[:, :-1] * w_offset_0 - rho_ic_offset_1 * w_offset_1
    )
    exner_new = (
        z_exner_expl
        + exner_ref_mc
        - z_beta * (z_alpha[:, :-1] * w_offset_0 - z_alpha_offset_1 * w_offset_1)
    )
    theta_v_new = (
        rho_now * theta_v_now * ((exner_new / exner_now - 1.0) * constants.CVD_O_RD + 1.0) / rho_new
    )
    return rho_new, exner_new, theta_v_new


class TestComputeResultsForThermodynamicVariables(StencilTest):
    PROGRAM = compute_results_for_thermodynamic_variables
    OUTPUTS = ("rho_new", "exner_new", "theta_v_new")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        z_rho_expl: np.ndarray,
        vwind_impl_wgt: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        rho_ic: np.ndarray,
        w: np.ndarray,
        z_exner_expl: np.ndarray,
        exner_ref_mc: np.ndarray,
        z_alpha: np.ndarray,
        z_beta: np.ndarray,
        rho_now: np.ndarray,
        theta_v_now: np.ndarray,
        exner_now: np.ndarray,
        dtime: float,
        **kwargs: Any,
    ) -> dict:
        (rho_new, exner_new, theta_v_new) = compute_results_for_thermodynamic_variables_numpy(
            connectivities=connectivities,
            z_rho_expl=z_rho_expl,
            vwind_impl_wgt=vwind_impl_wgt,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ic=rho_ic,
            w=w,
            z_exner_expl=z_exner_expl,
            exner_ref_mc=exner_ref_mc,
            z_alpha=z_alpha,
            z_beta=z_beta,
            rho_now=rho_now,
            theta_v_now=theta_v_now,
            exner_now=exner_now,
            dtime=dtime,
        )
        return dict(rho_new=rho_new, exner_new=exner_new, theta_v_new=theta_v_new)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        z_rho_expl = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        vwind_impl_wgt = data_alloc.random_field(grid, dims.CellDim, dtype=ta.wpfloat)
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        rho_ic = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        w = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        z_exner_expl = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        exner_ref_mc = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_alpha = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.vpfloat
        )
        z_beta = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        rho_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        theta_v_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        exner_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_new = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        exner_new = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        theta_v_new = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        dtime = ta.wpfloat("5.0")

        return dict(
            z_rho_expl=z_rho_expl,
            vwind_impl_wgt=vwind_impl_wgt,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ic=rho_ic,
            w=w,
            z_exner_expl=z_exner_expl,
            exner_ref_mc=exner_ref_mc,
            z_alpha=z_alpha,
            z_beta=z_beta,
            rho_now=rho_now,
            theta_v_now=theta_v_now,
            exner_now=exner_now,
            rho_new=rho_new,
            exner_new=exner_new,
            theta_v_new=theta_v_new,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
