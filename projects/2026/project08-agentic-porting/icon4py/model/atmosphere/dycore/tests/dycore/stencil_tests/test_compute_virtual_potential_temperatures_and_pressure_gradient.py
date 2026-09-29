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

from icon4py.model.atmosphere.dycore.stencils.compute_virtual_potential_temperatures_and_pressure_gradient import (
    compute_virtual_potential_temperatures_and_pressure_gradient,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_virtual_potential_temperatures_and_pressure_gradient_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    wgtfac_c: np.ndarray,
    z_rth_pr_2: np.ndarray,
    theta_v: np.ndarray,
    vwind_expl_wgt: np.ndarray,
    exner_pr: np.ndarray,
    d_exner_dz_ref_ic: np.ndarray,
    ddqz_z_half: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_rth_pr_2_offset = np.roll(z_rth_pr_2, axis=1, shift=1)
    theta_v_offset = np.roll(theta_v, axis=1, shift=1)
    exner_pr_offset = np.roll(exner_pr, axis=1, shift=1)
    vwind_expl_wgt = np.expand_dims(vwind_expl_wgt, axis=-1)

    z_theta_v_pr_ic = wgtfac_c * z_rth_pr_2 + (1.0 - wgtfac_c) * z_rth_pr_2_offset
    z_theta_v_pr_ic[:, 0] = 0
    theta_v_ic = wgtfac_c * theta_v + (1 - wgtfac_c) * theta_v_offset
    theta_v_ic[:, 0] = 0
    z_th_ddz_exner_c = (
        vwind_expl_wgt * theta_v_ic * (exner_pr_offset - exner_pr) / ddqz_z_half
        + z_theta_v_pr_ic * d_exner_dz_ref_ic
    )
    z_th_ddz_exner_c[:, 0] = 0

    return (
        z_theta_v_pr_ic,
        theta_v_ic,
        z_th_ddz_exner_c,
    )


class TestComputeVirtualPotentialTemperaturesAndPressureGradient(StencilTest):
    PROGRAM = compute_virtual_potential_temperatures_and_pressure_gradient
    OUTPUTS = ("z_theta_v_pr_ic", "theta_v_ic", "z_th_ddz_exner_c")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        wgtfac_c: np.ndarray,
        z_rth_pr_2: np.ndarray,
        theta_v: np.ndarray,
        vwind_expl_wgt: np.ndarray,
        exner_pr: np.ndarray,
        d_exner_dz_ref_ic: np.ndarray,
        ddqz_z_half: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        (
            z_theta_v_pr_ic,
            theta_v_ic,
            z_th_ddz_exner_c,
        ) = compute_virtual_potential_temperatures_and_pressure_gradient_numpy(
            connectivities=connectivities,
            wgtfac_c=wgtfac_c,
            z_rth_pr_2=z_rth_pr_2,
            theta_v=theta_v,
            vwind_expl_wgt=vwind_expl_wgt,
            exner_pr=exner_pr,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            ddqz_z_half=ddqz_z_half,
        )

        return dict(
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        wgtfac_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_rth_pr_2 = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        theta_v = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        vwind_expl_wgt = random_field(grid, dims.CellDim, dtype=wpfloat)
        exner_pr = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        d_exner_dz_ref_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        ddqz_z_half = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_theta_v_pr_ic = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        theta_v_ic = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        z_th_ddz_exner_c = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            wgtfac_c=wgtfac_c,
            z_rth_pr_2=z_rth_pr_2,
            theta_v=theta_v,
            vwind_expl_wgt=vwind_expl_wgt,
            exner_pr=exner_pr,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            ddqz_z_half=ddqz_z_half,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            z_th_ddz_exner_c=z_th_ddz_exner_c,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
