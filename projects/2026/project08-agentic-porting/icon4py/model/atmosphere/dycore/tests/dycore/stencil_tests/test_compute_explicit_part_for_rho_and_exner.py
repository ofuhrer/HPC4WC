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

from icon4py.model.atmosphere.dycore.stencils.compute_explicit_part_for_rho_and_exner import (
    compute_explicit_part_for_rho_and_exner,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def compute_explicit_part_for_rho_and_exner_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    rho_nnow: np.ndarray,
    inv_ddqz_z_full: np.ndarray,
    z_flxdiv_mass: np.ndarray,
    z_contr_w_fl_l: np.ndarray,
    exner_pr: np.ndarray,
    z_beta: np.ndarray,
    z_flxdiv_theta: np.ndarray,
    theta_v_ic: np.ndarray,
    ddt_exner_phy: np.ndarray,
    dtime: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_rho_expl = rho_nnow - dtime * inv_ddqz_z_full * (
        z_flxdiv_mass + z_contr_w_fl_l[:, :-1] - z_contr_w_fl_l[:, 1:]
    )

    z_exner_expl = (
        exner_pr
        - z_beta
        * (
            z_flxdiv_theta
            + (theta_v_ic * z_contr_w_fl_l)[:, :-1]
            - (theta_v_ic * z_contr_w_fl_l)[:, 1:]
        )
        + dtime * ddt_exner_phy
    )
    return (z_rho_expl, z_exner_expl)


class TestComputeExplicitPartForRhoAndExner(StencilTest):
    PROGRAM = compute_explicit_part_for_rho_and_exner
    OUTPUTS = ("z_rho_expl", "z_exner_expl")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        rho_nnow: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        z_flxdiv_mass: np.ndarray,
        z_contr_w_fl_l: np.ndarray,
        exner_pr: np.ndarray,
        z_beta: np.ndarray,
        z_flxdiv_theta: np.ndarray,
        theta_v_ic: np.ndarray,
        ddt_exner_phy: np.ndarray,
        dtime: float,
        **kwargs: Any,
    ) -> dict:
        (z_rho_expl, z_exner_expl) = compute_explicit_part_for_rho_and_exner_numpy(
            connectivities=connectivities,
            rho_nnow=rho_nnow,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_flxdiv_mass=z_flxdiv_mass,
            z_contr_w_fl_l=z_contr_w_fl_l,
            exner_pr=exner_pr,
            z_beta=z_beta,
            z_flxdiv_theta=z_flxdiv_theta,
            theta_v_ic=theta_v_ic,
            ddt_exner_phy=ddt_exner_phy,
            dtime=dtime,
        )
        return dict(z_rho_expl=z_rho_expl, z_exner_expl=z_exner_expl)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        dtime = ta.wpfloat("1.0")
        rho_nnow = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_flxdiv_mass = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_contr_w_fl_l = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        exner_pr = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        z_beta = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_flxdiv_theta = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        theta_v_ic = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=ta.wpfloat
        )
        ddt_exner_phy = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)

        z_rho_expl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        z_exner_expl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            rho_nnow=rho_nnow,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_flxdiv_mass=z_flxdiv_mass,
            z_contr_w_fl_l=z_contr_w_fl_l,
            exner_pr=exner_pr,
            z_beta=z_beta,
            z_flxdiv_theta=z_flxdiv_theta,
            theta_v_ic=theta_v_ic,
            ddt_exner_phy=ddt_exner_phy,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
