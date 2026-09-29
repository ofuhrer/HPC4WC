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

import icon4py.model.common.type_alias as ta
from icon4py.model.atmosphere.dycore.stencils.accumulate_prep_adv_fields import (
    accumulate_prep_adv_fields,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def accumulate_prep_adv_fields_numpy(
    z_vn_avg: np.ndarray,
    mass_fl_e: np.ndarray,
    vn_traj: np.ndarray,
    mass_flx_me: np.ndarray,
    r_nsubsteps: ta.wpfloat,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    vn_traj = vn_traj + r_nsubsteps * z_vn_avg
    mass_flx_me = mass_flx_me + r_nsubsteps * mass_fl_e
    return vn_traj, mass_flx_me


class TestAccumulatePrepAdvFields(StencilTest):
    PROGRAM = accumulate_prep_adv_fields
    OUTPUTS = ("vn_traj", "mass_flx_me")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        z_vn_avg: np.ndarray,
        mass_fl_e: np.ndarray,
        vn_traj: np.ndarray,
        mass_flx_me: np.ndarray,
        r_nsubsteps: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        vn_traj, mass_flx_me = accumulate_prep_adv_fields_numpy(
            z_vn_avg,
            mass_fl_e,
            vn_traj,
            mass_flx_me,
            r_nsubsteps,
        )

        return dict(vn_traj=vn_traj, mass_flx_me=mass_flx_me)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        mass_fl_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        mass_flx_me = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_vn_avg = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        vn_traj = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        r_nsubsteps = wpfloat("9.0")

        return dict(
            z_vn_avg=z_vn_avg,
            mass_fl_e=mass_fl_e,
            vn_traj=vn_traj,
            mass_flx_me=mass_flx_me,
            r_nsubsteps=r_nsubsteps,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
