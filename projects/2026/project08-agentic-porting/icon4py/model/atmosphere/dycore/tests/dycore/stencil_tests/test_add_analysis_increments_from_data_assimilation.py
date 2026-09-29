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

from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_from_data_assimilation import (
    add_analysis_increments_from_data_assimilation,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def add_analysis_increments_from_data_assimilation_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    z_rho_expl: np.ndarray,
    rho_incr: np.ndarray,
    z_exner_expl: np.ndarray,
    exner_incr: np.ndarray,
    iau_wgt_dyn: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_rho_expl = z_rho_expl + iau_wgt_dyn * rho_incr
    z_exner_expl = z_exner_expl + iau_wgt_dyn * exner_incr
    return (z_rho_expl, z_exner_expl)


class TestAddAnalysisIncrementsFromDataAssimilation(StencilTest):
    PROGRAM = add_analysis_increments_from_data_assimilation
    OUTPUTS = ("z_rho_expl", "z_exner_expl")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        z_rho_expl: np.ndarray,
        rho_incr: np.ndarray,
        z_exner_expl: np.ndarray,
        exner_incr: np.ndarray,
        iau_wgt_dyn: float,
        **kwargs: Any,
    ) -> dict:
        z_rho_expl, z_exner_expl = add_analysis_increments_from_data_assimilation_numpy(
            connectivities=connectivities,
            z_rho_expl=z_rho_expl,
            rho_incr=rho_incr,
            z_exner_expl=z_exner_expl,
            exner_incr=exner_incr,
            iau_wgt_dyn=iau_wgt_dyn,
        )
        return dict(z_rho_expl=z_rho_expl, z_exner_expl=z_exner_expl)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        z_exner_expl = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        exner_incr = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_rho_expl = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_incr = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        iau_wgt_dyn = ta.wpfloat("8.0")

        return dict(
            z_rho_expl=z_rho_expl,
            z_exner_expl=z_exner_expl,
            rho_incr=rho_incr,
            exner_incr=exner_incr,
            iau_wgt_dyn=iau_wgt_dyn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
