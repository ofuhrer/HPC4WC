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

from icon4py.model.atmosphere.dycore.stencils.compute_dwdz_for_divergence_damping import (
    _compute_dwdz_for_divergence_damping,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_dwdz_for_divergence_damping_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    inv_ddqz_z_full: np.ndarray,
    w: np.ndarray,
    w_concorr_c: np.ndarray,
) -> np.ndarray:
    z_dwdz_dd = inv_ddqz_z_full * (
        (w[:, :-1] - w[:, 1:]) - (w_concorr_c[:, :-1] - w_concorr_c[:, 1:])
    )
    return z_dwdz_dd


class TestComputeDwdzForDivergenceDamping(StencilTest):
    PROGRAM = _compute_dwdz_for_divergence_damping
    OUTPUTS = ("out",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        inv_ddqz_z_full: np.ndarray,
        w: np.ndarray,
        w_concorr_c: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_dwdz_dd = compute_dwdz_for_divergence_damping_numpy(
            connectivities, inv_ddqz_z_full=inv_ddqz_z_full, w=w, w_concorr_c=w_concorr_c
        )
        return dict(out=z_dwdz_dd)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, Any]:
        inv_ddqz_z_full = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=wpfloat)
        w_concorr_c = random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, dtype=vpfloat
        )
        z_dwdz_dd = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            inv_ddqz_z_full=inv_ddqz_z_full,
            w=w,
            w_concorr_c=w_concorr_c,
            out=z_dwdz_dd,
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KDim: (0, gtx.int32(grid.num_levels)),
            },
        )
