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

from icon4py.model.atmosphere.dycore.stencils.set_theta_v_prime_ic_at_lower_boundary import (
    set_theta_v_prime_ic_at_lower_boundary,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest

from .test_interpolate_to_surface import interpolate_to_surface_numpy


def set_theta_v_prime_ic_at_lower_boundary_numpy(
    wgtfacq_c: np.ndarray,
    z_rth_pr: np.ndarray,
    theta_ref_ic: np.ndarray,
    z_theta_v_pr_ic: np.ndarray,
    theta_v_ic: np.ndarray,
) -> tuple[np.ndarray, ...]:
    z_theta_v_pr_ic = interpolate_to_surface_numpy(
        wgtfacq_c=wgtfacq_c,
        interpolant=z_rth_pr,
        interpolation_to_surface=z_theta_v_pr_ic,
    )
    theta_v_ic[:, 3:] = (theta_ref_ic + z_theta_v_pr_ic)[:, 3:]
    return (z_theta_v_pr_ic, theta_v_ic)


class TestInitThetaVPrimeIcAtLowerBoundary(StencilTest):
    PROGRAM = set_theta_v_prime_ic_at_lower_boundary
    OUTPUTS = ("z_theta_v_pr_ic", "theta_v_ic")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        wgtfacq_c: np.ndarray,
        z_rth_pr: np.ndarray,
        theta_ref_ic: np.ndarray,
        z_theta_v_pr_ic: np.ndarray,
        theta_v_ic: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        (z_theta_v_pr_ic, theta_v_ic) = set_theta_v_prime_ic_at_lower_boundary_numpy(
            wgtfacq_c=wgtfacq_c,
            z_rth_pr=z_rth_pr,
            theta_ref_ic=theta_ref_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
        )
        return dict(z_theta_v_pr_ic=z_theta_v_pr_ic, theta_v_ic=theta_v_ic)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        wgtfacq_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_rth_pr = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        theta_ref_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        z_theta_v_pr_ic = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        theta_v_ic = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            wgtfacq_c=wgtfacq_c,
            z_rth_pr=z_rth_pr,
            theta_ref_ic=theta_ref_ic,
            z_theta_v_pr_ic=z_theta_v_pr_ic,
            theta_v_ic=theta_v_ic,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=3,
            vertical_end=gtx.int32(grid.num_levels),
        )
