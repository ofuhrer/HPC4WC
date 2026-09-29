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

from icon4py.model.atmosphere.dycore.stencils.interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges import (
    interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing import stencil_tests


def interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_vn_ie_numpy(
    wgtfac_e: np.ndarray, vn: np.ndarray
) -> np.ndarray:
    vn_ie_k_minus_1 = np.roll(vn, shift=1, axis=1)
    vn_ie = wgtfac_e * vn + (1.0 - wgtfac_e) * vn_ie_k_minus_1
    vn_ie[:, 0] = vn[:, 0]
    return vn_ie


def interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_z_kin_hor_e_numpy(
    vn: np.ndarray, vt: np.ndarray
) -> np.ndarray:
    z_kin_hor_e = 0.5 * (vn * vn + vt * vt)
    return z_kin_hor_e


def interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_numpy(
    wgtfac_e: np.ndarray, vn: np.ndarray, vt: np.ndarray, **kwargs: Any
) -> tuple[np.ndarray, np.ndarray]:
    vn_ie = interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_vn_ie_numpy(
        wgtfac_e, vn
    )
    z_kin_hor_e = (
        interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_z_kin_hor_e_numpy(vn, vt)
    )
    return (
        vn_ie,
        z_kin_hor_e,
    )


class TestInterpolateVnToHalfLevelsAndComputeKineticEnergyOnEdges(stencil_tests.StencilTest):
    PROGRAM = interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges
    OUTPUTS = ("vn_ie", "z_kin_hor_e")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        wgtfac_e: np.ndarray,
        vn: np.ndarray,
        vt: np.ndarray,
        vn_ie: np.ndarray,
        z_kin_hor_e: np.ndarray,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
        vertical_start: gtx.int32,
        vertical_end: gtx.int32,
        **kwargs: Any,
    ) -> dict:
        subset = (slice(horizontal_start, horizontal_end), slice(vertical_start, vertical_end))
        vn_ie, z_kin_hor_e = vn_ie.copy(), z_kin_hor_e.copy()
        vn_ie[subset], z_kin_hor_e[subset] = (
            x[subset]
            for x in interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_numpy(
                wgtfac_e, vn, vt
            )
        )

        return dict(
            vn_ie=vn_ie,
            z_kin_hor_e=z_kin_hor_e,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        wgtfac_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        vt = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        vn_ie = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_kin_hor_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        return dict(
            wgtfac_e=wgtfac_e,
            vn=vn,
            vt=vt,
            vn_ie=vn_ie,
            z_kin_hor_e=z_kin_hor_e,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
