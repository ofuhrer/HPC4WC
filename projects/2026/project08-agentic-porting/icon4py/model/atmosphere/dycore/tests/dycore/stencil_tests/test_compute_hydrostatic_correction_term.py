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

from icon4py.model.atmosphere.dycore.stencils.compute_hydrostatic_correction_term import (
    compute_hydrostatic_correction_term,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest


def compute_hydrostatic_correction_term_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    theta_v: np.ndarray,
    ikoffset: np.ndarray,
    zdiff_gradp: np.ndarray,
    theta_v_ic: np.ndarray,
    inv_ddqz_z_full: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    grav_o_cpd: float,
) -> np.ndarray:
    def _apply_index_field(
        shape: tuple, to_index: np.ndarray, neighbor_table: np.ndarray, offset_field: np.ndarray
    ) -> tuple:
        indexed, indexed_p1 = np.zeros(shape), np.zeros(shape)
        for iprimary in range(shape[0]):
            for isparse in range(shape[1]):
                for ik in range(shape[2]):
                    indexed[iprimary, isparse, ik] = to_index[
                        neighbor_table[iprimary, isparse],
                        ik + offset_field[iprimary, isparse, ik],
                    ]
                    indexed_p1[iprimary, isparse, ik] = to_index[
                        neighbor_table[iprimary, isparse],
                        ik + offset_field[iprimary, isparse, ik] + 1,
                    ]
        return indexed, indexed_p1

    e2c = connectivities[dims.E2CDim]
    full_shape = ikoffset.shape

    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, -1)

    theta_v_at_kidx, _ = _apply_index_field(full_shape, theta_v, e2c, ikoffset)

    theta_v_ic_at_kidx, theta_v_ic_at_kidx_p1 = _apply_index_field(
        full_shape, theta_v_ic, e2c, ikoffset
    )

    inv_ddqz_z_full_at_kidx, _ = _apply_index_field(full_shape, inv_ddqz_z_full, e2c, ikoffset)

    z_theta1 = (
        theta_v_at_kidx[:, 0, :]
        + zdiff_gradp[:, 0, :]
        * (theta_v_ic_at_kidx[:, 0, :] - theta_v_ic_at_kidx_p1[:, 0, :])
        * inv_ddqz_z_full_at_kidx[:, 0, :]
    )

    z_theta2 = (
        theta_v_at_kidx[:, 1, :]
        + zdiff_gradp[:, 1, :]
        * (theta_v_ic_at_kidx[:, 1, :] - theta_v_ic_at_kidx_p1[:, 1, :])
        * inv_ddqz_z_full_at_kidx[:, 1, :]
    )

    z_hydro_corr = (
        grav_o_cpd
        * inv_dual_edge_length
        * (z_theta2 - z_theta1)
        * 4.0
        / ((z_theta1 + z_theta2) ** 2)
    )

    nlevels = theta_v.shape[1]
    return z_hydro_corr[:, nlevels - 1 : nlevels]


@pytest.mark.continuous_benchmarking
class TestComputeHydrostaticCorrectionTerm(StencilTest):
    OUTPUTS = ("z_hydro_corr",)
    PROGRAM = compute_hydrostatic_correction_term
    STATIC_PARAMS = {
        StandardStaticVariants.NONE: (),
        StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
        ),
        StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        theta_v: np.ndarray,
        ikoffset: np.ndarray,
        zdiff_gradp: np.ndarray,
        theta_v_ic: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        grav_o_cpd: float,
        **kwargs: Any,
    ) -> dict:
        z_hydro_corr = compute_hydrostatic_correction_term_numpy(
            connectivities=connectivities,
            theta_v=theta_v,
            ikoffset=ikoffset,
            zdiff_gradp=zdiff_gradp,
            theta_v_ic=theta_v_ic,
            inv_ddqz_z_full=inv_ddqz_z_full,
            inv_dual_edge_length=inv_dual_edge_length,
            grav_o_cpd=grav_o_cpd,
        )
        return dict(z_hydro_corr=z_hydro_corr)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        ikoffset = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32
        )
        rng = np.random.default_rng()
        for k in range(grid.num_levels):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            ikoffset.ndarray[:, :, k] = rng.integers(  # type: ignore[index]
                low=0 - k,
                high=grid.num_levels - k - 1,
                size=(ikoffset.shape[0], ikoffset.shape[1]),
            )

        theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        zdiff_gradp = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=ta.vpfloat
        )
        theta_v_ic = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        grav_o_cpd = ta.wpfloat("10.0")

        z_hydro_corr = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)

        z_hydro_corr = gtx.constructors.zeros(
            domain={
                dims.EdgeDim: (0, grid.num_edges),
                dims.KDim: (grid.num_levels - 1, grid.num_levels),
            },
            dtype=ta.vpfloat,
        )

        return dict(
            theta_v=theta_v,
            ikoffset=ikoffset,
            z_hydro_corr=z_hydro_corr,
            zdiff_gradp=zdiff_gradp,
            theta_v_ic=theta_v_ic,
            inv_ddqz_z_full=inv_ddqz_z_full,
            inv_dual_edge_length=inv_dual_edge_length,
            grav_o_cpd=grav_o_cpd,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=gtx.int32(grid.num_levels - 1),
            vertical_end=gtx.int32(grid.num_levels),
        )
