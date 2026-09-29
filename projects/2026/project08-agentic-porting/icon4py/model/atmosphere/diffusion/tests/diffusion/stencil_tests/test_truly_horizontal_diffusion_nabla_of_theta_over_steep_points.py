# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.truly_horizontal_diffusion_nabla_of_theta_over_steep_points import (
    truly_horizontal_diffusion_nabla_of_theta_over_steep_points,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def truly_horizontal_diffusion_nabla_of_theta_over_steep_points_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    zd_vertoffset: np.ndarray,
    zd_diffcoef: np.ndarray,
    geofac_n2s_c: np.ndarray,
    geofac_n2s_nbh: np.ndarray,
    vcoef: np.ndarray,
    theta_v: np.ndarray,
    z_temp: np.ndarray,
    **kwargs,
) -> np.ndarray:
    c2e2c = connectivities[dims.C2E2CDim]
    full_shape = vcoef.shape

    geofac_n2s_nbh = np.expand_dims(geofac_n2s_nbh, axis=2)

    theta_v_at_zd_vertidx = np.zeros_like(vcoef)
    theta_v_at_zd_vertidx_p1 = np.zeros_like(vcoef)
    for ic in range(full_shape[0]):
        for isparse in range(full_shape[1]):
            for ik in range(full_shape[2]):
                theta_v_at_zd_vertidx[ic, isparse, ik] = theta_v[
                    c2e2c[ic, isparse], ik + zd_vertoffset[ic, isparse, ik]
                ]
                theta_v_at_zd_vertidx_p1[ic, isparse, ik] = theta_v[
                    c2e2c[ic, isparse], ik + zd_vertoffset[ic, isparse, ik] + 1
                ]

    sum_over = np.sum(
        geofac_n2s_nbh * (vcoef * theta_v_at_zd_vertidx + (1.0 - vcoef) * theta_v_at_zd_vertidx_p1),
        axis=1,
    )

    geofac_n2s_c = np.expand_dims(geofac_n2s_c, axis=1)  # add KDim
    z_temp = z_temp + zd_diffcoef * (theta_v * geofac_n2s_c + sum_over)
    return z_temp


class TestTrulyHorizontalDiffusionNablaOfThetaOverSteepPoints(StencilTest):
    PROGRAM = truly_horizontal_diffusion_nabla_of_theta_over_steep_points
    OUTPUTS = ("z_temp",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        zd_vertoffset: np.ndarray,
        zd_diffcoef: np.ndarray,
        geofac_n2s_c: np.ndarray,
        geofac_n2s_nbh: np.ndarray,
        vcoef: np.ndarray,
        theta_v: np.ndarray,
        z_temp: np.ndarray,
        **kwargs,
    ) -> dict:
        z_temp = truly_horizontal_diffusion_nabla_of_theta_over_steep_points_numpy(
            connectivities=connectivities,
            zd_vertoffset=zd_vertoffset,
            zd_diffcoef=zd_diffcoef,
            geofac_n2s_c=geofac_n2s_c,
            geofac_n2s_nbh=geofac_n2s_nbh,
            vcoef=vcoef,
            theta_v=theta_v,
            z_temp=z_temp,
        )
        return dict(z_temp=z_temp)

    @pytest.fixture
    def input_data(self, grid):
        zd_vertoffset = zero_field(grid, dims.CellDim, dims.C2E2CDim, dims.KDim, dtype=gtx.int32)
        rng = np.random.default_rng()
        for k in range(grid.num_levels):
            # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
            zd_vertoffset[:, :, k] = rng.integers(
                low=0 - k,
                high=grid.num_levels - k - 1,
                size=(zd_vertoffset.ndarray.shape[0], zd_vertoffset.ndarray.shape[1]),
            )

        zd_diffcoef = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        geofac_n2s_c = random_field(grid, dims.CellDim, dtype=wpfloat)
        geofac_n2s_nbh = random_field(grid, dims.CellDim, dims.C2E2CDim, dtype=wpfloat)
        vcoef = random_field(grid, dims.CellDim, dims.C2E2CDim, dims.KDim, dtype=wpfloat)
        theta_v = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        z_temp = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            zd_vertoffset=zd_vertoffset,
            zd_diffcoef=zd_diffcoef,
            geofac_n2s_c=geofac_n2s_c,
            geofac_n2s_nbh=geofac_n2s_nbh,
            theta_v=theta_v,
            z_temp=z_temp,
            vcoef=vcoef,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
