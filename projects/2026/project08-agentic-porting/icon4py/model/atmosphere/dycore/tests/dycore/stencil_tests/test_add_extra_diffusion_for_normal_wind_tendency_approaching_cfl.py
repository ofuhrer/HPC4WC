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

from icon4py.model.atmosphere.dycore.stencils.add_extra_diffusion_for_normal_wind_tendency_approaching_cfl import (
    add_extra_diffusion_for_normal_wind_tendency_approaching_cfl,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    levelmask: np.ndarray,
    c_lin_e: np.ndarray,
    z_w_con_c_full: np.ndarray,
    ddqz_z_full_e: np.ndarray,
    area_edge: np.ndarray,
    tangent_orientation: np.ndarray,
    inv_primal_edge_length: np.ndarray,
    zeta: np.ndarray,
    geofac_grdiv: np.ndarray,
    vn: np.ndarray,
    ddt_vn_apc: np.ndarray,
    cfl_w_limit: ta.wpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
) -> np.ndarray:
    w_con_e = np.zeros_like(vn)
    difcoef = np.zeros_like(vn)

    levelmask_offset_0 = levelmask[:-1]
    levelmask_offset_1 = levelmask[1:]

    c_lin_e = np.expand_dims(c_lin_e, axis=-1)
    geofac_grdiv = np.expand_dims(geofac_grdiv, axis=-1)
    area_edge = np.expand_dims(area_edge, axis=-1)
    tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)

    e2c = connectivities[dims.E2CDim]
    w_con_e = np.where(
        (levelmask_offset_0) | (levelmask_offset_1),
        np.sum(
            np.where(
                (e2c != -1)[:, :, np.newaxis],
                c_lin_e * z_w_con_c_full[e2c],
                0,
            ),
            axis=1,
        ),
        w_con_e,
    )
    difcoef = np.where(
        ((levelmask_offset_0) | (levelmask_offset_1))
        & (np.abs(w_con_e) > cfl_w_limit * ddqz_z_full_e),
        scalfac_exdiff
        * np.minimum(
            0.85 - cfl_w_limit * dtime,
            np.abs(w_con_e) * dtime / ddqz_z_full_e - cfl_w_limit * dtime,
        ),
        difcoef,
    )
    e2v = connectivities[dims.E2VDim]
    e2c2eo = connectivities[dims.E2C2EODim]
    ddt_vn_apc = np.where(
        ((levelmask_offset_0) | (levelmask_offset_1))
        & (np.abs(w_con_e) > cfl_w_limit * ddqz_z_full_e),
        ddt_vn_apc
        + difcoef
        * area_edge
        * (
            np.sum(
                np.where(
                    (e2c2eo != -1)[:, :, np.newaxis],
                    geofac_grdiv * vn[e2c2eo],
                    0,
                ),
                axis=1,
            )
            + tangent_orientation * inv_primal_edge_length * (zeta[e2v][:, 1] - zeta[e2v][:, 0])
        ),
        ddt_vn_apc,
    )
    return ddt_vn_apc


@pytest.mark.embedded_remap_error
class TestAddExtraDiffusionForNormalWindTendencyApproachingCfl(StencilTest):
    PROGRAM = add_extra_diffusion_for_normal_wind_tendency_approaching_cfl
    OUTPUTS = ("ddt_vn_apc",)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        levelmask = data_alloc.random_mask(grid, dims.KDim, extend={dims.KDim: 1})
        c_lin_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat)
        z_w_con_c_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        ddqz_z_full_e = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat, low=0.0
        )
        area_edge = data_alloc.random_field(grid, dims.EdgeDim)
        tangent_orientation = data_alloc.random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        zeta = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=ta.vpfloat)
        geofac_grdiv = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EODim)
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddt_vn_apc = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        cfl_w_limit = ta.vpfloat("4.0")
        scalfac_exdiff = 6.0
        dtime = 2.0
        return dict(
            levelmask=levelmask,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            zeta=zeta,
            geofac_grdiv=geofac_grdiv,
            vn=vn,
            ddt_vn_apc=ddt_vn_apc,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
            c_lin_e=c_lin_e,
            z_w_con_c_full=z_w_con_c_full,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        levelmask: np.ndarray,
        c_lin_e: np.ndarray,
        z_w_con_c_full: np.ndarray,
        ddqz_z_full_e: np.ndarray,
        area_edge: np.ndarray,
        tangent_orientation: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        zeta: np.ndarray,
        geofac_grdiv: np.ndarray,
        vn: np.ndarray,
        ddt_vn_apc: np.ndarray,
        cfl_w_limit: ta.wpfloat,
        scalfac_exdiff: ta.wpfloat,
        dtime: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        ddt_vn_apc = add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_numpy(
            connectivities=connectivities,
            levelmask=levelmask,
            c_lin_e=c_lin_e,
            z_w_con_c_full=z_w_con_c_full,
            ddqz_z_full_e=ddqz_z_full_e,
            area_edge=area_edge,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
            zeta=zeta,
            geofac_grdiv=geofac_grdiv,
            vn=vn,
            ddt_vn_apc=ddt_vn_apc,
            cfl_w_limit=cfl_w_limit,
            scalfac_exdiff=scalfac_exdiff,
            dtime=dtime,
        )
        return dict(ddt_vn_apc=ddt_vn_apc)
