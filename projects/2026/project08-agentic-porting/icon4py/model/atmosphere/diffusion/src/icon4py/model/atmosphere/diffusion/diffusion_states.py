# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid


@dataclasses.dataclass(frozen=True)
class DiffusionDiagnosticState:
    """Represents the diagnostic fields needed in diffusion."""

    # fields for 3D elements in turbdiff
    hdef_ic: fa.CellKField[float]  # ! divergence at half levels(nproma,nlevp1,nblks_c)     [1/s]
    div_ic: fa.CellKField[
        float
    ]  # ! horizontal wind field deformation (nproma,nlevp1,nblks_c)     [1/s^2]
    dwdx: fa.CellKField[
        float
    ]  # zonal gradient of vertical wind speed (nproma,nlevp1,nblks_c)     [1/s]
    dwdy: fa.CellKField[float]  # meridional gradient of vertical wind speed (nproma,nlevp1,nblks_c)


@dataclasses.dataclass(frozen=True)
class DiffusionMetricState:
    """Represents the metric state fields needed in diffusion."""

    theta_ref_mc: fa.CellKField[float]
    wgtfac_c: fa.CellKField[
        float
    ]  # weighting factor for interpolation from full to half levels (nproma,nlevp1,nblks_c)
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32]
    zd_diffcoef: fa.CellKField[float]
    zd_intcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], float]


@dataclasses.dataclass(frozen=True)
class DiffusionInterpolationState:
    """Represents the ICON interpolation state needed in diffusion."""

    e_bln_c_s: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2EDim], float
    ]  # coefficent for bilinear interpolation from edge to cell ()
    rbf_coeff_1: gtx.Field[
        gtx.Dims[dims.VertexDim, dims.V2EDim], float
    ]  # rbf_vec_coeff_v_1(nproma, rbf_vec_dim_v, nblks_v)
    rbf_coeff_2: gtx.Field[
        gtx.Dims[dims.VertexDim, dims.V2EDim], float
    ]  # rbf_vec_coeff_v_2(nproma, rbf_vec_dim_v, nblks_v)

    geofac_div: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2EDim], float
    ]  # factor for divergence (nproma,cell_type,nblks_c)

    geofac_n2s: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2E2CODim], float
    ]  # factor for nabla2-scalar (nproma,cell_type+1,nblks_c)
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], float]
    geofac_grg_y: gtx.Field[
        gtx.Dims[dims.CellDim, dims.C2E2CODim], float
    ]  # factors for green gauss gradient (nproma,4,nblks_c,2)
    nudgecoeff_e: fa.EdgeField[float]  # Nudging coefficients for edges

    @functools.cached_property
    def geofac_n2s_c(self) -> fa.CellField[float]:
        return gtx.as_field((dims.CellDim,), data=self.geofac_n2s.ndarray[:, 0])

    @functools.cached_property
    def geofac_n2s_nbh(self) -> gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], float]:
        geofac_nbh_ar = self.geofac_n2s.ndarray[:, 1:]
        return gtx.as_field((dims.CellDim, dims.C2E2CDim), geofac_nbh_ar)


def initialize_diffusion_diagnostic_state(
    grid: icon_grid.IconGrid, allocator: gtx_typing.Allocator
) -> DiffusionDiagnosticState:
    hdef_ic = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.vpfloat,
    )
    div_ic = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.vpfloat,
    )
    dwdx = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.vpfloat,
    )
    dwdy = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.vpfloat,
    )
    return DiffusionDiagnosticState(
        hdef_ic=hdef_ic,
        div_ic=div_ic,
        dwdx=dwdx,
        dwdy=dwdy,
    )
