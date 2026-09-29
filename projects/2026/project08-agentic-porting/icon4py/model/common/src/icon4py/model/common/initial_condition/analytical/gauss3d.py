# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common import constants as phy_const, dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import (
    geometry_attributes as geometry_meta,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.initial_condition.analytical import utils as testcases_utils
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.states import static_fields

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Gauss3DConfig:
    u0: float = 0.0
    t0: float = 300.0
    brunt_vais: float = 0.01
    # The default values are from mo_nh_testcases.f90 and mo_nh_testcases_nml.f90

    fortran_name_map: ClassVar[dict[str, str]] = {
        "nh_u0": "u0",
        "nh_t0": "t0",
        "nh_brunt_vais": "brunt_vais",
    }


def gauss3d(
    *,
    config: Gauss3DConfig,
    vertical_config: v_grid.VerticalGridConfig,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
) -> None:
    """
    Initial condition for Gauss 3D test case.

    The reference experiment config for this is
    exp.exclaim_gauss3d_sb.
    """
    allocator = model_backends.get_allocator(backend)
    array_ns = data_alloc.import_array_ns(allocator)

    geometry = static_fields.geometry
    metrics = static_fields.metrics
    primal_normal_x = geometry.get(geometry_meta.EDGE_NORMAL_U).ndarray
    inv_dual_edge_length = geometry.get(f"inverse_of_{geometry_meta.DUAL_EDGE_LENGTH}").ndarray
    edge_cell_distance = geometry.get(geometry_meta.EDGE_CELL_DISTANCE).ndarray
    primal_edge_length = geometry.get(geometry_meta.EDGE_LENGTH).ndarray
    cell_area = geometry.get(geometry_meta.CELL_AREA).ndarray
    geopot = phy_const.GRAV * metrics.get(metrics_attributes.Z_MC).ndarray
    z_ifc = metrics.get(metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL).ndarray
    exner_ref_mc = metrics.get(metrics_attributes.EXNER_REF_MC).ndarray
    d_exner_dz_ref_ic = metrics.get(metrics_attributes.D_EXNER_DZ_REF_IC).ndarray
    theta_ref_mc = metrics.get(metrics_attributes.THETA_REF_MC).ndarray
    theta_ref_ic = metrics.get(metrics_attributes.THETA_REF_IC).ndarray
    wgtfac_c = metrics.get(metrics_attributes.WGTFAC_C).ndarray
    ddqz_z_half = metrics.get(metrics_attributes.DDQZ_Z_HALF).ndarray
    zone_idx = testcases_utils.zone_indices(grid)

    num_edges = grid.num_edges
    num_levels = grid.num_levels

    u0 = config.u0
    t0 = config.t0
    brunt_vais = config.brunt_vais

    exner_ndarray = prognostic_state_now.exner.ndarray
    rho_ndarray = prognostic_state_now.rho.ndarray
    theta_v_ndarray = prognostic_state_now.theta_v.ndarray

    mask_array_edge_start_plus1_to_edge_end = array_ns.ones(num_edges, dtype=bool)
    mask_array_edge_start_plus1_to_edge_end[0 : zone_idx["end_edge_lateral_boundary_level_2"]] = (
        False
    )
    mask = array_ns.repeat(
        array_ns.expand_dims(mask_array_edge_start_plus1_to_edge_end, axis=-1),
        num_levels,
        axis=1,
    )
    u_field = array_ns.where(mask, u0, 0.0)
    prognostic_state_now.vn.ndarray[:, :] = u_field * primal_normal_x[:, array_ns.newaxis]

    for k_index in range(num_levels - 1, -1, -1):
        z_help = (brunt_vais / phy_const.GRAV) ** 2 * geopot[:, k_index]
        theta_v_ndarray[:, k_index] = t0 * array_ns.exp(z_help)

    if brunt_vais != 0.0:
        z_help = (brunt_vais / phy_const.GRAV) ** 2 * geopot[:, num_levels - 1]
        exner_ndarray[:, num_levels - 1] = (
            phy_const.GRAV / brunt_vais
        ) ** 2 / t0 / phy_const.CPD * (array_ns.exp(-z_help) - 1.0) + 1.0
    else:
        exner_ndarray[:, num_levels - 1] = 1.0 - geopot[:, num_levels - 1] / phy_const.CPD / t0

    testcases_utils.hydrostatic_adjustment_constant_thetav_ndarray(
        wgtfac_c=wgtfac_c,
        ddqz_z_half=ddqz_z_half,
        exner_ref_mc=exner_ref_mc,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        theta_ref_mc=theta_ref_mc,
        theta_ref_ic=theta_ref_ic,
        rho=rho_ndarray,
        exner=exner_ndarray,
        theta_v=theta_v_ndarray,
        num_levels=num_levels,
    )
    log.info("Hydrostatic adjustment (constant theta_v) computation completed.")

    _, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, allocator)

    prognostic_state_now.w.ndarray[:, :] = testcases_utils.init_w(
        grid=grid,
        z_ifc=z_ifc,
        inv_dual_edge_length=inv_dual_edge_length,
        edge_cell_distance=edge_cell_distance,
        primal_edge_length=primal_edge_length,
        cell_area=cell_area,
        vn=prognostic_state_now.vn.ndarray,
        vct_b=vct_b.ndarray,
        nlev=num_levels,
    )
    exchange.exchange(dims.CellDim, prognostic_state_now.w)
