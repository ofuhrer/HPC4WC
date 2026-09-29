# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_term_for_vertical_velocity import (
    _compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.compute_tangential_wind import (
    _compute_tangential_wind,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _interpolate_to_half_levels(
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    x: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.vpfloat]:
    wgtfac_e_wp = astype(wgtfac_e, wpfloat)
    x_ie_wp = wgtfac_e_wp * x + (wpfloat("1.0") - wgtfac_e_wp) * x(KDim - 1)
    return concat_where(dims.KDim > 0, astype(x_ie_wp, vpfloat), astype(x, vpfloat))


@gtx.field_operator
def _compute_horizontal_kinetic_energy(
    vn: fa.EdgeKField[ta.wpfloat],
    vt: fa.EdgeKField[ta.vpfloat],
) -> fa.EdgeKField[ta.vpfloat]:
    z_kin_hor_e_wp = wpfloat("0.5") * (vn * vn + astype(vt * vt, wpfloat))
    return astype(z_kin_hor_e_wp, vpfloat)


@gtx.field_operator
def _compute_diagnostics_from_normal_wind(
    tangential_wind_on_half_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    tangential_wind = _compute_tangential_wind(vn, rbf_vec_coeff_e)
    horizontal_kinetic_energy_at_edges_on_model_levels = _compute_horizontal_kinetic_energy(
        vn, tangential_wind
    )
    vn_on_half_levels = _interpolate_to_half_levels(wgtfac_e, vn)

    tangential_wind_on_half_levels = (
        _interpolate_to_half_levels(wgtfac_e, tangential_wind)
        if not skip_compute_predictor_vertical_advection
        else tangential_wind_on_half_levels
    )

    contravariant_correction_at_edges_on_model_levels = concat_where(
        nflatlev <= dims.KDim,
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, tangential_wind),
        contravariant_correction_at_edges_on_model_levels,
    )

    w_at_vertices = _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(w, c_intp)
    horizontal_advection_of_w_at_edges_on_half_levels = (
        _compute_horizontal_advection_term_for_vertical_velocity(
            vn_on_half_levels,
            inv_dual_edge_length,
            w,
            tangential_wind_on_half_levels,
            inv_primal_edge_length,
            tangent_orientation,
            w_at_vertices,
        )
        if not skip_compute_predictor_vertical_advection
        else horizontal_advection_of_w_at_edges_on_half_levels
    )

    return (
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
        horizontal_advection_of_w_at_edges_on_half_levels,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_diagnostics_from_normal_wind(
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    wgtfacq_e: fa.EdgeKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Formerly known as fused_velocity_advection_stencil_1_to_7_predictor.

    This computes the derived horizontal wind components, kinetic energy, horizontal advection of the vertical velocity,
    and the contravariant correction.
    It also extrapolates vertical velocity at the top level.

    Agrs:
        - tangential_wind: tangential wind at model levels
        - tangential_wind_on_half_levels: tangential wind interpolated to half levels
        - vn_on_half_levels: normal wind interpolated to half levels
        - horizontal_kinetic_energy_at_edges_on_model_levels: horizontal kinetic energy computed at edge of model levels
        - contravariant_correction_at_edges_on_model_levels: contravariant metric correction at edge of model levels
        - horizontal_advection_of_w_at_edges_on_half_levels: horizontal advection for vertical velocity
        - vn: normal wind at edges
        - w: vertical wind at cell centers
        - rbf_vec_coeff_e: interpolation field (RBF vector coefficient on edges)
        - wgtfac_e: metrics field
        - ddxn_z_full: metrics field (derivative of topography in the normal direction)
        - ddxt_z_full: metrics field (derivative of topography in the tangential direction)
        - wgtfacq_e: metrics field (weights for interpolation)
        - c_intp: interpolation field
        - inv_dual_edge_length: inverse dual edge length
        - inv_primal_edge_length: inverse primal edge length
        - tangent_orientation: orientation of the edge with respect to the grid
        - skip_compute_predictor_vertical_advection: logical flag to skip the vertical advection
        - nflatlev: number of flat levels
        - horizontal_start: start index in the horizontal direction
        - horizontal_end: end index in the horizontal direction
        - vertical_start: start index in the vertical direction
        - vertical_end: end index in the vertical direction


    Returns:
        - tangential_wind
        - tangential_wind_on_half_levels
        - vn_on_half_levels
        - horizontal_kinetic_energy_at_edges_on_model_levels
        - contravariant_correction_at_edges_on_model_levels
        - horizontal_advection_of_w_at_edges_on_half_levels
    """

    _compute_diagnostics_from_normal_wind(
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        vn=vn,
        w=w,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        wgtfac_e=wgtfac_e,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        nflatlev=nflatlev,
        out=(
            tangential_wind,
            tangential_wind_on_half_levels,
            vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels,
            horizontal_advection_of_w_at_edges_on_half_levels,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )
    _extrapolate_at_top(
        wgtfacq_e,
        vn,
        out=vn_on_half_levels,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )
