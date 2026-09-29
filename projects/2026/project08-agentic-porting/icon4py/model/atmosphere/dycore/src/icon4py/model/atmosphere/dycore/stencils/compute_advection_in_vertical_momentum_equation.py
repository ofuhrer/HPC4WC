# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import abs, astype, broadcast, maximum, where  # noqa: A004
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.dycore.stencils.add_extra_diffusion_for_w_con_approaching_cfl import (
    _add_extra_diffusion_for_w_con_approaching_cfl,
)
from icon4py.model.atmosphere.dycore.stencils.add_interpolated_horizontal_advection_of_w import (
    _add_interpolated_horizontal_advection_of_w,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_term_for_vertical_velocity import (
    _compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    _mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_vp import (
    _interpolate_cell_field_to_half_levels_vp,
)
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center import (
    _interpolate_to_cell_center,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _interpolate_contravariant_vertical_velocity_to_full_levels(
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[vpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[vpfloat]:
    # TODO(havogt): Note that `concat_where(dims.KDim == nlev-1, ...)` is currently broken
    # because of insufficiency in the domain inference of GT4Py,
    # see https://github.com/GridTools/gt4py/issues/2205.
    return concat_where(
        dims.KDim < nlev - 1,
        vpfloat("0.5")
        * (
            contravariant_corrected_w_at_cells_on_half_levels
            + contravariant_corrected_w_at_cells_on_half_levels(KDim + 1)
        ),
        vpfloat("0.5") * contravariant_corrected_w_at_cells_on_half_levels,
    )


@gtx.field_operator
def _compute_horizontal_advection_of_w(
    w: fa.CellKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeKField[ta.vpfloat]:
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
    )

    return astype(horizontal_advection_of_w_at_edges_on_half_levels, vpfloat)


@gtx.field_operator
def _add_vertical_advection_of_w_to_advective_vertical_wind_tendency(
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    contravariant_corrected_w_at_cells_on_half_levels_wp = astype(
        contravariant_corrected_w_at_cells_on_half_levels, wpfloat
    )
    coeff1_dwdz_wp, coeff2_dwdz_wp = astype((coeff1_dwdz, coeff2_dwdz), wpfloat)

    vertical_wind_advective_tendency_wp = -contravariant_corrected_w_at_cells_on_half_levels_wp * (
        w(KDim - 1) * coeff1_dwdz_wp
        - w(KDim + 1) * coeff2_dwdz_wp
        + w * astype(coeff2_dwdz - coeff1_dwdz, wpfloat)
    )
    return astype(vertical_wind_advective_tendency_wp, vpfloat)


@gtx.field_operator
def _compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[bool],
    fa.CellKField[ta.vpfloat],
]:
    contravariant_corrected_w_at_cells_on_half_levels_wp, ddqz_z_half_wp = astype(
        (contravariant_corrected_w_at_cells_on_half_levels, ddqz_z_half), wpfloat
    )

    cfl_clipping = where(
        abs(contravariant_corrected_w_at_cells_on_half_levels) > cfl_w_limit * ddqz_z_half,
        broadcast(True, (dims.CellDim, dims.KDim)),
        False,
    )

    vertical_cfl = where(
        cfl_clipping,
        contravariant_corrected_w_at_cells_on_half_levels_wp * dtime / ddqz_z_half_wp,
        broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
    )
    vertical_cfl_vp = astype(vertical_cfl, vpfloat)

    contravariant_corrected_w_at_cells_on_half_levels_wp = where(
        (cfl_clipping) & (vertical_cfl_vp < -vpfloat("0.85")),
        astype(-vpfloat("0.85") * ddqz_z_half, wpfloat) / dtime,
        contravariant_corrected_w_at_cells_on_half_levels_wp,
    )

    contravariant_corrected_w_at_cells_on_half_levels_wp = where(
        (cfl_clipping) & (vertical_cfl_vp > vpfloat("0.85")),
        astype(vpfloat("0.85") * ddqz_z_half, wpfloat) / dtime,
        contravariant_corrected_w_at_cells_on_half_levels_wp,
    )

    return (
        astype(contravariant_corrected_w_at_cells_on_half_levels_wp, vpfloat),
        cfl_clipping,
        vertical_cfl_vp,
    )


@gtx.field_operator
def _compute_contravariant_corrected_w(
    w: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
) -> fa.CellKField[ta.vpfloat]:
    contravariant_corrected_w_at_cells_on_half_levels = (
        astype(w, vpfloat) - contravariant_correction_at_cells_on_half_levels
    )

    return contravariant_corrected_w_at_cells_on_half_levels


@gtx.field_operator
def _compute_contravariant_corrected_w_and_cfl(
    w: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[fa.CellKField[ta.vpfloat], fa.CellKField[bool], fa.CellKField[ta.vpfloat]]:
    #: intermediate variable contravariant_corrected_w_at_cells_on_half_levels is originally declared as z_w_con_c in ICON
    contravariant_corrected_w_at_cells_on_half_levels = _compute_contravariant_corrected_w(
        w, contravariant_correction_at_cells_on_half_levels
    )

    (contravariant_corrected_w_at_cells_on_half_levels, cfl_clipping, vertical_cfl) = concat_where(
        (dims.KDim >= maximum(2, end_index_of_damping_layer - 2)) & (dims.KDim < nlev - 3),
        _compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
            ddqz_z_half=ddqz_z_half,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
        ),
        (
            contravariant_corrected_w_at_cells_on_half_levels,
            broadcast(False, (dims.CellDim, dims.KDim)),
            broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim)),
        ),
    )

    return contravariant_corrected_w_at_cells_on_half_levels, cfl_clipping, vertical_cfl


@gtx.field_operator
def _compute_advective_vertical_wind_tendency(
    vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.wpfloat],
    contravariant_corrected_w_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    cfl_clipping: fa.CellKField[bool],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
) -> fa.CellKField[ta.vpfloat]:
    vertical_wind_advective_tendency = concat_where(
        1 <= dims.KDim,
        _add_vertical_advection_of_w_to_advective_vertical_wind_tendency(
            contravariant_corrected_w_at_cells_on_half_levels, w, coeff1_dwdz, coeff2_dwdz
        ),
        vertical_wind_advective_tendency,
    )

    vertical_wind_advective_tendency = concat_where(
        1 <= dims.KDim,
        _add_interpolated_horizontal_advection_of_w(
            e_bln_c_s,
            horizontal_advection_of_w_at_edges_on_half_levels,
            vertical_wind_advective_tendency,
        ),
        vertical_wind_advective_tendency,
    )

    vertical_wind_advective_tendency = _add_extra_diffusion_for_w_con_approaching_cfl(
        cfl_clipping,
        owner_mask,
        contravariant_corrected_w_at_cells_on_half_levels,
        ddqz_z_half,
        area,
        geofac_n2s,
        w,
        vertical_wind_advective_tendency,
        scalfac_exdiff,
        cfl_w_limit,
        dtime,
    )

    return vertical_wind_advective_tendency


@gtx.field_operator
def _compute_advection_in_corrector_vertical_momentum(
    vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[fa.CellKField[ta.vpfloat], fa.CellKField[ta.vpfloat], fa.CellKField[ta.vpfloat]]:
    #: intermediate variable horizontal_advection_of_w_at_edges_on_half_levels is originally declared as z_v_grad_w in ICON
    horizontal_advection_of_w_at_edges_on_half_levels = _compute_horizontal_advection_of_w(
        w=w,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
    )

    (
        contravariant_corrected_w_at_cells_on_half_levels,
        cfl_clipping,
        vertical_cfl,
    ) = _compute_contravariant_corrected_w_and_cfl(
        w=w,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        nlev=nlev,
        end_index_of_damping_layer=end_index_of_damping_layer,
    )

    vertical_wind_advective_tendency = _compute_advective_vertical_wind_tendency(
        vertical_wind_advective_tendency=vertical_wind_advective_tendency,
        w=w,
        horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
        cfl_clipping=cfl_clipping,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        e_bln_c_s=e_bln_c_s,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        owner_mask=owner_mask,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
    )

    contravariant_corrected_w_at_cells_on_model_levels = (
        _interpolate_contravariant_vertical_velocity_to_full_levels(
            contravariant_corrected_w_at_cells_on_half_levels, nlev
        )
    )

    return (
        vertical_wind_advective_tendency,
        contravariant_corrected_w_at_cells_on_model_levels,
        vertical_cfl,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_advection_in_corrector_vertical_momentum(
    vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    vertical_cfl: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    end_index_of_damping_layer: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    This computes the vertical momentum advection in the vertical momentum equation

    Args:
        - vertical_wind_advective_tendency: vertical advection tendency of the vertical wind
        - contravariant_corrected_w_at_cells_on_model_levels: contravariant-corrected vertical velocity at cells on model levels
        - vertical_cfl: vertical cfl number at cells on half levels
        - w: vertical wind at cell centers
        - tangential_wind: tangential wind at edges on model levels
        - vn_on_half_levels: normal wind at edges on half levels
        - contravariant_correction_at_edges_on_model_levels: contravariant correction at edges on model levels
        - coeff1_dwdz: metrics field (first coefficient for vertical derivative of vertical wind)
        - coeff2_dwdz: metrics field (second coefficient for vertical derivative of vertical wind)
        - c_intp: interpolation field for cell-to-vertex interpolation
        - inv_dual_edge_length: inverse dual edge length
        - inv_primal_edge_length: inverse primal edge length
        - tangent_orientation: orientation of the edge with respect to the grid
        - e_bln_c_s: interpolation field (edge-to-cell interpolation weights)
        - ddqz_z_half: metrics field
        - area: cell area
        - geofac_n2s: interpolation field
        - owner_mask: ownership mask for each cell
        - scalfac_exdiff: scalar factor for external diffusion
        - cfl_w_limit: CFL limit for vertical velocity
        - dtime: time step
        - nflatlev: number of flat levels
        - end_index_of_damping_layer: vertical index where damping ends
        - horizontal_start: start index in the horizontal dimension
        - horizontal_end: end index in the horizontal dimension
        - vertical_start: start index in the vertical dimension at model top
        - vertical_end: end index in the vertical dimension at model bottom (or number of full/model vertical levels)

    Returns:
        - vertical_wind_advective_tendency
        - contravariant_corrected_w_at_cells_on_model_levels
        - vertical_cfl
    """

    _compute_advection_in_corrector_vertical_momentum(
        vertical_wind_advective_tendency=vertical_wind_advective_tendency,
        w=w,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        e_bln_c_s=e_bln_c_s,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        owner_mask=owner_mask,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        nlev=vertical_end,
        end_index_of_damping_layer=end_index_of_damping_layer,
        out=(
            vertical_wind_advective_tendency,
            contravariant_corrected_w_at_cells_on_model_levels,
            vertical_cfl,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _interpolate_contravariant_correction_to_cells_on_half_levels(
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    nflatlev: gtx.int32,
) -> fa.CellKField[ta.vpfloat]:
    contravariant_correction_at_cells_model_levels = _interpolate_to_cell_center(
        contravariant_correction_at_edges_on_model_levels, e_bln_c_s
    )
    contravariant_correction_at_cells_model_levels = astype(
        contravariant_correction_at_cells_model_levels, vpfloat
    )

    contravariant_correction_at_cells_on_half_levels = concat_where(
        dims.KDim >= nflatlev + 1,
        _interpolate_cell_field_to_half_levels_vp(
            wgtfac_c=wgtfac_c, interpolant=contravariant_correction_at_cells_model_levels
        ),
        broadcast(vpfloat("0.0"), (dims.CellDim, dims.KDim)),
    )

    return contravariant_correction_at_cells_on_half_levels


@gtx.field_operator
def _compute_advection_in_predictor_vertical_momentum(
    vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.wpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
]:
    contravariant_correction_at_cells_on_half_levels = _interpolate_contravariant_correction_to_cells_on_half_levels(
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        nflatlev=nflatlev,
    )

    (
        contravariant_corrected_w_at_cells_on_half_levels,
        cfl_clipping,
        vertical_cfl,
    ) = _compute_contravariant_corrected_w_and_cfl(
        w=w,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        nlev=nlev,
        end_index_of_damping_layer=end_index_of_damping_layer,
    )

    if not skip_compute_predictor_vertical_advection:
        vertical_wind_advective_tendency = _compute_advective_vertical_wind_tendency(
            vertical_wind_advective_tendency=vertical_wind_advective_tendency,
            w=w,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
            contravariant_corrected_w_at_cells_on_half_levels=contravariant_corrected_w_at_cells_on_half_levels,
            cfl_clipping=cfl_clipping,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            e_bln_c_s=e_bln_c_s,
            ddqz_z_half=ddqz_z_half,
            area=area,
            geofac_n2s=geofac_n2s,
            owner_mask=owner_mask,
            scalfac_exdiff=scalfac_exdiff,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
        )

    contravariant_corrected_w_at_cells_on_model_levels = (
        _interpolate_contravariant_vertical_velocity_to_full_levels(
            contravariant_corrected_w_at_cells_on_half_levels, nlev
        )
    )

    return (
        contravariant_correction_at_cells_on_half_levels,
        vertical_wind_advective_tendency,
        contravariant_corrected_w_at_cells_on_model_levels,
        vertical_cfl,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_advection_in_predictor_vertical_momentum(
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    vertical_cfl: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    horizontal_advection_of_w_at_edges_on_half_levels: fa.EdgeKField[ta.wpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    This computes the vertical momentum advection in the vertical momentum equation

    Args:
        - contravariant_correction_at_cells_on_half_levels: contravariant correction at cells on model levels
        - vertical_wind_advective_tendency: vertical advection tendency of the vertical wind
        - contravariant_corrected_w_at_cells_on_model_levels: contravariant-corrected vertical velocity at cells on model levels
        - vertical_cfl: vertical cfl number at cells on half levels
        - w: vertical wind at cells on half levels
        - horizontal_advection_of_w_at_edges_on_half_levels: horizontal advection of w at edges on half levels
        - contravariant_correction_at_edges_on_model_levels: contravariant correction at edges on model levels
        - coeff1_dwdz: metrics field (first coefficient for vertical derivative of vertical wind)
        - coeff2_dwdz: metrics field (second coefficient for vertical derivative of vertical wind)
        - e_bln_c_s: interpolation field (edge-to-cell interpolation weights)
        - wgtfac_c: metric coefficient for interpolating a cell variable from full to half levels
        - ddqz_z_half: metrics field
        - area: cell area
        - geofac_n2s: interpolation field
        - owner_mask: ownership mask for each cell
        - scalfac_exdiff: scalar factor for external diffusion
        - cfl_w_limit: CFL limit for vertical velocity
        - dtime: time step
        - skip_compute_predictor_vertical_advection: logical flag to skip the vertical advection
        - nflatlev: number of flat levels
        - end_index_of_damping_layer: vertical index where damping ends
        - horizontal_start: start index in the horizontal dimension
        - horizontal_end: end index in the horizontal dimension
        - vertical_start: start index in the vertical dimension at model top
        - vertical_end: end index in the vertical dimension at model bottom (or number of full/model vertical levels)

    Returns:
        - contravariant_correction_at_cells_on_half_levels
        - vertical_wind_advective_tendency
        - contravariant_corrected_w_at_cells_on_model_levels
        - vertical_cfl
    """

    _compute_advection_in_predictor_vertical_momentum(
        vertical_wind_advective_tendency=vertical_wind_advective_tendency,
        w=w,
        horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        owner_mask=owner_mask,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        nflatlev=nflatlev,
        nlev=vertical_end,
        end_index_of_damping_layer=end_index_of_damping_layer,
        out=(
            contravariant_correction_at_cells_on_half_levels,
            vertical_wind_advective_tendency,
            contravariant_corrected_w_at_cells_on_model_levels,
            vertical_cfl,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
