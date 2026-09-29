# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import (
    abs,  # noqa: A004
    astype,
    maximum,
    minimum,
    neighbor_sum,
    where,
)
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.dycore.stencils.mo_math_divrot_rot_vertex_ri_dsl import (
    _mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C, E2C2EO, E2V, E2C2EODim, E2CDim, E2VDim, KDim
from icon4py.model.common.interpolation.stencils.interpolate_to_cell_center import (
    _interpolate_to_cell_center,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_advective_normal_wind_tendency(
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    upward_vorticity_at_vertices_on_model_levels: fa.VertexKField[ta.vpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, E2CDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
) -> fa.EdgeKField[ta.vpfloat]:
    #: intermediate variable horizontal_kinetic_energy_at_cells_on_model_levels is originally declared as z_ekinh in ICON
    horizontal_kinetic_energy_at_cells_on_model_levels = _interpolate_to_cell_center(
        horizontal_kinetic_energy_at_edges_on_model_levels, e_bln_c_s
    )
    horizontal_kinetic_energy_at_cells_on_model_levels = astype(
        horizontal_kinetic_energy_at_cells_on_model_levels, vpfloat
    )

    (
        contravariant_corrected_w_at_cells_on_model_levels_wp,
        ddqz_z_full_e_wp,
        tangential_wind_wp,
    ) = astype(
        (contravariant_corrected_w_at_cells_on_model_levels, ddqz_z_full_e, tangential_wind),
        wpfloat,
    )

    horizontal_advection = (
        horizontal_kinetic_energy_at_edges_on_model_levels
        * (coeff_gradekin[E2CDim(0)] - coeff_gradekin[E2CDim(1)])
        + coeff_gradekin[E2CDim(1)] * horizontal_kinetic_energy_at_cells_on_model_levels(E2C[1])
        - coeff_gradekin[E2CDim(0)] * horizontal_kinetic_energy_at_cells_on_model_levels(E2C[0])
    )

    vertical_advection = (
        neighbor_sum(
            c_lin_e * contravariant_corrected_w_at_cells_on_model_levels_wp(E2C), axis=E2CDim
        )
        * astype((vn_on_half_levels - vn_on_half_levels(KDim + 1)), wpfloat)
        / ddqz_z_full_e_wp
    )

    coriolis_term = tangential_wind_wp * (
        coriolis_frequency
        + astype(
            vpfloat("0.5")
            * neighbor_sum(upward_vorticity_at_vertices_on_model_levels(E2V), axis=E2VDim),
            wpfloat,
        )
    )
    normal_wind_advective_tendency_wp = -(horizontal_advection + vertical_advection + coriolis_term)

    return astype(normal_wind_advective_tendency_wp, vpfloat)


@gtx.field_operator
def _compute_extra_diffusion(
    vn: fa.EdgeKField[ta.wpfloat],
    upward_vorticity_at_vertices_on_model_levels: fa.VertexKField[ta.vpfloat],
    difcoef: fa.EdgeKField[ta.wpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2EODim], ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    gradient_of_divergence_of_vn = neighbor_sum(geofac_grdiv * vn(E2C2EO), axis=E2C2EODim)

    gradient_of_vorticity = (
        tangent_orientation
        * inv_primal_edge_length
        * astype(
            upward_vorticity_at_vertices_on_model_levels(E2V[1])
            - upward_vorticity_at_vertices_on_model_levels(E2V[0]),
            wpfloat,
        )
    )

    extra_diffusion_on_vn = (
        difcoef * area_edge * (gradient_of_divergence_of_vn + gradient_of_vorticity)
    )

    return extra_diffusion_on_vn


@gtx.field_operator
def _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_without_levelmask(
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, E2CDim], ta.wpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    upward_vorticity_at_vertices_on_model_levels: fa.VertexKField[ta.vpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2EODim], ta.wpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    cfl_w_limit: ta.vpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
) -> fa.EdgeKField[ta.vpfloat]:
    (
        contravariant_corrected_w_at_cells_on_model_levels_wp,
        ddqz_z_full_e_wp,
        normal_wind_advective_tendency_wp,
        cfl_w_limit_wp,
    ) = astype(
        (
            contravariant_corrected_w_at_cells_on_model_levels,
            ddqz_z_full_e,
            normal_wind_advective_tendency,
            cfl_w_limit,
        ),
        wpfloat,
    )

    #: intermediate variable contravariant_corrected_w_at_edges_on_model_levels is originally declared as w_con_e in ICON
    contravariant_corrected_w_at_edges_on_model_levels = neighbor_sum(
        c_lin_e * contravariant_corrected_w_at_cells_on_model_levels_wp(E2C), axis=E2CDim
    )
    difcoef = scalfac_exdiff * minimum(
        wpfloat("0.85") - cfl_w_limit_wp * dtime,
        abs(contravariant_corrected_w_at_edges_on_model_levels) * dtime / ddqz_z_full_e_wp
        - cfl_w_limit_wp * dtime,
    )
    normal_wind_advective_tendency_wp = where(
        abs(contravariant_corrected_w_at_edges_on_model_levels)
        > astype(cfl_w_limit * ddqz_z_full_e, wpfloat),
        normal_wind_advective_tendency_wp
        + _compute_extra_diffusion(
            vn=vn,
            upward_vorticity_at_vertices_on_model_levels=upward_vorticity_at_vertices_on_model_levels,
            difcoef=difcoef,
            area_edge=area_edge,
            geofac_grdiv=geofac_grdiv,
            tangent_orientation=tangent_orientation,
            inv_primal_edge_length=inv_primal_edge_length,
        ),
        normal_wind_advective_tendency_wp,
    )
    return astype(normal_wind_advective_tendency_wp, vpfloat)


@gtx.field_operator
def _compute_advection_in_horizontal_momentum(
    vn: fa.EdgeKField[ta.wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    cfl_w_limit: ta.vpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
    apply_extra_diffusion_on_vn: bool,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> fa.EdgeKField[ta.vpfloat]:
    upward_vorticity_at_vertices_on_model_levels = _mo_math_divrot_rot_vertex_ri_dsl(vn, geofac_rot)
    upward_vorticity_at_vertices_on_model_levels = astype(
        upward_vorticity_at_vertices_on_model_levels, vpfloat
    )

    normal_wind_advective_tendency = _compute_advective_normal_wind_tendency(
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        upward_vorticity_at_vertices_on_model_levels=upward_vorticity_at_vertices_on_model_levels,
        tangential_wind=tangential_wind,
        vn_on_half_levels=vn_on_half_levels,
        contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
        coriolis_frequency=coriolis_frequency,
        e_bln_c_s=e_bln_c_s,
        c_lin_e=c_lin_e,
        coeff_gradekin=coeff_gradekin,
        ddqz_z_full_e=ddqz_z_full_e,
    )

    if apply_extra_diffusion_on_vn:
        normal_wind_advective_tendency = concat_where(
            ((maximum(2, end_index_of_damping_layer - 2)) <= dims.KDim) & (dims.KDim < (nlev - 4)),
            _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl_without_levelmask(
                c_lin_e=c_lin_e,
                contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
                ddqz_z_full_e=ddqz_z_full_e,
                area_edge=area_edge,
                tangent_orientation=tangent_orientation,
                inv_primal_edge_length=inv_primal_edge_length,
                upward_vorticity_at_vertices_on_model_levels=upward_vorticity_at_vertices_on_model_levels,
                geofac_grdiv=geofac_grdiv,
                vn=vn,
                normal_wind_advective_tendency=normal_wind_advective_tendency,
                cfl_w_limit=cfl_w_limit,
                scalfac_exdiff=scalfac_exdiff,
                dtime=dtime,
            ),
            normal_wind_advective_tendency,
        )

    return normal_wind_advective_tendency


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_advection_in_horizontal_momentum(
    normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    contravariant_corrected_w_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    cfl_w_limit: ta.vpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
    apply_extra_diffusion_on_vn: bool,
    end_index_of_damping_layer: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Formerly known as fused_velocity_advection_stencil_19_to_20.

    This computes the horizontal advection in the horizontal momentum equation

    Args:
        - normal_wind_advective_tendency: horizontal advection tendency of the normal wind
        - vn: normal wind at edges
        - horizontal_kinetic_energy_at_edges_on_model_levels: horizontal kinetic energy at edges on model levels
        - horizontal_kinetic_energy_at_cells_on_model_levels: horizontal kinetic energy at cell centers on model levels
        - tangential_wind: tangential wind at model levels
        - coriolis_frequency: coriolis frequency parameter
        - contravariant_corrected_w_at_cells_on_model_levels: contravariant-corrected vertical velocity at model levels
        - vn_on_half_levels: normal wind on half levels
        - e_bln_c_s: interpolation field (edge-to-cell interpolation weights)
        - geofac_rot: metric field for rotor computation
        - coeff_gradekin: metrics field/coefficient for the gradient of kinematic energy
        - c_lin_e: metrics field for linear interpolation from cells to edges
        - ddqz_z_full_e: metrics field equal to vertical spacing
        - area_edge: area associated with each edge
        - tangent_orientation: orientation of the edge with respect to the grid
        - inv_primal_edge_length: inverse primal edge length
        - geofac_grdiv: metrics field used to compute the gradient of a divergence (of vn)
        - cfl_w_limit: CFL limit for vertical velocity
        - scalfac_exdiff: scalar factor for external diffusion
        - dtime: time step
        - apply_extra_diffusion_on_vn: option to apply extra diffusion to vn
        - end_index_of_damping_layer: vertical index where damping ends
        - horizontal_start: start index in the horizontal domain
        - horizontal_end: end index in the horizontal domain
        - vertical_start: start index in the vertical domain at model top
        - vertical_end: end index in the vertical domain at model bottom (or number of full/model vertical levels)

    Returns:
        - normal_wind_advective_tendency: horizontal advection tendency of the normal wind

    """

    _compute_advection_in_horizontal_momentum(
        vn=vn,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        tangential_wind=tangential_wind,
        coriolis_frequency=coriolis_frequency,
        contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
        vn_on_half_levels=vn_on_half_levels,
        e_bln_c_s=e_bln_c_s,
        geofac_rot=geofac_rot,
        coeff_gradekin=coeff_gradekin,
        c_lin_e=c_lin_e,
        ddqz_z_full_e=ddqz_z_full_e,
        area_edge=area_edge,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        geofac_grdiv=geofac_grdiv,
        cfl_w_limit=cfl_w_limit,
        scalfac_exdiff=scalfac_exdiff,
        dtime=dtime,
        apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
        nlev=vertical_end,
        end_index_of_damping_layer=end_index_of_damping_layer,
        out=normal_wind_advective_tendency,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
