# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Contains metric fields calculations for the vertical grid, ported from mo_vertical_grid.f90."""

from __future__ import annotations

from collections.abc import Callable

import gt4py.next as gtx
from gt4py.next import (
    abs,  # noqa: A004
    astype,
    broadcast,
    int32,
    max_over,
    maximum,
    minimum,
    neighbor_sum,
    sin,
    tanh,
    where,
)
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.dimension import C2E, C2E2C, C2E2CO, E2C, C2E2CODim, KDim
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    _cell_2_edge_interpolation,
)
from icon4py.model.common.interpolation.stencils.compute_cell_2_vertex_interpolation import (
    _compute_cell_2_vertex_interpolation,
)
from icon4py.model.common.math.gradient import _grad_fd_tang, grad_fd_norm
from icon4py.model.common.math.vertical_operations import (
    difference_level_plus1_on_cells,
    with_boundaries_on_half_levels_on_cells,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc


# TODO(nfarabullini): ddqz_z_half vertical dimension is khalf, use K2KHalf once merged for z_ifc and z_mc
# TODO(nfarabullini): change dimension type hint for ddqz_z_half to cell, khalf
@gtx.field_operator
def _compute_ddqz_z_half(
    z_ifc: fa.CellKField[wpfloat],
    z_mc: fa.CellKField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    return with_boundaries_on_half_levels_on_cells(
        top=2.0 * (z_ifc - z_mc),
        interior=z_mc(KDim - 1) - z_mc,
        bottom=2.0 * (z_mc(KDim - 1) - z_ifc),
        nlev=nlev,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED, backend=None)
def compute_ddqz_z_half(  # noqa: PLR0917 [too-many-positional-arguments]
    z_ifc: fa.CellKField[wpfloat],
    z_mc: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[wpfloat],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute functional determinant of the metrics (is positive) on half levels.

    See mo_vertical_grid.f90

    Args:
        z_ifc: geometric height on half levels
        z_mc: geometric height on full levels
        k: vertical dimension index
        nlev: total number of levels
        ddqz_z_half: (output) functional determinant of the metrics (is positive), half levels
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_ddqz_z_half(
        z_ifc,
        z_mc,
        nlev,
        out=ddqz_z_half,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_ddqz_z_full_and_inverse(
    z_ifc: fa.CellKField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    ddqz_z_full = difference_level_plus1_on_cells(z_ifc)
    inverse_ddqz_z_full = 1.0 / ddqz_z_full
    return ddqz_z_full, inverse_ddqz_z_full


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ddqz_z_full_and_inverse(  # noqa: PLR0917 [too-many-positional-arguments]
    z_ifc: fa.CellKField[wpfloat],
    ddqz_z_full: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute ddqz_z_full and its inverse inv_ddqz_z_full.

    Functional determinant of the metrics (is positive) on full levels and inverse inverse layer thickness(for runtime optimization).
    See mo_vertical_grid.f90

    Args:
        z_ifc: geometric height on half levels
        ddqz_z_full: (output) functional determinant of the metrics (is positive), full levels
        inv_ddqz_z_full: (output) inverse layer thickness (for runtime optimization)
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index

    """
    _compute_ddqz_z_full_and_inverse(
        z_ifc,
        out=(ddqz_z_full, inv_ddqz_z_full),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_scaling_factor_for_3d_divdamp(
    vct_a: fa.KField[wpfloat],
    divdamp_trans_start: wpfloat,
    divdamp_trans_end: wpfloat,
    divdamp_type: gtx.int32,
) -> fa.KField[wpfloat]:
    scaling_factor_for_3d_divdamp = broadcast(1.0, (dims.KDim,))
    if divdamp_type == 32:
        zf = 0.5 * (vct_a + vct_a(KDim + 1))  # depends on nshift_total, assumed to be always 0
        scaling_factor_for_3d_divdamp = where(
            zf >= divdamp_trans_end, 0.0, scaling_factor_for_3d_divdamp
        )
        scaling_factor_for_3d_divdamp = where(
            zf >= divdamp_trans_start,
            (divdamp_trans_end - zf) / (divdamp_trans_end - divdamp_trans_start),
            scaling_factor_for_3d_divdamp,
        )
    return scaling_factor_for_3d_divdamp


@gtx.program
def compute_scaling_factor_for_3d_divdamp(  # noqa: PLR0917 [too-many-positional-arguments]
    vct_a: fa.KField[wpfloat],
    scaling_factor_for_3d_divdamp: fa.KField[wpfloat],
    divdamp_trans_start: wpfloat,
    divdamp_trans_end: wpfloat,
    divdamp_type: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute scaling factor for 3D divergence damping terms (declared as scalfac_dd3d in ICON).

    See mo_vertical_grid.f90

    Args:
        vct_a: Field[Dims[dims.KDim], float],
        scaling_factor_for_3d_divdamp: (output) scaling factor for 3D divergence damping terms, and start level from which they are > 0
        divdamp_trans_start: lower bound of transition zone between 2D and 3D div damping in case of divdamp_type = 32
        divdamp_trans_end: upper bound of transition zone between 2D and 3D div damping in case of divdamp_type = 32
        divdamp_type: type of divergence damping (2D or 3D divergence)
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_scaling_factor_for_3d_divdamp(
        vct_a,
        divdamp_trans_start,
        divdamp_trans_end,
        divdamp_type,
        out=scaling_factor_for_3d_divdamp,
        domain={dims.KDim: (vertical_start, vertical_end)},
    )


@gtx.field_operator
def _compute_rayleigh_w(  # noqa: PLR0917 [too-many-positional-arguments]
    vct_a: fa.KField[wpfloat],
    damping_height: wpfloat,
    rayleigh_type: gtx.int32,
    rayleigh_coeff: wpfloat,
    vct_a_1: wpfloat,
    pi_const: wpfloat,
) -> fa.KField[wpfloat]:
    rayleigh_w = broadcast(0.0, (dims.KDim,))
    z_sin_diff = maximum(0.0, vct_a - damping_height)
    z_tanh_diff = vct_a_1 - vct_a  # vct_a(1) - vct_a
    if rayleigh_type == 1:  # RayleighType.CLASSIC
        rayleigh_w = (
            rayleigh_coeff
            * (sin(pi_const / 2.0 * z_sin_diff / maximum(0.001, vct_a_1 - damping_height))) ** 2
        )

    elif rayleigh_type == 2:  # RayleighType.KLEMP
        rayleigh_w = rayleigh_coeff * (
            1.0 - tanh(3.8 * z_tanh_diff / maximum(0.000001, vct_a_1 - damping_height))
        )
    return rayleigh_w


@gtx.program
def compute_rayleigh_w(  # noqa: PLR0917 [too-many-positional-arguments]
    rayleigh_w: fa.KField[wpfloat],
    vct_a: fa.KField[wpfloat],
    damping_height: wpfloat,
    rayleigh_type: gtx.int32,
    rayleigh_coeff: wpfloat,
    vct_a_1: wpfloat,
    pi_const: wpfloat,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute rayleigh_w factor.

    See mo_vertical_grid.f90

    Args:
        rayleigh_w: (output) Rayleigh damping
        vct_a: Field[Dims[dims.KDim], float]
        vct_a_1: 1D of vct_a
        damping_height: height at which w-damping and sponge layer start
        rayleigh_type: type of Rayleigh damping (1: CLASSIC, 2: Klemp (2008))
        rayleigh_classic: classical Rayleigh damping, which makes use of a reference state.
        rayleigh_klemp: Klemp (2008) type Rayleigh damping
        rayleigh_coeff: Rayleigh damping coefficient in w-equation
        pi_const: pi constant
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_rayleigh_w(
        vct_a,
        damping_height,
        rayleigh_type,
        rayleigh_coeff,
        vct_a_1,
        pi_const,
        out=rayleigh_w,
        domain={dims.KDim: (vertical_start, vertical_end)},
    )


@gtx.field_operator
def _compute_coeff_dwdz(
    ddqz_z_full: fa.CellKField[wpfloat], z_ifc: fa.CellKField[wpfloat]
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    coeff1_dwdz = ddqz_z_full / ddqz_z_full(KDim - 1) / (z_ifc(KDim - 1) - z_ifc(KDim + 1))
    coeff2_dwdz = ddqz_z_full(KDim - 1) / ddqz_z_full / (z_ifc(KDim - 1) - z_ifc(KDim + 1))

    return coeff1_dwdz, coeff2_dwdz


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_coeff_dwdz(  # noqa: PLR0917 [too-many-positional-arguments]
    ddqz_z_full: fa.CellKField[wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute coeff1_dwdz and coeff2_dwdz factors.

    See mo_vertical_grid.f90

    Args:
        ddqz_z_full: functional determinant of the metrics (is positive), full levels
        z_ifc: geometric height of half levels
        coeff1_dwdz: coefficient for second-order acurate dw/dz term
        coeff2_dwdz: coefficient for second-order acurate dw/dz term
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """

    _compute_coeff_dwdz(
        ddqz_z_full,
        z_ifc,
        out=(coeff1_dwdz, coeff2_dwdz),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program
def compute_ddxn_z_half_e(  # noqa: PLR0917 [too-many-positional-arguments]
    z_ifc: fa.CellKField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    ddxn_z_half_e: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    grad_fd_norm(
        z_ifc,
        inv_dual_edge_length,
        out=ddxn_z_half_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_ddxt_z_half_e(
    cell_in: fa.CellKField[wpfloat],
    c_int: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
):
    z_ifv = _compute_cell_2_vertex_interpolation(cell_in, c_int)
    ddxt_z_half_e = _grad_fd_tang(
        z_ifv,
        inv_primal_edge_length,
        tangent_orientation,
    )
    return ddxt_z_half_e


@gtx.program
def compute_ddxt_z_half_e(  # noqa: PLR0917 [too-many-positional-arguments]
    cell_in: fa.CellKField[wpfloat],
    c_int: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    ddxt_z_half_e: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_ddxt_z_half_e(
        cell_in,
        c_int,
        inv_primal_edge_length,
        tangent_orientation,
        out=ddxt_z_half_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_exner_w_explicit_weight_parameter(
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    return 1.0 - exner_w_implicit_weight_parameter


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_exner_w_explicit_weight_parameter(
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    """
    Compute exner_w_explicit_weight_parameter.

    See mo_vertical_grid.f90

    Args:
        exner_w_implicit_weight_parameter: offcentering in vertical mass flux
        exner_w_explicit_weight_parameter: (output) 1 - exner_w_implicit_weight_parameter
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index

    """

    _compute_exner_w_explicit_weight_parameter(
        exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
        out=exner_w_explicit_weight_parameter,
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def _compute_maxslp_maxhgtd(
    ddxn_z_full: fa.EdgeKField[wpfloat],
    dual_edge_length: fa.EdgeField[wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    tmp = abs(ddxn_z_full)
    maxslp = max_over(tmp(C2E), axis=dims.C2EDim)

    tmp_maxhgtd = abs(ddxn_z_full * dual_edge_length)
    maxhgtd = max_over(tmp_maxhgtd(C2E), axis=dims.C2EDim)
    return maxslp, maxhgtd


@gtx.program
def compute_maxslp_maxhgtd(  # noqa: PLR0917 [too-many-positional-arguments]
    ddxn_z_full: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], wpfloat],
    dual_edge_length: gtx.Field[gtx.Dims[dims.EdgeDim], wpfloat],
    maxslp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    maxhgtd: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute z_maxslp and z_maxhgtd.

    See mo_vertical_grid.f90.

    Args:
        ddxn_z_full: dual_edge_length
        dual_edge_length: dual_edge_length
        maxslp: output
        maxhgtd: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """
    _compute_maxslp_maxhgtd(
        ddxn_z_full=ddxn_z_full,
        dual_edge_length=dual_edge_length,
        out=(maxslp, maxhgtd),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_exner_exfac(
    maxslp: fa.CellKField[wpfloat],
    maxhgtd: fa.CellKField[wpfloat],
    exner_expol: wpfloat,
    lateral_boundary_level_2: gtx.int32,
) -> fa.CellKField[wpfloat]:
    exner_exfac = concat_where(
        dims.CellDim >= lateral_boundary_level_2,
        exner_expol * minimum(1.0 - (4.0 * maxslp) ** 2, 1.0 - (0.002 * maxhgtd) ** 2),
        exner_expol,
    )
    exner_exfac = maximum(0.0, exner_exfac)
    exner_exfac = where(maxslp > 1.5, maximum(-1.0 / 6.0, 1.0 / 9.0 * (1.5 - maxslp)), exner_exfac)

    return exner_exfac


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_exner_exfac(  # noqa: PLR0917 [too-many-positional-arguments]
    maxslp: fa.CellKField[wpfloat],
    maxhgtd: fa.CellKField[wpfloat],
    exner_exfac: fa.CellKField[wpfloat],
    exner_expol: wpfloat,
    lateral_boundary_level_2: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute exner_exfac.

    Exner extrapolation reaches zero for a slope of 1/4 or a height difference of 500 m between adjacent grid points (empirically determined values). See mo_vertical_grid.f90

    Args:
        maxslp: maxslp
        maxhgtd: maxhgtd
        exner_exfac: Exner factor
        exner_expol: Exner extrapolation factor
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index

    """
    _compute_exner_exfac(
        maxhgtd=maxhgtd,
        maxslp=maxslp,
        exner_expol=exner_expol,
        lateral_boundary_level_2=lateral_boundary_level_2,
        out=exner_exfac,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program
def compute_wgtfac_e(  # noqa: PLR0917 [too-many-positional-arguments]
    wgtfac_c: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], float],
    wgtfac_e: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute wgtfac_e.

    See mo_vertical_grid.f90

    Args:
        wgtfac_c: weighting factor for quadratic interpolation to surface
        c_lin_e: interpolation field
        wgtfac_e: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """

    _cell_2_edge_interpolation(
        in_field=wgtfac_c,
        coeff=c_lin_e,
        out=wgtfac_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


def compute_flat_max_idx(
    *,
    e2c: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    c_lin_e: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    k_lev: data_alloc.NDArray,
    exchange: decomposition.ExchangeRuntime,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(e2c)
    k_lev_minus1 = k_lev[:-1]
    coeff_ = array_ns.expand_dims(c_lin_e, axis=-1)
    z_me = array_ns.sum(z_mc[e2c] * coeff_, axis=1)
    exchange.exchange(dims.EdgeDim, z_me, stream=decomposition.BLOCK)
    z_ifc_e_0 = z_ifc[e2c[:, 0], :-1]
    z_ifc_e_k_0 = z_ifc[e2c[:, 0], 1:]
    z_ifc_e_1 = z_ifc[e2c[:, 1], :-1]
    z_ifc_e_k_1 = z_ifc[e2c[:, 1], 1:]
    k_lev_minus1_expand = array_ns.expand_dims(k_lev_minus1, axis=0).repeat(z_me.shape[0], axis=0)
    flat_edge_index = array_ns.where(
        (z_me <= z_ifc_e_0) & (z_me >= z_ifc_e_k_0) & (z_me <= z_ifc_e_1) & (z_me >= z_ifc_e_k_1),
        k_lev_minus1_expand,
        0,
    )
    flat_idx_max = array_ns.amax(flat_edge_index, axis=1)
    return flat_idx_max


def compute_nflat_gradp(
    flat_idx_max: data_alloc.NDArray,
    e_owner_mask: data_alloc.NDArray,
    lateral_boundary_level: int,
    nlev: int,
    min_reduction: Callable[
        [data_alloc.NDArray], data_alloc.ScalarT
    ] = decomposition.single_node_reductions.min,
) -> int:
    """
    compute the nflat_gradp value as the minimum value of the flat_idx_max array.
    """
    array_ns = data_alloc.array_namespace(flat_idx_max)
    boundary_mask = array_ns.arange(flat_idx_max.shape[0]) >= lateral_boundary_level
    mask_array = array_ns.where(
        e_owner_mask & boundary_mask,
        flat_idx_max,
        nlev,
    )
    nflat_gradp = min_reduction(mask_array)
    return nflat_gradp


@gtx.field_operator
def _compute_downward_extrapolation_distance(
    z_ifc: fa.CellField[wpfloat],
) -> fa.EdgeField[wpfloat]:
    extrapol_dist = 5.0
    x = max_over(z_ifc(E2C), axis=dims.E2CDim)
    return x - extrapol_dist


@gtx.field_operator
def _compute_pressure_gradient_downward_extrapolation_mask_distance(  # noqa: PLR0917 [too-many-positional-arguments]
    z_mc: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    topography: fa.CellField[wpfloat],
    e_owner_mask: fa.EdgeField[bool],
    flat_idx_max: fa.EdgeField[gtx.int32],
    e_lev: fa.EdgeField[gtx.int32],
    k_lev: fa.KField[gtx.int32],
    horizontal_start_distance: int32,
    horizontal_end_distance: int32,
) -> fa.EdgeKField[wpfloat]:
    """
    Compute an edge mask and extrapolation distance for grid points requiring downward extrapolation of the pressure gradient.

    See pg_edgeidx and pg_exdist in mo_vertical_grid.f90

    Args:
        z_mc: height of cells [m]
        c_lin_e:  interpolation coefficient from cells to edges
        topography: ground level height of cells [m]
        e_owner_mask: mask edges owned by PE.
        flat_idx_max: level from where edge levels start to become flat
        e_lev: edge indices
        k_lev: k-level indices
        horizontal_start_distance: start index in edge fields from where extrapolation distance is computed
        horizontal_end_distance: end index in edge fields until where extrapolation distance is computed

    Returns:
        pg_exdist_dsl: extrapolation distance

    """

    e_lev = broadcast(e_lev, (dims.EdgeDim, dims.KDim))
    k_lev = broadcast(k_lev, (dims.EdgeDim, dims.KDim))
    z_me = _cell_2_edge_interpolation(in_field=z_mc, coeff=c_lin_e)
    downward_distance = _compute_downward_extrapolation_distance(topography)
    extrapolation_distance = concat_where(
        (horizontal_start_distance <= dims.EdgeDim) & (dims.EdgeDim < horizontal_end_distance),
        downward_distance,
        0.0,
    )

    pg_exdist_dsl = where(
        (k_lev >= (flat_idx_max + 1)) & (z_me < extrapolation_distance) & e_owner_mask,
        z_me - extrapolation_distance,
        0.0,
    )

    return pg_exdist_dsl


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_pressure_gradient_downward_extrapolation_mask_distance(  # noqa: PLR0917 [too-many-positional-arguments]
    z_mc: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], float],
    topography: fa.CellField[wpfloat],
    e_owner_mask: fa.EdgeField[bool],
    flat_idx_max: fa.EdgeField[gtx.int32],
    e_lev: fa.EdgeField[gtx.int32],
    k_lev: fa.KField[gtx.int32],
    pg_exdist_dsl: fa.EdgeKField[wpfloat],
    horizontal_start_distance: int32,
    horizontal_end_distance: int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_pressure_gradient_downward_extrapolation_mask_distance(
        z_mc=z_mc,
        c_lin_e=c_lin_e,
        topography=topography,
        flat_idx_max=flat_idx_max,
        e_owner_mask=e_owner_mask,
        e_lev=e_lev,
        k_lev=k_lev,
        horizontal_start_distance=horizontal_start_distance,
        horizontal_end_distance=horizontal_end_distance,
        out=pg_exdist_dsl,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_mask_prog_halo_c(
    c_refin_ctrl: fa.CellField[gtx.int32],
) -> fa.CellField[bool]:
    mask_prog_halo_c = where((c_refin_ctrl >= 1) & (c_refin_ctrl <= 4), False, True)
    return mask_prog_halo_c


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_mask_prog_halo_c(
    c_refin_ctrl: fa.CellField[gtx.int32],
    mask_prog_halo_c: fa.CellField[bool],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    """
    Compute mask_prog_halo_c.

    See mo_vertical_grid.f90

    Args:
        c_refin_ctrl: Cell field of refin_ctrl
        mask_prog_halo_c: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    _compute_mask_prog_halo_c(
        c_refin_ctrl,
        out=mask_prog_halo_c,
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def _compute_horizontal_mask_for_3d_divdamp(
    e_refin_ctrl: fa.EdgeField[gtx.int32],
    grf_nudge_start_e: gtx.int32,
    grf_nudgezone_width: gtx.int32,
) -> fa.EdgeField[wpfloat]:
    e_refin_ctrl_wp = astype(e_refin_ctrl, wpfloat)
    grf_nudge_start_e_wp = astype(grf_nudge_start_e, wpfloat)
    grf_nudgezone_width_wp = astype(grf_nudgezone_width, wpfloat)
    horizontal_mask_for_3d_divdamp = where(
        (e_refin_ctrl > (grf_nudge_start_e + grf_nudgezone_width - 1)),
        1.0
        / (grf_nudgezone_width_wp - 1.0)
        * (e_refin_ctrl_wp - (grf_nudge_start_e_wp + grf_nudgezone_width_wp - 1.0)),
        0.0,
    )
    horizontal_mask_for_3d_divdamp = where(
        (e_refin_ctrl <= 0)
        | (e_refin_ctrl_wp >= (grf_nudge_start_e_wp + 2.0 * (grf_nudgezone_width_wp - 1.0))),
        1.0,
        horizontal_mask_for_3d_divdamp,
    )
    return horizontal_mask_for_3d_divdamp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_mask_for_3d_divdamp(  # noqa: PLR0917 [too-many-positional-arguments]
    e_refin_ctrl: fa.EdgeField[gtx.int32],
    horizontal_mask_for_3d_divdamp: fa.EdgeField[wpfloat],
    grf_nudge_start_e: gtx.int32,
    grf_nudgezone_width: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    """
    Compute horizontal_mask_for_3d_divdamp (declared as hmask_dd3d in ICON).

    See mo_vertical_grid.f90. Horizontal mask field for 3D divergence damping term.

    Args:
        e_refin_ctrl: Edge field of refin_ctrl
        horizontal_mask_for_3d_divdamp: output
        grf_nudge_start_e: mo_impl_constants_grf constant
        grf_nudgezone_width: mo_impl_constants_grf constant
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    _compute_horizontal_mask_for_3d_divdamp(
        e_refin_ctrl=e_refin_ctrl,
        grf_nudge_start_e=grf_nudge_start_e,
        grf_nudgezone_width=grf_nudgezone_width,
        out=horizontal_mask_for_3d_divdamp,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def _compute_weighted_cell_neighbor_sum(
    field: fa.CellKField[wpfloat],
    c_bln_avg: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
) -> fa.CellKField[wpfloat]:
    field_avg = neighbor_sum(field(C2E2CO) * c_bln_avg, axis=C2E2CODim)
    return field_avg


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_weighted_cell_neighbor_sum(  # noqa: PLR0917 [too-many-positional-arguments]
    maxslp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    maxhgtd: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    c_bln_avg: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], wpfloat],
    maxslp_avg: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    maxhgtd_avg: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
    Compute maxslp_avg and maxhgtd_avg.

    See mo_vertical_grid.f90.

    Args:
        maxslp: Max field over ddxn_z_full offset
        maxhgtd: Max field over ddxn_z_full offset*dual_edge_length offset
        c_bln_avg: Interpolation field
        maxslp_avg: output
        maxhgtd_avg: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
        vertical_start: vertical start index
        vertical_end: vertical end index
    """

    _compute_weighted_cell_neighbor_sum(
        field=maxslp,
        c_bln_avg=c_bln_avg,
        out=maxslp_avg,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )

    _compute_weighted_cell_neighbor_sum(
        field=maxhgtd,
        c_bln_avg=c_bln_avg,
        out=maxhgtd_avg,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_max_nbhgt(
    z_mc_nlev: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    max_nbhgt = max_over(z_mc_nlev(C2E2C), axis=dims.C2E2CDim)
    return max_nbhgt


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_max_nbhgt(
    z_mc_nlev: fa.CellField[wpfloat],
    max_nbhgt: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    """
    Compute max_nbhgt.

    See mo_vertical_grid.f90.

    Args:
        z_mc_nlev: Last K level of z_mc
        max_nbhgt: output
        horizontal_start: horizontal start index
        horizontal_end: horizontal end index
    """
    _compute_max_nbhgt(
        z_mc_nlev=z_mc_nlev,
        out=max_nbhgt,
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@gtx.scan_operator(axis=dims.KDim, forward=True, init=(0, False))
def _compute_param(  # noqa: PLR0917 [too-many-positional-arguments]
    param: tuple[gtx.int32, bool],
    z_me_jk: float,
    z_ifc_off: float,
    z_ifc_off_koff: float,
    lower: gtx.int32,
    nlev: gtx.int32,
) -> tuple[gtx.int32, bool]:
    param_0, param_1 = param
    if param_0 >= lower:
        if (param_0 == nlev) | (z_me_jk <= z_ifc_off) & (z_me_jk >= z_ifc_off_koff):
            param_1 = True
    return param_0 + 1, param_1


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _compute_z_ifc_off_koff(
    z_ifc_off: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    n = z_ifc_off(KDim + 1)
    return n


def compute_exner_w_implicit_weight_parameter(
    *,
    c2e: data_alloc.NDArray,
    vct_a: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    z_ddxn_z_half_e: data_alloc.NDArray,
    z_ddxt_z_half_e: data_alloc.NDArray,
    dual_edge_length: data_alloc.NDArray,
    vwind_offctr: float,
    nlev: int,
    horizontal_start_cell: int,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(c2e)
    factor = max(vwind_offctr, 0.75)

    zn_off = array_ns.abs(z_ddxn_z_half_e[:, nlev][c2e])
    zt_off = array_ns.abs(z_ddxt_z_half_e[:, nlev][c2e])
    stacked = array_ns.concatenate((zn_off, zt_off), axis=1)
    maxslope = 0.425 * array_ns.amax(stacked, axis=1) ** (0.75)
    diff = array_ns.minimum(
        0.25,
        0.00025 * (array_ns.amax(array_ns.abs(zn_off * dual_edge_length[c2e]), axis=1) - 250.0),
    )
    offctr = array_ns.minimum(
        factor, array_ns.maximum(vwind_offctr, array_ns.maximum(maxslope, diff))
    )
    exner_w_implicit_weight_parameter = 0.5 + offctr

    k_start = max(0, nlev - 9)

    zdiff2 = (z_ifc[:, 0:nlev] - z_ifc[:, 1 : nlev + 1]) / (vct_a[0:nlev] - vct_a[1 : nlev + 1])

    for jk in range(k_start, nlev):
        zdiff2_sliced = zdiff2[horizontal_start_cell:, jk]
        index_for_k = array_ns.nonzero(zdiff2_sliced < 0.6)[0]
        max_value_k = array_ns.maximum(
            1.2 - zdiff2_sliced, exner_w_implicit_weight_parameter[horizontal_start_cell:]
        )
        exner_w_implicit_weight_parameter[index_for_k + horizontal_start_cell] = max_value_k[
            index_for_k
        ]

    return exner_w_implicit_weight_parameter
