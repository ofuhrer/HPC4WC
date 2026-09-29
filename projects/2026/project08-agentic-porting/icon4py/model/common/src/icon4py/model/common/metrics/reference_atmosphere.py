# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, exp, log

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    _cell_2_edge_interpolation,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_reference_atmosphere_edge_fields(  # noqa: PLR0917 [too-many-positional-arguments]
    z_mc: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    z_me = _cell_2_edge_interpolation(in_field=z_mc, coeff=c_lin_e)
    denom = t0sl_bg - del_t_bg
    exp_z_me = exp(z_me / h_scal_bg)
    logval = log((exp_z_me * denom + del_t_bg) / t0sl_bg)
    z_aux_1 = p0sl_bg * exp(-grav / rd * h_scal_bg / denom * logval)
    z_temp = denom + del_t_bg * exp(-z_me / h_scal_bg)
    rho_ref_me = z_aux_1 / (rd * z_temp)
    rd_o_cpd = rd / cpd
    theta_ref_me = z_temp / (z_aux_1 / p0ref) ** rd_o_cpd
    return (rho_ref_me, theta_ref_me)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_reference_atmosphere_edge_fields(  # noqa: PLR0917 [too-many-positional-arguments]
    z_mc: fa.CellKField[wpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], float],
    rho_ref_me: fa.EdgeKField[wpfloat],
    theta_ref_me: fa.EdgeKField[wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_reference_atmosphere_edge_fields(
        z_mc=z_mc,
        c_lin_e=c_lin_e,
        p0ref=p0ref,
        p0sl_bg=p0sl_bg,
        grav=grav,
        cpd=cpd,
        rd=rd,
        h_scal_bg=h_scal_bg,
        t0sl_bg=t0sl_bg,
        del_t_bg=del_t_bg,
        out=(rho_ref_me, theta_ref_me),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def compute_z_temp(
    z_mc: fa.CellKField[wpfloat],
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
) -> fa.CellKField[wpfloat]:
    denom = t0sl_bg - del_t_bg
    z_temp = denom + del_t_bg * exp(-z_mc / h_scal_bg)
    return z_temp


@gtx.field_operator
def compute_z_aux1_cell(  # noqa: PLR0917 [too-many-positional-arguments]
    z_mc: fa.CellKField[wpfloat],
    p0sl_bg: wpfloat,
    grav: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
) -> fa.CellKField[wpfloat]:
    denom = t0sl_bg - del_t_bg
    logval = log((exp(z_mc / h_scal_bg) * denom + del_t_bg) / t0sl_bg)
    return p0sl_bg * exp(-grav / rd * h_scal_bg / denom * logval)


@gtx.field_operator
def _compute_reference_atmosphere_cell_fields(  # noqa: PLR0917 [too-many-positional-arguments]
    z_mc: fa.CellKField[wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    z_aux1 = compute_z_aux1_cell(
        z_mc=z_mc,
        p0sl_bg=p0sl_bg,
        grav=grav,
        rd=rd,
        h_scal_bg=h_scal_bg,
        t0sl_bg=t0sl_bg,
        del_t_bg=del_t_bg,
    )

    rd_o_cpd = rd / cpd
    exner_ref_mc = (z_aux1 / p0ref) ** rd_o_cpd
    z_temp = compute_z_temp(z_mc=z_mc, del_t_bg=del_t_bg, t0sl_bg=t0sl_bg, h_scal_bg=h_scal_bg)
    rho_ref_mc = z_aux1 / (rd * z_temp)
    theta_ref_mc = z_temp / exner_ref_mc
    return (
        theta_ref_mc,
        exner_ref_mc,
        rho_ref_mc,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_reference_atmosphere_cell_fields(  # noqa: PLR0917 [too-many-positional-arguments]
    z_height: fa.CellKField[wpfloat],
    exner_ref_mc: fa.CellKField[wpfloat],
    rho_ref_mc: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[wpfloat],
    p0ref: wpfloat,
    p0sl_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    h_scal_bg: wpfloat,
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    """
        Calculate reference atmosphere fields on full levels.

    Args:
        z_height: geometric height
        exner_ref_mc: (output) reference exner pressure on full level mass points
        rho_ref_mc: (output) reference density on full level mass points
        theta_ref_mc: (output) reference potential temperature on full level mass points
        p0ref: reference pressure for exner function [Pa]
        p0sl_bg: sea level pressuer [Pa]
        grav: gravitational constant [m/s^2]
        cpd: specific heat at constant pressure [J/K/kg]
        rd: gas constant for dry air [J/K/kg]
        h_scal_bg: height scale [m]
        t0sl_bg: sea level temperature [K]
        del_t_bg: temperature difference between sea level and asymptotic stratospheric temperature
        horizontal_start:int32 start index of horizontal domain
        horizontal_end:int32 end index of horizontal domain
        vertical_start:int32 start index of vertical domain
        vertical_end:int32 end index of vertical domain
    """
    _compute_reference_atmosphere_cell_fields(
        z_mc=z_height,
        p0ref=p0ref,
        p0sl_bg=p0sl_bg,
        grav=grav,
        cpd=cpd,
        rd=rd,
        h_scal_bg=h_scal_bg,
        t0sl_bg=t0sl_bg,
        del_t_bg=del_t_bg,
        out=(theta_ref_mc, exner_ref_mc, rho_ref_mc),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _compute_theta_d_exner_dz_ref_ic(  # noqa: PLR0917 [too-many-positional-arguments]
    z_ifc: fa.CellKField[wpfloat],
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
    grav: wpfloat,
    cpd: wpfloat,
    rd: wpfloat,
    p0sl_bg: wpfloat,
    rd_o_cpd: wpfloat,
    p0ref: wpfloat,
):
    """
    Calculate the reference Exner pressure and its first vertical derivative, half level mass points.
    """
    z_aux1 = p0sl_bg * exp(
        -grav
        / rd
        * h_scal_bg
        / (t0sl_bg - del_t_bg)
        * log((exp(z_ifc / h_scal_bg) * (t0sl_bg - del_t_bg) + del_t_bg) / t0sl_bg)
    )
    z_help = (z_aux1 / p0ref) ** rd_o_cpd
    z_temp = (t0sl_bg - del_t_bg) + del_t_bg * exp(-z_ifc / h_scal_bg)
    theta_ref_ic = z_temp / z_help
    d_exner_dz_ref_ic = -grav / cpd / theta_ref_ic
    return theta_ref_ic, d_exner_dz_ref_ic


@gtx.field_operator
def _compute_d2dexdz2_fac_mc(  # noqa: PLR0917 [too-many-positional-arguments]
    theta_ref_mc: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    exner_ref_mc: fa.CellKField[vpfloat],
    z_mc: fa.CellKField[wpfloat],
    cpd: wpfloat,
    grav: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    """
    Compute vertical derivative of d_exner_dz/theta_ref for full level mass points.

    (d2dexdz2_fac1_mc and d2dexdz2_fac2_mc factors in mo_vertical_grid.f90)

    Args:
        theta_ref_mc: reference Potential temperature, full level mass points
        inv_ddqz_z_full: inverse layer thickness (for runtime optimization)
        exner_ref_mc: reference Exner pressure, full level mass points
        z_mc: geometric height defined on full levels
        cpd: Specific heat at constant pressure [J/K/kg]
        grav: average gravitational acceleration
        del_t_bg: difference between sea level temperature and asymptotic stratospheric temperature
        h_scal_bg: height scale for reference atmosphere [m]

    Returns:
        fac1: first vertical derivative of reference Exner pressure, full level mass points, divided by theta_ref
        fac2: vertical derivative of d_exner_dz/theta_ref, full level mass points


    """
    del_t_bg = astype(del_t_bg, vpfloat)
    cpd = astype(cpd, vpfloat)
    grav = astype(grav, vpfloat)
    h_scal_bg = astype(h_scal_bg, vpfloat)
    z_mc = astype(z_mc, vpfloat)
    fac1 = -grav / (cpd * theta_ref_mc**2) * inv_ddqz_z_full
    fac2 = (
        2.0
        * grav
        / (cpd * theta_ref_mc**3)
        * (grav / cpd - del_t_bg / h_scal_bg * exp(-z_mc / h_scal_bg))
        / exner_ref_mc
    )

    return fac1, fac2


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_theta_d_exner_dz_ref_ic(  # noqa: PLR0917 [too-many-positional-arguments]
    z_ifc: fa.CellKField[wpfloat],
    d_exner_dz_ref_ic: fa.CellKField[wpfloat],
    theta_ref_ic: fa.CellKField[wpfloat],
    t0sl_bg: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
    grav: wpfloat,
    rd: wpfloat,
    cpd: wpfloat,
    p0sl_bg: wpfloat,
    rd_o_cpd: wpfloat,
    p0ref: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_theta_d_exner_dz_ref_ic(
        z_ifc=z_ifc,
        t0sl_bg=t0sl_bg,
        del_t_bg=del_t_bg,
        h_scal_bg=h_scal_bg,
        grav=grav,
        cpd=cpd,
        rd=rd,
        p0sl_bg=p0sl_bg,
        rd_o_cpd=rd_o_cpd,
        p0ref=p0ref,
        out=(theta_ref_ic, d_exner_dz_ref_ic),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_d2dexdz2_fac_mc(  # noqa: PLR0917 [too-many-positional-arguments]
    theta_ref_mc: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    exner_ref_mc: fa.CellKField[vpfloat],
    z_mc: fa.CellKField[wpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    cpd: float,
    grav: wpfloat,
    del_t_bg: wpfloat,
    h_scal_bg: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_d2dexdz2_fac_mc(
        theta_ref_mc=theta_ref_mc,
        inv_ddqz_z_full=inv_ddqz_z_full,
        exner_ref_mc=exner_ref_mc,
        z_mc=z_mc,
        cpd=cpd,
        grav=grav,
        del_t_bg=del_t_bg,
        h_scal_bg=h_scal_bg,
        out=(d2dexdz2_fac1_mc, d2dexdz2_fac2_mc),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
