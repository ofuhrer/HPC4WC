# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.utils import data_allocation as data_alloc


def hydrostatic_adjustment_ndarray(
    *,
    wgtfac_c: data_alloc.NDArray,
    ddqz_z_half: data_alloc.NDArray,
    exner_ref_mc: data_alloc.NDArray,
    d_exner_dz_ref_ic: data_alloc.NDArray,
    theta_ref_mc: data_alloc.NDArray,
    theta_ref_ic: data_alloc.NDArray,
    rho: data_alloc.NDArray,
    exner: data_alloc.NDArray,
    theta_v: data_alloc.NDArray,
    num_levels: int,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(wgtfac_c)
    # virtual temperature
    temp_v = theta_v * exner

    for k in range(num_levels - 2, -1, -1):
        fac1 = (
            wgtfac_c[:, k + 1] * (temp_v[:, k + 1] - theta_ref_mc[:, k + 1] * exner[:, k + 1])
            - (1.0 - wgtfac_c[:, k + 1]) * theta_ref_mc[:, k] * exner[:, k + 1]
        )
        fac2 = (1.0 - wgtfac_c[:, k + 1]) * temp_v[:, k] * exner[:, k + 1]
        fac3 = exner_ref_mc[:, k + 1] - exner_ref_mc[:, k] - exner[:, k + 1]

        quadratic_a = (theta_ref_ic[:, k + 1] * exner[:, k + 1] + fac1) / ddqz_z_half[:, k + 1]
        quadratic_b = -(
            quadratic_a * fac3 + fac2 / ddqz_z_half[:, k + 1] + fac1 * d_exner_dz_ref_ic[:, k + 1]
        )
        quadratic_c = -(fac2 * fac3 / ddqz_z_half[:, k + 1] + fac2 * d_exner_dz_ref_ic[:, k + 1])

        exner[:, k] = (
            quadratic_b + array_ns.sqrt(quadratic_b**2 + 4.0 * quadratic_a * quadratic_c)
        ) / (2.0 * quadratic_a)
        theta_v[:, k] = temp_v[:, k] / exner[:, k]
        rho[:, k] = (
            exner[:, k] ** phy_const.CVD_O_RD * phy_const.P0REF / (phy_const.RD * theta_v[:, k])
        )

    return rho, exner, theta_v


def hydrostatic_adjustment_constant_thetav_ndarray(
    *,
    wgtfac_c: data_alloc.NDArray,
    ddqz_z_half: data_alloc.NDArray,
    exner_ref_mc: data_alloc.NDArray,
    d_exner_dz_ref_ic: data_alloc.NDArray,
    theta_ref_mc: data_alloc.NDArray,
    theta_ref_ic: data_alloc.NDArray,
    rho: data_alloc.NDArray,
    exner: data_alloc.NDArray,
    theta_v: data_alloc.NDArray,
    num_levels: int,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """
    Computes a hydrostatically balanced profile. In constrast to the above
    hydrostatic_adjustment_ndarray, the virtual temperature is kept (assumed)
    constant during the adjustment, leading to a simpler formula.
    """

    for k in range(num_levels - 2, -1, -1):
        theta_v_pr_ic = wgtfac_c[:, k + 1] * (theta_v[:, k + 1] - theta_ref_mc[:, k + 1]) + (
            1.0 - wgtfac_c[:, k + 1]
        ) * (theta_v[:, k] - theta_ref_mc[:, k])

        exner[:, k] = (
            exner[:, k + 1]
            + (exner_ref_mc[:, k] - exner_ref_mc[:, k + 1])
            - ddqz_z_half[:, k + 1]
            / (theta_v_pr_ic + theta_ref_ic[:, k + 1])
            * theta_v_pr_ic
            * d_exner_dz_ref_ic[:, k + 1]
        )

    for k in range(num_levels - 1, -1, -1):
        rho[:, k] = (
            exner[:, k] ** phy_const.CVD_O_RD * phy_const.P0REF / (phy_const.RD * theta_v[:, k])
        )

    return rho, exner


def zonalwind_2_normalwind_ndarray(
    *,
    grid: icon_grid.IconGrid,
    jw_u0: float,
    jw_up: float,
    lat_perturbation_center: float,
    lon_perturbation_center: float,
    edge_lat: data_alloc.NDArray,
    edge_lon: data_alloc.NDArray,
    primal_normal_x: data_alloc.NDArray,
    eta_v_e: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """
    Compute normal wind at edge center from vertical eta coordinate (eta_v_e).

    Args:
        grid: IconGrid
        jw_u0: base zonal wind speed factor
        jw_up: perturbation amplitude
        lat_perturbation_center: perturbation center in latitude
        lon_perturbation_center: perturbation center in longitude
        edge_lat: edge center latitude
        edge_lon: edge center longitude
        primal_normal_x: zonal component of primal normal vector at edge center
        eta_v_e: vertical eta coordinate at edge center
    Returns: normal wind
    """
    # TODO(OngChia): this function needs a test
    array_ns = data_alloc.array_namespace(edge_lat)

    mask = array_ns.ones((grid.num_edges, grid.num_levels), dtype=bool)
    mask[
        0 : grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)),
        :,
    ] = False
    edge_lat = array_ns.repeat(array_ns.expand_dims(edge_lat, axis=-1), eta_v_e.shape[1], axis=1)
    edge_lon = array_ns.repeat(array_ns.expand_dims(edge_lon, axis=-1), eta_v_e.shape[1], axis=1)
    primal_normal_x = array_ns.repeat(
        array_ns.expand_dims(primal_normal_x, axis=-1), eta_v_e.shape[1], axis=1
    )
    u = array_ns.where(
        mask, jw_u0 * (array_ns.cos(eta_v_e) ** 1.5) * (array_ns.sin(2.0 * edge_lat) ** 2), 0.0
    )
    if jw_up > 1.0e-20:
        u = array_ns.where(
            mask,
            u
            + jw_up
            * array_ns.exp(
                -(
                    (
                        10.0
                        * array_ns.arccos(
                            array_ns.sin(lat_perturbation_center) * array_ns.sin(edge_lat)
                            + array_ns.cos(lat_perturbation_center)
                            * array_ns.cos(edge_lat)
                            * array_ns.cos(edge_lon - lon_perturbation_center)
                        )
                    )
                    ** 2
                )
            ),
            u,
        )
    vn = u * primal_normal_x

    return vn


# TODO(OngChia): Can this kind of simple arithmetic operation be replaced by a more general stencil that does the same operation on a general field?
@gtx.field_operator
def _compute_perturbed_exner(
    exner: fa.CellKField[ta.wpfloat],
    reference_exner: fa.CellKField[ta.vpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the perturbed exner function (exner_pr in ICON).
        perturbed_exner = exner - reference_exner
    This stencil is copied from subroutine compute_exner_pert in mo_nh_init_utils in ICON. It should be called
    during the initialization to initialize perturbed_exner_at_cells_on_model_levels of DiagnosticStateHydro
    if the model does not restart from a restart file
    Args:
        exner: exner function
        reference_exner: reference exner function
    Returns:
        Perturbed exner function
    """
    perturbed_exner = exner - reference_exner
    return perturbed_exner


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_perturbed_exner(  # noqa: PLR0917 [too-many-positional-arguments]
    exner: fa.CellKField[ta.wpfloat],
    reference_exner: fa.CellKField[ta.vpfloat],
    perturbed_exner: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_perturbed_exner(
        exner=exner,
        reference_exner=reference_exner,
        out=perturbed_exner,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


def create_gt4py_field_for_prognostic_and_diagnostic_variables(
    *,
    vn_ndarray: data_alloc.NDArray,
    w_ndarray: data_alloc.NDArray,
    exner_ndarray: data_alloc.NDArray,
    rho_ndarray: data_alloc.NDArray,
    theta_v_ndarray: data_alloc.NDArray,
    temperature_ndarray: data_alloc.NDArray,
    pressure_ndarray: data_alloc.NDArray,
    pressure_ifc_ndarray: data_alloc.NDArray,
    grid: icon_grid.IconGrid,
    allocator: gtx_typing.Allocator,
) -> tuple[
    fa.EdgeKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    vn = gtx.as_field((dims.EdgeDim, dims.KDim), vn_ndarray, allocator=allocator)
    w = gtx.as_field((dims.CellDim, dims.KDim), w_ndarray, allocator=allocator)
    exner = gtx.as_field((dims.CellDim, dims.KDim), exner_ndarray, allocator=allocator)
    rho = gtx.as_field((dims.CellDim, dims.KDim), rho_ndarray, allocator=allocator)
    temperature = gtx.as_field((dims.CellDim, dims.KDim), temperature_ndarray, allocator=allocator)
    virtual_temperature = gtx.as_field(
        (dims.CellDim, dims.KDim), temperature_ndarray, allocator=allocator
    )
    pressure = gtx.as_field((dims.CellDim, dims.KDim), pressure_ndarray, allocator=allocator)
    theta_v = gtx.as_field((dims.CellDim, dims.KDim), theta_v_ndarray, allocator=allocator)
    pressure_ifc = gtx.as_field(
        (dims.CellDim, dims.KDim), pressure_ifc_ndarray, allocator=allocator
    )

    vn_next = gtx.as_field((dims.EdgeDim, dims.KDim), vn_ndarray, allocator=allocator)
    w_next = gtx.as_field((dims.CellDim, dims.KDim), w_ndarray, allocator=allocator)
    exner_next = gtx.as_field((dims.CellDim, dims.KDim), exner_ndarray, allocator=allocator)
    rho_next = gtx.as_field((dims.CellDim, dims.KDim), rho_ndarray, allocator=allocator)
    theta_v_next = gtx.as_field((dims.CellDim, dims.KDim), theta_v_ndarray, allocator=allocator)

    u = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator)
    v = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator)

    return (
        vn,
        w,
        exner,
        rho,
        theta_v,
        vn_next,
        w_next,
        exner_next,
        rho_next,
        theta_v_next,
        temperature,
        virtual_temperature,
        pressure,
        pressure_ifc,
        u,
        v,
    )
