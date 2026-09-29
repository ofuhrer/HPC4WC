# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    thermodynamic_functions as thermo,
)
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.math.stencils import generic_math_operations_array_ns
from icon4py.model.common.utils import data_allocation as data_alloc


def apply_hydrostatic_adjustment_ndarray(
    *,
    rho: data_alloc.NDArray,
    exner: data_alloc.NDArray,
    theta_v: data_alloc.NDArray,
    exner_ref_mc: data_alloc.NDArray,
    d_exner_dz_ref_ic: data_alloc.NDArray,
    theta_ref_mc: data_alloc.NDArray,
    theta_ref_ic: data_alloc.NDArray,
    wgtfac_c: data_alloc.NDArray,
    ddqz_z_half: data_alloc.NDArray,
    num_levels: int,
) -> None:
    """
    apply hydrostatic adjustment to update rho, exner, and theta_v arrays
    """
    array_ns = data_alloc.array_namespace(rho)
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
    u0: float,
    baroclinic_amplitude: float,
    lat_perturbation_center: float,
    lon_perturbation_center: float,
    edge_lat: data_alloc.NDArray,
    edge_lon: data_alloc.NDArray,
    primal_normal_x: data_alloc.NDArray,
    eta_v_at_edge: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """
    Compute normal wind at edge center from vertical eta coordinate (eta_v_at_edge).

    Args:
        grid: IconGrid
        u0: base zonal wind speed factor, or maximum wind speed
        baroclinic_amplitude: perturbation amplitude
        lat_perturbation_center: perturbation center in latitude
        lon_perturbation_center: perturbation center in longitude
        edge_lat: edge center latitude
        edge_lon: edge center longitude
        primal_normal_x: zonal component of primal normal vector at edge center
        eta_v_at_edge: vertical eta coordinate at edge center
    Returns: normal wind
    """
    # TODO(OngChia): this function needs a test
    array_ns = data_alloc.array_namespace(edge_lat)
    ub = grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    mask = array_ns.ones((grid.num_edges, grid.num_levels), dtype=bool)
    mask[
        0:ub,
        :,
    ] = False
    edge_lat = array_ns.repeat(
        array_ns.expand_dims(edge_lat, axis=-1), eta_v_at_edge.shape[1], axis=1
    )
    edge_lon = array_ns.repeat(
        array_ns.expand_dims(edge_lon, axis=-1), eta_v_at_edge.shape[1], axis=1
    )
    primal_normal_x = array_ns.repeat(
        array_ns.expand_dims(primal_normal_x, axis=-1), eta_v_at_edge.shape[1], axis=1
    )
    u = array_ns.where(
        mask,
        u0 * (array_ns.cos(eta_v_at_edge) ** 1.5) * (array_ns.sin(2.0 * edge_lat) ** 2),
        0.0,
    )
    if baroclinic_amplitude > 1.0e-20:
        u = array_ns.where(
            mask,
            u
            + baroclinic_amplitude
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


def init_w(
    *,
    grid: icon_grid.IconGrid,
    z_ifc: data_alloc.NDArray,
    inv_dual_edge_length: data_alloc.NDArray,
    edge_cell_distance: data_alloc.NDArray,
    primal_edge_length: data_alloc.NDArray,
    cell_area: data_alloc.NDArray,
    vn: data_alloc.NDArray,
    vct_b: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(z_ifc)
    # The bounds need to include the first halo line because of the e2c -> c2e connectivity
    lb_e = grid.start_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    ub_e = grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.END))
    lb_c = grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    ub_c = grid.end_index(h_grid.domain(dims.CellDim)(h_grid.Zone.END))

    c2e = grid.get_connectivity(dims.C2E).ndarray
    e2c = grid.get_connectivity(dims.E2C).ndarray

    z_grad_e = generic_math_operations_array_ns.compute_directional_derivative_on_edges(
        cell_field=z_ifc[:, nlev],
        e2c=e2c,
        inv_dual_edge_length=inv_dual_edge_length,
        lb_e=lb_e,
        ub_e=ub_e,
        num_edges=grid.num_edges,
    )
    z_wsfc_e = vn[:, nlev - 1] * z_grad_e

    z_wsfc_c = generic_math_operations_array_ns.interpolate_edges_to_cell(
        edge_field=z_wsfc_e,
        c2e=c2e,
        e2c=e2c,
        edge_cell_length=edge_cell_distance,
        primal_edge_length=primal_edge_length,
        cell_area=cell_area,
        ub_c=ub_c,
        num_cells=grid.num_cells,
    )

    w = array_ns.zeros((grid.num_cells, nlev + 1))
    w[lb_c:ub_c, nlev] = z_wsfc_c[lb_c:ub_c]
    w[lb_c:ub_c, 1:] = z_wsfc_c[lb_c:ub_c, array_ns.newaxis] * vct_b[array_ns.newaxis, 1:]

    return w


def init_inwp_tracers(
    *,
    rho: data_alloc.NDArray,
    virtual_temperature: data_alloc.NDArray,
    pressure: data_alloc.NDArray,
    cell_area: data_alloc.NDArray,
    ddqz_z_full: data_alloc.NDArray,
    qv: data_alloc.NDArray,
    global_reductions: decomposition_defs.Reductions,
    n_iter: int,
    rh_at_1000hpa: float,
    qv_max: float,
    global_moisture_content: float,
    normalize_global_moisture: bool,
) -> None:
    """Initialize the water-vapour tracer ``qv`` from a relative-humidity profile.

    Host port of ``init_nh_inwp_tracers`` (mo_nh_jabw_exp.f90): qv follows a
    linearly decreasing relative humidity with height, is iterated against the
    moisture-dependent temperature, and (for the APE cases) finally rescaled so the
    global mean column-integrated moisture matches ``global_moisture_content``. The
    other hydrometeors keep their zero-initialized value.

    ``pressure`` is the hydrostatic pressure and ``virtual_temperature`` is
    ``theta_v * exner``; both are independent of qv and passed in already diagnosed.
    ``rho`` is the dry-air density from the hydrostatic adjustment (no moisture
    feedback).
    """
    array_ns = data_alloc.array_namespace(rho)

    # linearly decreasing relative humidity with height (independent of qv)
    relative_humidity = array_ns.maximum(
        rh_at_1000hpa - 0.5 + pressure / phy_const.RELATIVE_HUMIDITY_REFERENCE_PRESSURE, 0.0
    )

    def _qv(temperature: data_alloc.NDArray) -> data_alloc.NDArray:
        q = thermo.qv_from_relative_humidity(temperature, pressure, rho, relative_humidity)
        # stratosphere and tropics caps (init_nh_inwp_tracers)
        q = array_ns.where(
            pressure <= phy_const.STRATOSPHERE_PRESSURE_THRESHOLD,
            array_ns.minimum(q, phy_const.STRATOSPHERIC_QV_CAP),
            q,
        )
        return array_ns.minimum(q, qv_max)

    # first guess uses qv = 0, i.e. temperature == virtual temperature
    temperature = virtual_temperature
    qv_values = _qv(temperature)
    for _ in range(n_iter - 1):
        # re-diagnose the actual temperature with the moisture feedback; the other
        # hydrometeors are zero here, so only qv enters (see diagnose_temperature).
        temperature = virtual_temperature / (1.0 + phy_const.RV_O_RD_MINUS_1 * qv_values)
        qv_values = _qv(temperature)

    if normalize_global_moisture:
        # rescale qv so the global mean column-integrated moisture matches the
        # prescribed value (Fortran opt_global_moist / ztmc_ape). The per-cell
        # qv_max and stratosphere caps are not re-applied after this uniform
        # rescale, matching init_nh_inwp_tracers in mo_nh_jabw_exp.f90.
        column_moisture = global_reductions.sum(
            ddqz_z_full * rho * qv_values * cell_area[:, array_ns.newaxis]
        )
        total_area = global_reductions.sum(cell_area)
        mean_column_moisture = column_moisture / total_area
        if mean_column_moisture > 1.0e-25:
            qv_values = qv_values * (global_moisture_content / mean_column_moisture)

    qv[:, :] = qv_values


# ---------------------------------------------------------------------------
# Shared helpers (used by individual analytical IC modules)
# ---------------------------------------------------------------------------


def zone_indices(grid: icon_grid.IconGrid) -> dict[str, int]:
    edge_domain = h_grid.domain(dims.EdgeDim)
    cell_domain = h_grid.domain(dims.CellDim)
    return {
        "end_edge_lateral_boundary_level_2": grid.end_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        ),
        "end_edge_end": grid.end_index(edge_domain(h_grid.Zone.END)),
        "end_cell_lateral_boundary_level_2": grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        ),
        "end_cell_end": grid.end_index(cell_domain(h_grid.Zone.END)),
    }
