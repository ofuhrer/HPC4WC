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
import math
from typing import TYPE_CHECKING, ClassVar

import gt4py.next as gtx

from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.diagnostic_calculations import pressure as pressure_diagnostics
from icon4py.model.common.grid import (
    geometry_attributes as geometry_meta,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.initial_condition.analytical import utils as testcases_utils
from icon4py.model.common.interpolation import interpolation_attributes
from icon4py.model.common.interpolation.stencils import cell_2_edge_interpolation
from icon4py.model.common.metrics import metrics_attributes
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.states import static_fields

log = logging.getLogger(__name__)


@dataclasses.dataclass
class JablonowskiWilliamsonConfig:
    # jabw* and APE testcases share this initial condition and config
    # with the difference that jabw* has p_sfc hardcoded to 1e5, while APE
    # reads zp_ape from the nh_testcase_nml
    # The default values are from mo_nh_jabw_exp.f90 and mo_nh_testcases_nml.f90
    p_sfc: float = 100000.0
    # amplitude of the u-perturbation [m/s] (jw_up); jabw_s resets it to 0.0.
    baroclinic_amplitude: float = 1.0
    u0: float = 35.0
    temp0: float = 288.0
    eta_0: float = 0.252
    eta_t: float = 0.2
    gamma: float = 0.005
    dtemp: float = 4.8e5
    lon_perturbation_center: float = math.pi / 9.0
    lat_perturbation_center: float = 2.0 * math.pi / 9.0
    # Moist tracer initialization (inwp tracers, see init_nh_inwp_tracers in
    # mo_nh_jabw_exp.f90). Relevant only when transport is active.
    rh_at_1000hpa: float = 0.7
    qv_max: float = 20e-3
    # number of iterations to converge qv against the moisture-dependent
    # temperature (Fortran l_rediag=.TRUE. => 10 iterations).
    moisture_init_iterations: int = 10
    # target column-integrated moisture for APE cases [kg/m**2] (ztmc_ape).
    global_moisture_content: float = 25.006
    # rescale qv to global_moisture_content (APE only; Fortran opt_global_moist).
    normalize_global_moisture: bool = False

    fortran_name_map: ClassVar[dict[str, str]] = {
        "jw_up": "baroclinic_amplitude",
        "jw_u0": "u0",
        "jw_temp0": "temp0",
        "zp_ape": "p_sfc",
        "rh_at_1000hpa": "rh_at_1000hpa",
        "qv_max": "qv_max",
        "ztmc_ape": "global_moisture_content",
    }


def jablonowski_williamson(  # noqa: PLR0915 [too-many-statements]
    *,
    config: JablonowskiWilliamsonConfig,
    vertical_config: v_grid.VerticalGridConfig,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
    global_reductions: decomposition_defs.Reductions,
) -> None:
    """
    Initial condition for Jablonowski-Williamson test.
    The u-perturbation amplitude (``baroclinic_amplitude``, jw_up) defaults to
    1.0 m/s as in ICON; the jabw_s/jabw_m cases reset it to 0.0 (steady state).

    The reference experiment config for this is
    exp.exclaim_nh35_tri_jws_sb.
    """
    allocator = model_backends.get_allocator(backend)
    array_ns = data_alloc.import_array_ns(allocator)

    geometry = static_fields.geometry
    metrics = static_fields.metrics
    interpolation = static_fields.interpolation

    cell_lat = geometry.get(geometry_meta.CELL_LAT).ndarray
    edge_lat = geometry.get(geometry_meta.EDGE_LAT).ndarray
    edge_lon = geometry.get(geometry_meta.EDGE_LON).ndarray
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
    ddqz_z_full_field = metrics.get(metrics_attributes.DDQZ_Z_FULL)
    c_lin_e = interpolation.get(interpolation_attributes.C_LIN_E)
    zone_idx = testcases_utils.zone_indices(grid)

    p_sfc = config.p_sfc
    jw_baroclinic_amplitude = config.baroclinic_amplitude
    u0 = config.u0
    temp0 = config.temp0
    eta_0 = config.eta_0
    eta_t = config.eta_t
    gamma = config.gamma
    dtemp = config.dtemp
    lon_perturbation_center = config.lon_perturbation_center
    lat_perturbation_center = config.lat_perturbation_center

    num_cells = grid.num_cells
    num_levels = grid.num_levels

    eta_v = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    eta_v_at_edge = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator)

    exner_ndarray = prognostic_state_now.exner.ndarray
    rho_ndarray = prognostic_state_now.rho.ndarray
    theta_v_ndarray = prognostic_state_now.theta_v.ndarray
    eta_v_ndarray = eta_v.ndarray

    sin_lat = array_ns.sin(cell_lat)
    cos_lat = array_ns.cos(cell_lat)
    fac1 = ta.wpfloat("1.0") / ta.wpfloat("6.3") - ta.wpfloat("2.0") * (sin_lat**6) * (
        cos_lat**2 + ta.wpfloat("1.0") / ta.wpfloat("3.0")
    )
    fac2 = (
        (
            ta.wpfloat("8.0")
            / ta.wpfloat("5.0")
            * (cos_lat**3)
            * (sin_lat**2 + ta.wpfloat("2.0") / ta.wpfloat("3.0"))
            - ta.wpfloat("0.25") * math.pi
        )
        * phy_const.EARTH_RADIUS
        * phy_const.EARTH_ANGULAR_VELOCITY
    )
    lapse_rate = phy_const.RD * gamma / phy_const.GRAV
    for k_index in range(num_levels - 1, -1, -1):
        eta_old = array_ns.full(num_cells, fill_value=ta.wpfloat("1.0e-7"), dtype=ta.wpfloat)
        log.info(f"In Newton iteration, k = {k_index}")
        for _ in range(100):
            eta_v_ndarray[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
            cos_etav = array_ns.cos(eta_v_ndarray[:, k_index])
            sin_etav = array_ns.sin(eta_v_ndarray[:, k_index])

            temperature_avg = temp0 * (eta_old**lapse_rate)
            geopot_avg = temp0 * phy_const.GRAV / gamma * (ta.wpfloat("1.0") - eta_old**lapse_rate)
            temperature_avg = array_ns.where(
                eta_old < eta_t, temperature_avg + dtemp * ((eta_t - eta_old) ** 5), temperature_avg
            )
            geopot_avg = array_ns.where(
                eta_old < eta_t,
                geopot_avg
                - phy_const.RD
                * dtemp
                * (
                    (array_ns.log(eta_old / eta_t) + ta.wpfloat("137.0") / ta.wpfloat("60.0"))
                    * (eta_t**5)
                    - ta.wpfloat("5.0") * (eta_t**4) * eta_old
                    + ta.wpfloat("5.0") * (eta_t**3) * (eta_old**2)
                    - ta.wpfloat("10.0") / ta.wpfloat("3.0") * (eta_t**2) * (eta_old**3)
                    + ta.wpfloat("1.25") * eta_t * (eta_old**4)
                    - ta.wpfloat("0.2") * (eta_old**5)
                ),
                geopot_avg,
            )

            geopot_jw = geopot_avg + u0 * (cos_etav**1.5) * (fac1 * u0 * (cos_etav**1.5) + fac2)
            temperature_jw = temperature_avg + ta.wpfloat(
                "0.75"
            ) * eta_old * math.pi * u0 / phy_const.RD * sin_etav * array_ns.sqrt(cos_etav) * (
                ta.wpfloat("2.0") * u0 * fac1 * (cos_etav**1.5) + fac2
            )
            newton_function = geopot_jw - geopot[:, k_index]
            newton_function_prime = -phy_const.RD / eta_old * temperature_jw
            eta_old = eta_old - newton_function / newton_function_prime

        eta_v_ndarray[:, k_index] = (eta_old - eta_0) * math.pi * 0.5
        exner_ndarray[:, k_index] = (eta_old * p_sfc / phy_const.P0REF) ** phy_const.RD_O_CPD
        theta_v_ndarray[:, k_index] = temperature_jw / exner_ndarray[:, k_index]
        rho_ndarray[:, k_index] = (
            exner_ndarray[:, k_index] ** phy_const.CVD_O_RD
            * phy_const.P0REF
            / phy_const.RD
            / theta_v_ndarray[:, k_index]
        )
    log.info("Newton iteration completed.")

    cell_2_edge_interpolation.cell_2_edge_interpolation.with_backend(backend)(
        in_field=eta_v,
        coeff=c_lin_e,
        out_field=eta_v_at_edge,
        horizontal_start=zone_idx["end_edge_lateral_boundary_level_2"],
        horizontal_end=zone_idx["end_edge_end"],
        vertical_start=0,
        vertical_end=num_levels,
        offset_provider=grid.connectivities,
    )
    exchange.exchange(dims.EdgeDim, eta_v_at_edge)
    log.info("Cell-to-edge eta_v computation completed.")

    prognostic_state_now.vn.ndarray[:, :] = testcases_utils.zonalwind_2_normalwind_ndarray(
        grid=grid,
        u0=u0,
        baroclinic_amplitude=jw_baroclinic_amplitude,
        lat_perturbation_center=lat_perturbation_center,
        lon_perturbation_center=lon_perturbation_center,
        edge_lat=edge_lat,
        edge_lon=edge_lon,
        primal_normal_x=primal_normal_x,
        eta_v_at_edge=eta_v_at_edge.ndarray,
    )
    log.info("U2vn computation completed.")

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

    testcases_utils.apply_hydrostatic_adjustment_ndarray(
        rho=rho_ndarray,
        exner=exner_ndarray,
        theta_v=theta_v_ndarray,
        exner_ref_mc=exner_ref_mc,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        theta_ref_mc=theta_ref_mc,
        theta_ref_ic=theta_ref_ic,
        wgtfac_c=wgtfac_c,
        ddqz_z_half=ddqz_z_half,
        num_levels=num_levels,
    )
    log.info("Hydrostatic adjustment computation completed.")

    # Moist initialization only runs when transport is active. The only tracer we
    # need to set is qv; the hydrometeors (qc, qi, ...) keep their zero-initialized
    # value, so we don't touch them.
    active_tracers = {name for name, _ in prognostic_state_now.tracer.active_fields()}
    if active_tracers:
        if prognostic_state_now.tracer.qv is None:
            raise ValueError(
                "Moist tracer initialization requires the 'qv' tracer to be active, "
                f"but only {sorted(active_tracers)} are present."
            )
        log.info("Initializing inwp tracers (qv from RH profile, others zero).")

        # virtual temperature theta_v * exner is the post-hydro_adjust base (not the
        # pre-adjust temperature_jw); it is independent of qv and feeds both the
        # hydrostatic pressure diagnosis and the moist-iteration first guess, so the
        # iteration converges to the same fixed point as Fortran.
        virtual_temperature = gtx.as_field(
            (dims.CellDim, dims.KDim), theta_v_ndarray * exner_ndarray, allocator=allocator
        )
        pressure_ndarray = pressure_diagnostics.diagnose_pressure_surface_to_top_ndarray(
            grid=grid,
            backend=backend,
            allocator=allocator,
            exner=prognostic_state_now.exner,
            virtual_temperature=virtual_temperature,
            ddqz_z_full=ddqz_z_full_field,
        )
        testcases_utils.init_inwp_tracers(
            rho=rho_ndarray,
            virtual_temperature=virtual_temperature.ndarray,
            pressure=pressure_ndarray,
            cell_area=cell_area,
            ddqz_z_full=ddqz_z_full_field.ndarray,
            qv=prognostic_state_now.tracer.qv.ndarray,
            global_reductions=global_reductions,
            n_iter=config.moisture_init_iterations,
            rh_at_1000hpa=config.rh_at_1000hpa,
            qv_max=config.qv_max,
            global_moisture_content=config.global_moisture_content,
            normalize_global_moisture=config.normalize_global_moisture,
        )
