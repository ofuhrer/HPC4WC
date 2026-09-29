# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging
import pathlib

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common import (
    constants as phy_const,
    dimension as dims,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, states as grid_states
from icon4py.model.common.interpolation.stencils import (
    cell_2_edge_interpolation,
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver.testcases import utils as testcases_utils
from icon4py.model.testing import serialbox as sb


log = logging.getLogger(__name__)


def model_initialization_gauss3d(  # noqa: PLR0915 [too-many-statements]
    *,
    grid: icon_grid.IconGrid,
    edge_param: grid_states.EdgeParams,
    path: pathlib.Path,
    backend: gtx_typing.Backend | None,
    rank=0,
) -> tuple[
    diffusion_states.DiffusionDiagnosticState,
    dycore_states.DiagnosticStateNonHydro,
    dycore_states.PrepAdvection,
    float,
    diagnostics.DiagnosticState,
    prognostics.PrognosticState,
    prognostics.PrognosticState,
]:
    """
    Initial condition for the Gauss 3D test.

    Args:
        grid: IconGrid
        edge_param: edge properties
        path: path where to find the input data
        backend: GT4Py backend
        rank: mpi rank of the current compute node
    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """

    allocator = model_backends.get_allocator(backend)
    data_provider = sb.IconSerialDataProvider(
        backend=backend,
        fname_prefix="icon_pydycore",
        path=str(path.absolute()),
        do_print=False,
        mpi_rank=rank,
    )

    xp = data_alloc.import_array_ns(allocator)

    wgtfac_c = data_provider.from_metrics_savepoint().wgtfac_c().ndarray
    ddqz_z_half = data_provider.from_metrics_savepoint().ddqz_z_half().ndarray
    theta_ref_mc = data_provider.from_metrics_savepoint().theta_ref_mc().ndarray
    theta_ref_ic = data_provider.from_metrics_savepoint().theta_ref_ic().ndarray
    exner_ref_mc = data_provider.from_metrics_savepoint().exner_ref_mc().ndarray
    d_exner_dz_ref_ic = data_provider.from_metrics_savepoint().d_exner_dz_ref_ic().ndarray
    geopot = data_provider.from_metrics_savepoint().geopot().ndarray

    primal_normal_x = edge_param.primal_normal[0].ndarray

    cell_2_edge_coeff = data_provider.from_interpolation_savepoint().c_lin_e()
    rbf_vec_coeff_c1 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c1()
    rbf_vec_coeff_c2 = data_provider.from_interpolation_savepoint().rbf_vec_coeff_c2()

    num_cells = grid.num_cells
    num_edges = grid.num_edges
    num_levels = grid.num_levels

    edge_domain = h_grid.domain(dims.EdgeDim)
    cell_domain = h_grid.domain(dims.CellDim)
    end_edge_lateral_boundary_level_2 = grid.end_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_edge_end = grid.end_index(edge_domain(h_grid.Zone.END))
    end_cell_lateral_boundary_level_2 = grid.end_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))

    w_ndarray = xp.zeros((num_cells, num_levels + 1), dtype=ta.wpfloat)
    exner_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)
    rho_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)
    temperature_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)
    pressure_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)
    theta_v_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)
    eta_v_ndarray = xp.zeros((num_cells, num_levels), dtype=ta.wpfloat)

    mask_array_edge_start_plus1_to_edge_end = xp.ones(num_edges, dtype=bool)
    mask_array_edge_start_plus1_to_edge_end[0:end_edge_lateral_boundary_level_2] = False
    mask = xp.repeat(
        xp.expand_dims(mask_array_edge_start_plus1_to_edge_end, axis=-1),
        num_levels,
        axis=1,
    )
    primal_normal_x = xp.repeat(xp.expand_dims(primal_normal_x, axis=-1), num_levels, axis=1)

    # Define test case parameters
    # The topography can only be read from serialized data for now, then these
    # variables should be defined here and used to compute the idealized
    # topography:
    # - mount_lon
    # - mount_lat
    # - mount_height
    # - mount_width
    nh_t0 = 300.0
    nh_u0 = 0.0
    nh_brunt_vais = 0.01
    log.info("Topography can only be read from serialized data for now.")

    # Horizontal wind field
    u = xp.where(mask, nh_u0, 0.0)
    vn_ndarray = u * primal_normal_x
    log.info("Wind profile assigned.")

    # Vertical temperature profile
    for k_index in range(num_levels - 1, -1, -1):
        z_help = (nh_brunt_vais / phy_const.GRAV) ** 2 * geopot[:, k_index]
        # profile of theta is explicitly given
        theta_v_ndarray[:, k_index] = nh_t0 * xp.exp(z_help)

    # Lower boundary condition for exner pressure
    if nh_brunt_vais != 0.0:
        z_help = (nh_brunt_vais / phy_const.GRAV) ** 2 * geopot[:, num_levels - 1]
        exner_ndarray[:, num_levels - 1] = (
            phy_const.GRAV / nh_brunt_vais
        ) ** 2 / nh_t0 / phy_const.CPD * (xp.exp(-z_help) - 1.0) + 1.0
    else:
        exner_ndarray[:, num_levels - 1] = 1.0 - geopot[:, num_levels - 1] / phy_const.CPD / nh_t0
    log.info("Vertical computations completed.")

    # Compute hydrostatically balanced exner, by integrating the (discretized!)
    # 3rd equation of motion under the assumption thetav=const.
    rho_ndarray, exner_ndarray = testcases_utils.hydrostatic_adjustment_constant_thetav_ndarray(
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
    log.info("Hydrostatic adjustment computation completed.")

    eta_v = gtx.as_field((dims.CellDim, dims.KDim), eta_v_ndarray, allocator=allocator)
    eta_v_e = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator)
    cell_2_edge_interpolation.cell_2_edge_interpolation.with_backend(backend)(
        eta_v,
        cell_2_edge_coeff,
        eta_v_e,
        end_edge_lateral_boundary_level_2,
        end_edge_end,
        0,
        num_levels,
        offset_provider=grid.connectivities,
    )
    log.info("Cell-to-edge eta_v computation completed.")

    pressure_ifc_ndarray = xp.zeros((num_cells, num_levels + 1), dtype=ta.wpfloat)
    (
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
    ) = testcases_utils.create_gt4py_field_for_prognostic_and_diagnostic_variables(
        vn_ndarray=vn_ndarray,
        w_ndarray=w_ndarray,
        exner_ndarray=exner_ndarray,
        rho_ndarray=rho_ndarray,
        theta_v_ndarray=theta_v_ndarray,
        temperature_ndarray=temperature_ndarray,
        pressure_ndarray=pressure_ndarray,
        pressure_ifc_ndarray=pressure_ifc_ndarray,
        grid=grid,
        allocator=allocator,
    )

    edge_2_cell_vector_rbf_interpolation.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
        vn,
        rbf_vec_coeff_c1,
        rbf_vec_coeff_c2,
        u,
        v,
        end_cell_lateral_boundary_level_2,
        end_cell_end,
        0,
        num_levels,
        offset_provider=grid.connectivities,
    )
    log.info("U, V computation completed.")

    perturbed_exner = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=allocator)
    testcases_utils.compute_perturbed_exner.with_backend(backend)(
        exner,
        data_provider.from_metrics_savepoint().exner_ref_mc(),
        perturbed_exner,
        0,
        num_cells,
        0,
        num_levels,
        offset_provider={},
    )
    log.info("perturbed_exner initialization completed.")

    diagnostic_state = diagnostics.DiagnosticState(
        pressure=pressure,
        pressure_ifc=pressure_ifc,
        temperature=temperature,
        virtual_temperature=virtual_temperature,
        u=u,
        v=v,
    )

    prognostic_state_now = prognostics.PrognosticState(
        w=w, vn=vn, theta_v=theta_v, rho=rho, exner=exner
    )
    prognostic_state_next = prognostics.PrognosticState(
        w=w_next, vn=vn_next, theta_v=theta_v_next, rho=rho_next, exner=exner_next
    )

    diffusion_diagnostic_state = diffusion_states.initialize_diffusion_diagnostic_state(
        grid=grid, allocator=allocator
    )
    solve_nonhydro_diagnostic_state = dycore_states.initialize_solve_nonhydro_diagnostic_state(
        perturbed_exner_at_cells_on_model_levels=perturbed_exner,
        grid=grid,
        allocator=allocator,
    )

    prep_adv = dycore_states.initialize_prep_advection(grid=grid, allocator=allocator)
    log.info("Initialization completed.")

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        0.0,  # divdamp_fac_o2 only != 0 for data assimilation
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )
