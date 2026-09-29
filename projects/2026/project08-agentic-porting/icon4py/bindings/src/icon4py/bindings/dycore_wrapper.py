# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


"""
Wrapper module for dycore granule.

Module contains a solve_nh_init and solve_nh_run function that follow the architecture of
Fortran granule interfaces:
- all arguments needed from external sources are passed.
- passing of scalar types or fields of simple types
"""

import dataclasses
import logging
from collections.abc import Callable
from typing import Annotated

import gt4py.next as gtx
import numpy as np
from gt4py.next import config as gtx_config
from gt4py.next.instrumentation import metrics as gtx_metrics
from gt4py.next.type_system import type_specifications as ts

from icon4py.bindings import (
    common as wrapper_common,
    config as wrapper_config,
    grid_wrapper,
    icon4py_export,
)
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro
from icon4py.model.common import dimension as dims, model_backends, utils as common_utils
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.utils import data_allocation as data_alloc, field_utils
from icon4py.tools import py2fgen


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SolveNonhydroGranule:
    solve_nh: solve_nonhydro.SolveNonhydro
    dummy_field_factory: Callable


granule: SolveNonhydroGranule | None  # TODO(havogt): remove module global state


@icon4py_export.export
def solve_nh_init(  # noqa: PLR0917 [too-many-positional-arguments]
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], gtx.float64],
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], gtx.float64],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], gtx.float64],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], gtx.float64],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    rbf_vec_coeff_e: wrapper_common.Float64Array2D,
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.float64],
    rbf_vec_coeff_v: wrapper_common.Float64Array3D,
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.float64],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    nudgecoeff_e: gtx.Field[gtx.Dims[dims.EdgeDim], gtx.float64],
    mask_prog_halo_c: gtx.Field[gtx.Dims[dims.CellDim], bool],
    rayleigh_w: gtx.Field[gtx.Dims[dims.KDim], gtx.float64],
    exner_exfac: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    exner_ref_mc: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    wgtfac_c: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    wgtfacq_c: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    inv_ddqz_z_full: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    rho_ref_mc: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    theta_ref_mc: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    vwind_expl_wgt: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    d_exner_dz_ref_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    ddqz_z_half: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    theta_ref_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    d2dexdz2_fac1_mc: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    d2dexdz2_fac2_mc: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    rho_ref_me: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    theta_ref_me: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    ddxn_z_full: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    zdiff_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.float64],
    vertidx_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32],
    pg_edgeidx: wrapper_common.OptionalInt32Array1D,
    pg_vertidx: wrapper_common.OptionalInt32Array1D,
    pg_exdist: wrapper_common.OptionalFloat64Array1D,
    ddqz_z_full_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    ddxt_z_full: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    wgtfac_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    wgtfacq_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    vwind_impl_wgt: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    hmask_dd3d: gtx.Field[gtx.Dims[dims.EdgeDim], gtx.float64],
    scalfac_dd3d: gtx.Field[gtx.Dims[dims.KDim], gtx.float64],
    coeff1_dwdz: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    coeff2_dwdz: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    c_owner_mask: gtx.Field[gtx.Dims[dims.CellDim], bool],
    itime_scheme: gtx.int32,
    iadv_rhotheta: gtx.int32,
    igradp_method: gtx.int32,
    rayleigh_type: gtx.int32,
    divdamp_order: gtx.int32,
    divdamp_type: gtx.int32,
    l_vert_nested: bool,
    ldeepatmo: bool,
    iau_init: bool,
    extra_diffu: bool,
    rhotheta_offctr: gtx.float64,
    veladv_offctr: gtx.float64,
    nudge_max_coeff: gtx.float64,  # note: this is the scaled ICON value, i.e. not the namelist value
    divdamp_fac: gtx.float64,
    divdamp_fac2: gtx.float64,
    divdamp_fac3: gtx.float64,
    divdamp_fac4: gtx.float64,
    divdamp_z: gtx.float64,
    divdamp_z2: gtx.float64,
    divdamp_z3: gtx.float64,
    divdamp_z4: gtx.float64,
    nflat_gradp: gtx.int32,
    backend: gtx.int32,
):
    if grid_wrapper.grid_state is None:
        raise Exception("Need to initialise grid using 'grid_init' before running 'solve_nh_init'.")

    xp = c_lin_e.array_ns
    on_gpu = xp != np  # TODO(havogt): expose `on_gpu` from py2fgen
    actual_backend = wrapper_common.select_backend(
        wrapper_common.BackendIntEnum(backend), on_gpu=on_gpu
    )
    backend_name = actual_backend.name if hasattr(actual_backend, "name") else actual_backend
    logger.info(f"Using Backend {backend_name} with on_gpu={on_gpu}")
    allocator = model_backends.get_allocator(actual_backend)

    pg_exdist_domain = rho_ref_me.domain
    if any(field is None for field in [pg_edgeidx, pg_vertidx, pg_exdist]):
        assert all(field is None for field in [pg_edgeidx, pg_vertidx, pg_exdist])
        pg_exdist_dsl = gtx.zeros(pg_exdist_domain, dtype=gtx.float64, allocator=allocator)
    else:
        pg_exdist_dsl = data_alloc.list2field(
            domain=pg_exdist_domain,
            values=pg_exdist,
            indices=(
                data_alloc.adjust_fortran_indices(pg_edgeidx),
                data_alloc.adjust_fortran_indices(pg_vertidx),
            ),
            default_value=gtx.float64(0.0),
            allocator=allocator,
        )

    config = solve_nonhydro.NonHydrostaticConfig(
        itime_scheme=itime_scheme,
        iadv_rhotheta=iadv_rhotheta,
        igradp_method=igradp_method,
        rayleigh_type=rayleigh_type,
        divdamp_order=divdamp_order,
        divdamp_type=divdamp_type,
        l_vert_nested=l_vert_nested,
        deepatmos_mode=ldeepatmo,
        iau_init=iau_init,
        extra_diffu=extra_diffu,
        rhotheta_offctr=rhotheta_offctr,
        veladv_offctr=veladv_offctr,
        max_nudging_coefficient=nudge_max_coeff,
        fourth_order_divdamp_factor=divdamp_fac,
        fourth_order_divdamp_factor2=divdamp_fac2,
        fourth_order_divdamp_factor3=divdamp_fac3,
        fourth_order_divdamp_factor4=divdamp_fac4,
        fourth_order_divdamp_z=divdamp_z,
        fourth_order_divdamp_z2=divdamp_z2,
        fourth_order_divdamp_z3=divdamp_z3,
        fourth_order_divdamp_z4=divdamp_z4,
    )
    nonhydro_params = solve_nonhydro.NonHydrostaticParams(config)

    # Create separate fields for the two components of the RBF vector coefficients and swap.
    # TODO(havogt): we could use GT4Py's named collections.
    rbf_coeff_1 = gtx.as_field(
        [dims.VertexDim, dims.V2EDim], xp.transpose(rbf_vec_coeff_v[:, 0, :]), allocator=allocator
    )
    rbf_coeff_2 = gtx.as_field(
        [dims.VertexDim, dims.V2EDim], xp.transpose(rbf_vec_coeff_v[:, 1, :]), allocator=allocator
    )

    # Swap indices in rbf_vec_coeff_e. TODO(havogt): Should eventually be done on the Fortran side.
    rbf_vec_coeff_e_transposed = gtx.as_field(
        [dims.EdgeDim, dims.E2C2EDim], xp.transpose(rbf_vec_coeff_e), allocator=allocator
    )
    interpolation_state = dycore_states.InterpolationState(
        c_lin_e=c_lin_e,
        c_intp=c_intp,
        e_flx_avg=e_flx_avg,
        geofac_grdiv=geofac_grdiv,
        geofac_rot=geofac_rot,
        pos_on_tplane_e_1=pos_on_tplane_e_1[:, 0:2],
        pos_on_tplane_e_2=pos_on_tplane_e_2[:, 0:2],
        rbf_vec_coeff_e=rbf_vec_coeff_e_transposed,
        e_bln_c_s=e_bln_c_s,
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        geofac_div=geofac_div,
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        nudgecoeff_e=nudgecoeff_e,
    )

    nlev = wgtfac_c.domain[dims.KDim].unit_range.stop - 1
    if len(wgtfacq_c.domain[dims.KDim].unit_range) != 3:
        raise ValueError(
            f"Expected wgtfacq_c to have a vertical dimension of size 3, but got {len(wgtfacq_c.domain[dims.KDim].unit_range)}."
        )
    # uses GT4Py's embedded shift to move the domain to surface levels
    wgtfacq_c = field_utils.flip(wgtfacq_c(dims.KDim - (nlev - 3)), dims.KDim, allocator=allocator)

    if len(wgtfacq_e.domain[dims.KDim].unit_range) != 3:
        raise ValueError(
            f"Expected wgtfacq_e to have a vertical dimension of size 3, but got {len(wgtfacq_e.domain[dims.KDim].unit_range)}."
        )
    # uses GT4Py's embedded shift to move the domain to surface levels
    wgtfacq_e = field_utils.flip(wgtfacq_e(dims.KDim - (nlev - 3)), dims.KDim, allocator=allocator)

    # In Fortran `vertidx_gradp` contains `0`s in areas where the array is not used.
    # When we translate to offsets we just subtract the current index, therefore these values will be negative.
    # Since in Fortran accessing index `0` would be out-of-bounds, we should be safe.
    vertoffset_gradp = field_utils.index2offset(
        data_alloc.adjust_fortran_indices(vertidx_gradp), dims.KDim, allocator
    )

    metric_state_nonhydro = dycore_states.MetricStateNonHydro(
        mask_prog_halo_c=mask_prog_halo_c,
        rayleigh_w=rayleigh_w,
        time_extrapolation_parameter_for_exner=exner_exfac,
        reference_exner_at_cells_on_model_levels=exner_ref_mc,
        wgtfac_c=wgtfac_c,
        wgtfacq_c=wgtfacq_c,
        inv_ddqz_z_full=inv_ddqz_z_full,
        reference_rho_at_cells_on_model_levels=rho_ref_mc,
        reference_theta_at_cells_on_model_levels=theta_ref_mc,
        exner_w_explicit_weight_parameter=vwind_expl_wgt,
        ddz_of_reference_exner_at_cells_on_half_levels=d_exner_dz_ref_ic,
        ddqz_z_half=ddqz_z_half,
        reference_theta_at_cells_on_half_levels=theta_ref_ic,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        reference_rho_at_edges_on_model_levels=rho_ref_me,
        reference_theta_at_edges_on_model_levels=theta_ref_me,
        ddxn_z_full=ddxn_z_full,
        zdiff_gradp=zdiff_gradp,
        vertoffset_gradp=vertoffset_gradp,
        nflat_gradp=gtx.int32(nflat_gradp - 1),  # Fortran vs Python indexing
        pg_exdist=pg_exdist_dsl,
        ddqz_z_full_e=ddqz_z_full_e,
        ddxt_z_full=ddxt_z_full,
        wgtfac_e=wgtfac_e,
        wgtfacq_e=wgtfacq_e,
        exner_w_implicit_weight_parameter=vwind_impl_wgt,
        horizontal_mask_for_3d_divdamp=hmask_dd3d,
        scaling_factor_for_3d_divdamp=scalfac_dd3d,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        coeff_gradekin=coeff_gradekin,
    )

    global granule  # noqa: PLW0603 [global-statement]
    granule = SolveNonhydroGranule(
        solve_nh=solve_nonhydro.SolveNonhydro(
            grid=grid_wrapper.grid_state.grid,
            config=config,
            params=nonhydro_params,
            metric_state_nonhydro=metric_state_nonhydro,
            interpolation_state=interpolation_state,
            vertical_params=grid_wrapper.grid_state.vertical_grid,
            edge_geometry=grid_wrapper.grid_state.edge_geometry,
            cell_geometry=grid_wrapper.grid_state.cell_geometry,
            owner_mask=c_owner_mask,
            backend=actual_backend,
            exchange=grid_wrapper.grid_state.exchange_runtime,
        ),
        dummy_field_factory=wrapper_common.cached_dummy_field_factory(allocator),
    )
    if wrapper_config.WAIT_FOR_COMPILATION:
        gtx.wait_for_compilation()


type NumpyFloatArray1D = Annotated[
    np.ndarray,
    py2fgen.ArrayParamDescriptor(
        rank=1,
        dtype=ts.ScalarKind.FLOAT64,
        memory_space=py2fgen.MemorySpace.HOST,
        is_optional=False,
    ),
]


@icon4py_export.export
def solve_nh_run(  # noqa: PLR0917 [too-many-positional-arguments]
    rho_now: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    rho_new: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    exner_now: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    exner_new: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    w_now: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    w_new: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    theta_v_now: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    theta_v_new: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    vn_now: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    vn_new: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    w_concorr_c: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    ddt_vn_apc_ntl1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    ddt_vn_apc_ntl2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    ddt_w_adv_ntl1: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    ddt_w_adv_ntl2: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    theta_v_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    rho_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    exner_pr: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    exner_dyn_incr: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    ddt_exner_phy: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    grf_tend_rho: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    grf_tend_thv: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    grf_tend_w: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    mass_fl_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    ddt_vn_phy: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    grf_tend_vn: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    vn_ie: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    vt: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    vn_incr: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64] | None,
    rho_incr: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    exner_incr: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    mass_flx_me: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    mass_flx_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    vol_flx_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    vn_traj: gtx.Field[gtx.Dims[dims.EdgeDim, dims.KDim], gtx.float64],
    dtime: gtx.float64,
    max_vcfl_size1_array: NumpyFloatArray1D,  # receive from Fortran as a single-element array
    lprep_adv: bool,
    at_initial_timestep: bool,
    divdamp_fac_o2: gtx.float64,
    ndyn_substeps_var: gtx.int32,
    idyn_timestep: gtx.int32,
    is_iau_active: bool,
    iau_wgt_dyn: gtx.float64,
):
    if granule is None:
        raise RuntimeError("SolveNonhydro granule not initialized. Call 'solve_nh_init' first.")

    xp = rho_now.array_ns

    if vn_incr is None:
        vn_incr = granule.dummy_field_factory("vn_incr", domain=vn_now.domain, dtype=vn_now.dtype)

    if rho_incr is None:
        rho_incr = granule.dummy_field_factory(
            "rho_incr", domain=rho_now.domain, dtype=rho_now.dtype
        )

    if exner_incr is None:
        exner_incr = granule.dummy_field_factory(
            "exner_incr", domain=exner_now.domain, dtype=exner_now.dtype
        )

    prep_adv = dycore_states.PrepAdvection(
        vn_traj=vn_traj,
        mass_flx_me=mass_flx_me,
        dynamical_vertical_mass_flux_at_cells_on_half_levels=mass_flx_ic,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=vol_flx_ic,
    )

    # Make `max_vcfl` a 0-d array to avoid cupy synchronization, see `velocity_advection.py`.
    # Note, `max_vcfl` needs to be passed back to Fortran after the timestep.
    max_vcfl = data_alloc.scalar_like_array(max_vcfl_size1_array[0], xp)

    diagnostic_state_nh = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=max_vcfl,
        theta_v_at_cells_on_half_levels=theta_v_ic,
        perturbed_exner_at_cells_on_model_levels=exner_pr,
        rho_at_cells_on_half_levels=rho_ic,
        exner_tendency_due_to_slow_physics=ddt_exner_phy,
        grf_tend_rho=grf_tend_rho,
        grf_tend_thv=grf_tend_thv,
        grf_tend_w=grf_tend_w,
        mass_flux_at_edges_on_model_levels=mass_fl_e,
        normal_wind_tendency_due_to_slow_physics_process=ddt_vn_phy,
        grf_tend_vn=grf_tend_vn,
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            ddt_vn_apc_ntl1, ddt_vn_apc_ntl2
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            ddt_w_adv_ntl1, ddt_w_adv_ntl2
        ),
        tangential_wind=vt,
        vn_on_half_levels=vn_ie,
        contravariant_correction_at_cells_on_half_levels=w_concorr_c,
        rho_iau_increment=rho_incr,
        normal_wind_iau_increment=vn_incr,
        exner_iau_increment=exner_incr,
        exner_dynamical_increment=exner_dyn_incr,
    )

    prognostic_state_nnow = PrognosticState(
        w=w_now,
        vn=vn_now,
        theta_v=theta_v_now,
        rho=rho_now,
        exner=exner_now,
    )
    prognostic_state_nnew = PrognosticState(
        w=w_new,
        vn=vn_new,
        theta_v=theta_v_new,
        rho=rho_new,
        exner=exner_new,
    )
    prognostic_states = common_utils.TimeStepPair(prognostic_state_nnow, prognostic_state_nnew)

    # adjust for Fortran indexes
    idyn_timestep = idyn_timestep - 1

    granule.solve_nh.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        prep_adv=prep_adv,
        second_order_divdamp_factor=divdamp_fac_o2,
        dtime=dtime,
        ndyn_substeps_var=ndyn_substeps_var,
        at_initial_timestep=at_initial_timestep,
        lprep_adv=lprep_adv,
        at_first_substep=idyn_timestep == 0,
        at_last_substep=idyn_timestep == (ndyn_substeps_var - 1),
        is_iau_active=is_iau_active,
        iau_wgt_dyn=iau_wgt_dyn,
    )

    # TODO(havogt): create separate bindings for writing the timers
    if gtx_config.COLLECT_METRICS_LEVEL > 0:
        gtx_metrics.dump_json("gt4py_timers.json")

    max_vcfl_size1_array[0] = diagnostic_state_nh.max_vertical_cfl[()]  # pass back to Fortran
