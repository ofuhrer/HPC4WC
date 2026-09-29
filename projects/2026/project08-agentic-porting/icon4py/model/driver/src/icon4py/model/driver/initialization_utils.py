# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import functools
import logging
import pathlib

import gt4py.next.typing as gtx_typing
import netCDF4 as nc4

from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common import dimension as dims, field_type_aliases as fa, utils as common_utils
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import icon as icon_grid, states as grid_states, vertical as v_grid
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver.testcases import gauss3d, jablonowski_williamson
from icon4py.model.testing import serialbox as sb


SB_ONLY_MSG = "Only ser_type='sb' is implemented so far."
INITIALIZATION_ERROR_MSG = "The requested experiment type is not implemented."

SIMULATION_START_DATE = "2021-06-20T12:00:10.000"
log = logging.getLogger(__name__)


class SerializationType(enum.StrEnum):
    SB = "serialbox"
    NC = "netcdf"


class ExperimentType(enum.StrEnum):
    JABW = "jabw"
    """initial condition of Jablonowski-Williamson test"""
    GAUSS3D = "gauss3d_torus"
    """initial condition of Gauss 3D test"""
    ANY = "any"
    """any test with initial conditions read from serialized data (remember to set correct SIMULATION_START_DATE)"""


def read_icon_grid(
    path: pathlib.Path,
    grid_file: pathlib.Path,
    backend: gtx_typing.Backend,
    rank: int = 0,
    ser_type: SerializationType = SerializationType.SB,
) -> icon_grid.IconGrid:
    """
    Read icon grid.

    Args:
        path: path where to find the input data
        grid_file: path of the grid
        backend: GT4Py backend
        rank: mpi rank of the current compute node
        ser_type: type of input data. Currently only 'sb (serialbox)' is supported. It reads from ppser serialized test data
    Returns:  IconGrid parsed from a given input type.
    """
    if ser_type == SerializationType.SB:
        return _grid_savepoint(
            backend=backend,
            path=path,
            grid_file=grid_file,
            rank=rank,
        ).construct_icon_grid(backend=backend)
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def model_initialization_serialbox(
    grid: icon_grid.IconGrid,
    path: pathlib.Path,
    backend: gtx_typing.Backend,
    rank: int = 0,
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
    Initial condition read from serialized data. Diagnostic variables are allocated as zero
    fields.

    Args:
        grid: IconGrid
        path: path where to find the input data
        backend: GT4Py backend
        rank: mpi rank of the current compute node
    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """

    data_provider = _serial_data_provider(backend, path, rank)
    diffusion_init_savepoint = data_provider.from_savepoint_diffusion_init(
        linit=True, date=SIMULATION_START_DATE
    )
    solve_nonhydro_init_savepoint = data_provider.from_savepoint_nonhydro_init(
        istep=1, date=SIMULATION_START_DATE, substep=1
    )
    velocity_init_savepoint = data_provider.from_savepoint_velocity_init(
        istep=1, date=SIMULATION_START_DATE, substep=1
    )
    prognostic_state_now = diffusion_init_savepoint.construct_prognostics()
    diffusion_diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=diffusion_init_savepoint.hdef_ic(),
        div_ic=diffusion_init_savepoint.div_ic(),
        dwdx=diffusion_init_savepoint.dwdx(),
        dwdy=diffusion_init_savepoint.dwdy(),
    )
    solve_nonhydro_diagnostic_state = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=data_alloc.scalar_like_array(0.0, backend),
        theta_v_at_cells_on_half_levels=solve_nonhydro_init_savepoint.theta_v_ic(),
        perturbed_exner_at_cells_on_model_levels=solve_nonhydro_init_savepoint.exner_pr(),
        rho_at_cells_on_half_levels=solve_nonhydro_init_savepoint.rho_ic(),
        exner_tendency_due_to_slow_physics=solve_nonhydro_init_savepoint.ddt_exner_phy(),
        grf_tend_rho=solve_nonhydro_init_savepoint.grf_tend_rho(),
        grf_tend_thv=solve_nonhydro_init_savepoint.grf_tend_thv(),
        grf_tend_w=solve_nonhydro_init_savepoint.grf_tend_w(),
        mass_flux_at_edges_on_model_levels=solve_nonhydro_init_savepoint.mass_fl_e(),
        normal_wind_tendency_due_to_slow_physics_process=solve_nonhydro_init_savepoint.ddt_vn_phy(),
        grf_tend_vn=solve_nonhydro_init_savepoint.grf_tend_vn(),
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            velocity_init_savepoint.ddt_vn_apc_pc(0),
            velocity_init_savepoint.ddt_vn_apc_pc(1),
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            velocity_init_savepoint.ddt_w_adv_pc(0),
            velocity_init_savepoint.ddt_w_adv_pc(1),
        ),
        tangential_wind=velocity_init_savepoint.vt(),
        vn_on_half_levels=velocity_init_savepoint.vn_ie(),
        contravariant_correction_at_cells_on_half_levels=velocity_init_savepoint.w_concorr_c(),
        rho_iau_increment=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
        normal_wind_iau_increment=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, allocator=backend
        ),
        exner_iau_increment=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend),
        exner_dynamical_increment=solve_nonhydro_init_savepoint.exner_dyn_incr(),
    )

    diagnostic_state = diagnostics.DiagnosticState(
        pressure=data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KDim,
            allocator=backend,
        ),
        pressure_ifc=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
        ),
        temperature=data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KDim,
            allocator=backend,
        ),
        virtual_temperature=data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KDim,
            allocator=backend,
        ),
        u=data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KDim,
            allocator=backend,
        ),
        v=data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KDim,
            allocator=backend,
        ),
    )

    prognostic_state_next = prognostics.PrognosticState(
        w=solve_nonhydro_init_savepoint.w_new(),
        vn=solve_nonhydro_init_savepoint.vn_new(),
        theta_v=solve_nonhydro_init_savepoint.theta_v_new(),
        rho=solve_nonhydro_init_savepoint.rho_new(),
        exner=solve_nonhydro_init_savepoint.exner_new(),
    )

    prep_adv = dycore_states.PrepAdvection(
        vn_traj=solve_nonhydro_init_savepoint.vn_traj(),
        mass_flx_me=solve_nonhydro_init_savepoint.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=solve_nonhydro_init_savepoint.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KDim,
            allocator=backend,
        ),
    )

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        solve_nonhydro_init_savepoint.divdamp_fac_o2(),
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )


def read_initial_state(
    *,
    grid: icon_grid.IconGrid,
    cell_param: grid_states.CellParams,
    edge_param: grid_states.EdgeParams,
    path: pathlib.Path,
    backend: gtx_typing.Backend,
    rank=0,
    experiment_type: ExperimentType = ExperimentType.ANY,
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
    Read initial prognostic and diagnostic fields.

    Args:
        grid: IconGrid
        cell_param: cell properties
        edge_param: edge properties
        path: path to the serialized input data
        backend: GT4Py backend
        rank: mpi rank of the current compute node
        experiment_type: (optional) defaults to ANY=any, type of initial condition to be read

    Returns:  A tuple containing Diagnostic variables for diffusion and solve_nonhydro granules,
        PrepAdvection, second order divdamp factor, diagnostic variables, and two prognostic
        variables (now and next).
    """
    if experiment_type == ExperimentType.JABW:
        (
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prep_adv,
            divdamp_fac_o2,
            diagnostic_state,
            prognostic_state_now,
            prognostic_state_next,
        ) = jablonowski_williamson.model_initialization_jabw(
            grid=grid,
            cell_param=cell_param,
            edge_param=edge_param,
            path=path,
            backend=backend,
            rank=rank,
        )
    elif experiment_type == ExperimentType.GAUSS3D:
        (
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prep_adv,
            divdamp_fac_o2,
            diagnostic_state,
            prognostic_state_now,
            prognostic_state_next,
        ) = gauss3d.model_initialization_gauss3d(
            grid=grid,
            edge_param=edge_param,
            path=path,
            backend=backend,
            rank=rank,
        )
    elif experiment_type == ExperimentType.ANY:
        (
            diffusion_diagnostic_state,
            solve_nonhydro_diagnostic_state,
            prep_adv,
            divdamp_fac_o2,
            diagnostic_state,
            prognostic_state_now,
            prognostic_state_next,
        ) = model_initialization_serialbox(
            grid=grid,
            path=path,
            backend=backend,
            rank=rank,
        )
    else:
        raise NotImplementedError(INITIALIZATION_ERROR_MSG)

    return (
        diffusion_diagnostic_state,
        solve_nonhydro_diagnostic_state,
        prep_adv,
        divdamp_fac_o2,
        diagnostic_state,
        prognostic_state_now,
        prognostic_state_next,
    )


def read_geometry_fields(
    *,
    path: pathlib.Path,
    grid_file: pathlib.Path,
    vertical_grid_config: v_grid.VerticalGridConfig,
    backend: gtx_typing.Backend,
    rank: int = 0,
    ser_type: SerializationType = SerializationType.SB,
) -> tuple[
    grid_states.EdgeParams,
    grid_states.CellParams,
    v_grid.VerticalGrid,
    fa.CellField[bool],
]:
    """
    Read fields containing grid properties.

    Args:
        path: path to the serialized input data
        grid_file: path of the grid
        vertical_grid_config: Vertical grid configuration
        backend: GT4py backend
        rank: mpi rank of the current compute node
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns: a tuple containing fields describing edges, cells, vertical properties of the model
        the data is originally obtained from the grid file (horizontal fields) or some special input files.
    """
    if ser_type == SerializationType.SB:
        sp = _grid_savepoint(backend, path, grid_file, rank)
        edge_geometry = sp.construct_edge_geometry()
        cell_geometry = sp.construct_cell_geometry()
        vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_grid_config, backend)
        vertical_geometry = v_grid.VerticalGrid(
            config=vertical_grid_config,
            vct_a=vct_a,
            vct_b=vct_b,
        )
        return edge_geometry, cell_geometry, vertical_geometry, sp.c_owner_mask()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


# TODO(OngChia): cannot be cached (@functools.cache) after adding backend. TypeError: unhashable type: 'CompiledbFactory'
def _serial_data_provider(
    backend: gtx_typing.Backend,
    path: pathlib.Path,
    rank: int,
) -> sb.IconSerialDataProvider:
    return sb.IconSerialDataProvider(
        backend=backend,
        fname_prefix="icon_pydycore",
        path=str(path.absolute()),
        do_print=False,
        mpi_rank=rank,
    )


# TODO(OngChia): cannot be cached (@functools.cache) after adding backend. TypeError: unhashable type: 'CompiledbFactory'
def _grid_savepoint(
    backend: gtx_typing.Backend,
    path: pathlib.Path,
    grid_file: pathlib.Path,
    rank: int,
) -> sb.IconGridSavepoint:
    grid_params, grid_uuid = _create_grid_params(grid_file)
    sp = _serial_data_provider(backend, path, rank).from_savepoint_grid(
        grid_uuid,
        grid_params,
    )
    return sp


def read_decomp_info(
    path: pathlib.Path,
    grid_file: pathlib.Path,
    process_props: decomposition.ProcessProperties,
    backend: gtx_typing.Backend,
    ser_type=SerializationType.SB,
) -> decomposition.DecompositionInfo:
    if ser_type == SerializationType.SB:
        return _grid_savepoint(
            backend,
            path,
            grid_file,
            process_props.rank,
        ).construct_decomposition_info()
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def read_static_fields(
    path: pathlib.Path,
    grid_file: pathlib.Path,
    backend: gtx_typing.Backend,
    rank: int = 0,
    ser_type: SerializationType = SerializationType.SB,
) -> tuple[
    diffusion_states.DiffusionMetricState,
    diffusion_states.DiffusionInterpolationState,
    dycore_states.MetricStateNonHydro,
    dycore_states.InterpolationState,
    diagnostics.DiagnosticMetricState,
]:
    """
    Read fields for metric and interpolation state.

     Args:
        path: path to the serialized input data
        grid_file: path of the grid
        backend: GT4Py backend
        rank: mpi rank, defaults to 0 for serial run
        ser_type: (optional) defaults to SB=serialbox, type of input data to be read

    Returns:
        a tuple containing the metric_state and interpolation state,
        the fields are precalculated in the icon setup.

    """
    if ser_type == SerializationType.SB:
        data_provider = _serial_data_provider(backend, path, rank)

        interpolation_savepoint = data_provider.from_interpolation_savepoint()
        metrics_savepoint = data_provider.from_metrics_savepoint()
        grg = interpolation_savepoint.geofac_grg()
        diffusion_interpolation_state = diffusion_states.DiffusionInterpolationState(
            e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
            rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
            rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
            geofac_div=interpolation_savepoint.geofac_div(),
            geofac_n2s=interpolation_savepoint.geofac_n2s(),
            geofac_grg_x=grg[0],
            geofac_grg_y=grg[1],
            nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
        )
        diffusion_metric_state = diffusion_states.DiffusionMetricState(
            theta_ref_mc=metrics_savepoint.theta_ref_mc(),
            wgtfac_c=metrics_savepoint.wgtfac_c(),
            zd_intcoef=metrics_savepoint.zd_intcoef(),
            zd_vertoffset=metrics_savepoint.zd_vertoffset(),
            zd_diffcoef=metrics_savepoint.zd_diffcoef(),
        )
        solve_nonhydro_interpolation_state = dycore_states.InterpolationState(
            c_lin_e=interpolation_savepoint.c_lin_e(),
            c_intp=interpolation_savepoint.c_intp(),
            e_flx_avg=interpolation_savepoint.e_flx_avg(),
            geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
            geofac_rot=interpolation_savepoint.geofac_rot(),
            pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
            pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
            rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
            e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
            rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
            rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
            geofac_div=interpolation_savepoint.geofac_div(),
            geofac_n2s=interpolation_savepoint.geofac_n2s(),
            geofac_grg_x=grg[0],
            geofac_grg_y=grg[1],
            nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
        )
        metrics_savepoint = data_provider.from_metrics_savepoint()
        grid_savepoint = _grid_savepoint(backend, path, grid_file, rank)
        solve_nonhydro_metric_state = dycore_states.MetricStateNonHydro(
            mask_prog_halo_c=metrics_savepoint.mask_prog_halo_c(),
            rayleigh_w=metrics_savepoint.rayleigh_w(),
            time_extrapolation_parameter_for_exner=metrics_savepoint.exner_exfac(),
            reference_exner_at_cells_on_model_levels=metrics_savepoint.exner_ref_mc(),
            wgtfac_c=metrics_savepoint.wgtfac_c(),
            wgtfacq_c=metrics_savepoint.wgtfacq_c(),
            inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
            reference_rho_at_cells_on_model_levels=metrics_savepoint.rho_ref_mc(),
            reference_theta_at_cells_on_model_levels=metrics_savepoint.theta_ref_mc(),
            exner_w_explicit_weight_parameter=metrics_savepoint.vwind_expl_wgt(),
            ddz_of_reference_exner_at_cells_on_half_levels=metrics_savepoint.d_exner_dz_ref_ic(),
            ddqz_z_half=metrics_savepoint.ddqz_z_half(),
            reference_theta_at_cells_on_half_levels=metrics_savepoint.theta_ref_ic(),
            d2dexdz2_fac1_mc=metrics_savepoint.d2dexdz2_fac1_mc(),
            d2dexdz2_fac2_mc=metrics_savepoint.d2dexdz2_fac2_mc(),
            reference_rho_at_edges_on_model_levels=metrics_savepoint.rho_ref_me(),
            reference_theta_at_edges_on_model_levels=metrics_savepoint.theta_ref_me(),
            ddxn_z_full=metrics_savepoint.ddxn_z_full(),
            zdiff_gradp=metrics_savepoint.zdiff_gradp(),
            vertoffset_gradp=metrics_savepoint.vertoffset_gradp(),
            nflat_gradp=grid_savepoint.nflat_gradp(),
            pg_exdist=metrics_savepoint.pg_exdist_dsl(),
            ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
            ddxt_z_full=metrics_savepoint.ddxt_z_full(),
            wgtfac_e=metrics_savepoint.wgtfac_e(),
            wgtfacq_e=metrics_savepoint.wgtfacq_e(),
            exner_w_implicit_weight_parameter=metrics_savepoint.vwind_impl_wgt(),
            horizontal_mask_for_3d_divdamp=metrics_savepoint.hmask_dd3d(),
            scaling_factor_for_3d_divdamp=metrics_savepoint.scalfac_dd3d(),
            coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
            coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
            coeff_gradekin=metrics_savepoint.coeff_gradekin(),
        )

        diagnostic_metric_state = diagnostics.DiagnosticMetricState(
            ddqz_z_full=metrics_savepoint.ddqz_z_full(),
            rbf_vec_coeff_c1=interpolation_savepoint.rbf_vec_coeff_c1(),
            rbf_vec_coeff_c2=interpolation_savepoint.rbf_vec_coeff_c2(),
        )

        return (
            diffusion_metric_state,
            diffusion_interpolation_state,
            solve_nonhydro_metric_state,
            solve_nonhydro_interpolation_state,
            diagnostic_metric_state,
        )
    else:
        raise NotImplementedError(SB_ONLY_MSG)


def configure_logging(
    run_path: str,
    experiment_name: str,
    enable_output: bool = True,
    process_props: decomposition.ProcessProperties | None = None,
) -> None:
    """
    Configure logging.

    Log output is sent to console and to a file.

    Args:
        run_path: path to the output folder where the logfile should be stored
        experiment_name: name of the simulation
        enable_output: enable output logging messages above debug level
        process_props: ProcessProperties

    """
    if not enable_output:
        return
    run_dir = (
        pathlib.Path(run_path).absolute() if run_path else pathlib.Path(__file__).absolute().parent
    )
    run_dir.mkdir(exist_ok=True)
    logfile = run_dir.joinpath(f"dummy_dycore_driver_{experiment_name}.log")
    logfile.touch(exist_ok=True)
    logging_level = logging.DEBUG
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s %(filename)-20s (%(lineno)-4d) : %(funcName)-20s:  %(levelname)-8s %(message)s",
        filemode="w",
        filename=logfile,
    )
    console_handler = logging.StreamHandler()
    console_handler.addFilter(decomposition.ParallelLogger(process_props))

    log_format = "{rank} {asctime} - {filename}: {funcName:<20}: {levelname:<7} {message}"
    formatter = logging.Formatter(fmt=log_format, style="{", defaults={"rank": None})
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging_level)
    logging.getLogger("").addHandler(console_handler)


@functools.cache
def _create_grid_params(
    grid_file: pathlib.Path,
) -> tuple[icon_grid.GridParams, str]:
    """
    Create grid params and its uuid.

    Args:
        grid_file: path of the grid file

    Returns:
        grid_params: GridParams
        grid_uuid: id (uuid) of the horizontal grid
    """
    grid = nc4.Dataset(grid_file, "r", format="NETCDF4")
    grid_root = grid.getncattr("grid_root")
    grid_level = grid.getncattr("grid_level")
    grid_uuid = grid.getncattr("uuidOfHGrid")
    try:
        grid_geometry_type = icon_grid.GeometryType(grid.getncattr("grid_geometry"))
    except AttributeError:
        log.warning(
            "Global attribute grid_geometry is not found in the grid. Icosahedral grid is assumed."
        )
        grid_geometry_type = icon_grid.GeometryType.ICOSAHEDRON

    match grid_geometry_type:
        case icon_grid.GeometryType.ICOSAHEDRON:
            grid_params = icon_grid.GridParams(
                icon_grid.IcosahedronParams(
                    subdivision=icon_grid.GridSubdivision(root=grid_root, level=grid_level),
                ),
            )
        case icon_grid.GeometryType.TORUS:
            grid_params = icon_grid.GridParams(
                icon_grid.TorusParams(
                    domain_length=grid.getncattr("domain_length"),
                    domain_height=grid.getncattr("domain_height"),
                ),
            )

    grid.close()

    return grid_params, grid_uuid
