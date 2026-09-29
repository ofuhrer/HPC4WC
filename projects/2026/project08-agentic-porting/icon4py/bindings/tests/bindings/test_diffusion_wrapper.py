# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from unittest import mock

import cffi
import numpy as np
import pytest

from icon4py.bindings import common as wrapper_common, diffusion_wrapper
from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import states as grid_states, vertical as v_grid
from icon4py.model.testing import definitions, test_utils as testing_test_utils
from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import test_utils

from . import utils
from .test_grid_init import grid_init


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
    ],
)
@pytest.mark.parametrize("backend", [None])  # TODO(havogt): consider parametrizing over backends
def test_diffusion_wrapper_granule_inputs(  # noqa: PLR0917 [too-many-positional-arguments]
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    grid_init,  # initializes the grid as side-effect
    icon_grid,
    experiment,
):
    cfg = experiment.config.diffusion

    # --- Extract Metric State Parameters ---
    theta_ref_mc = test_utils.array_to_array_info(metrics_savepoint.theta_ref_mc().ndarray)
    wgtfac_c = test_utils.array_to_array_info(metrics_savepoint.wgtfac_c().ndarray)

    # The wrapper expects [cellidx, c2e2c_ids] and then extracts `zd_cellidx[0,:]` because it only needs the cellidxs
    # (this is because slicing causes issue in the bindings, but not for serialization)
    zd_cellidx = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_cellidx", metrics_savepoint.savepoint))[
            np.newaxis, :
        ]
    )
    zd_vertidx = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_vertidx", metrics_savepoint.savepoint))
    )
    zd_intcoef = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_intcoef", metrics_savepoint.savepoint))
    )
    zd_diffcoef = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_diffcoef", metrics_savepoint.savepoint))
    )

    # --- Extract Interpolation State Parameters ---
    e_bln_c_s = test_utils.array_to_array_info(interpolation_savepoint.e_bln_c_s().ndarray)
    geofac_div = test_utils.array_to_array_info(interpolation_savepoint.geofac_div().ndarray)
    geofac_grg_x_field, geofac_grg_y_field = interpolation_savepoint.geofac_grg()
    geofac_grg_x = test_utils.array_to_array_info(geofac_grg_x_field.ndarray)
    geofac_grg_y = test_utils.array_to_array_info(geofac_grg_y_field.ndarray)
    geofac_n2s = test_utils.array_to_array_info(interpolation_savepoint.geofac_n2s().ndarray)
    nudgecoeff_e = test_utils.array_to_array_info(interpolation_savepoint.nudgecoeff_e().ndarray)

    # we need the raw Fortran data instead of the postprocessed GT4Py field, see dycore_wrapper.solve_nh_init
    rbf_vec_coeff_v_array = np.squeeze(
        interpolation_savepoint.serializer.read(
            "rbf_vec_coeff_v", interpolation_savepoint.savepoint
        ).astype(float)
    )
    rbf_vec_coeff_v_array = interpolation_savepoint._reduce_to_dim_size(
        rbf_vec_coeff_v_array, [dims.V2EDim, dims.V2EDim, dims.VertexDim]
    )
    rbf_vec_coeff_v = test_utils.array_to_array_info(rbf_vec_coeff_v_array)

    # --- Extract Diagnostic and Prognostic State Parameters ---
    hdef_ic = test_utils.array_to_array_info(savepoint_diffusion_init.hdef_ic().ndarray)
    div_ic = test_utils.array_to_array_info(savepoint_diffusion_init.div_ic().ndarray)
    dwdx = test_utils.array_to_array_info(savepoint_diffusion_init.dwdx().ndarray)
    dwdy = test_utils.array_to_array_info(savepoint_diffusion_init.dwdy().ndarray)
    w = test_utils.array_to_array_info(savepoint_diffusion_init.w().ndarray)
    vn = test_utils.array_to_array_info(savepoint_diffusion_init.vn().ndarray)
    exner = test_utils.array_to_array_info(savepoint_diffusion_init.exner().ndarray)
    theta_v = test_utils.array_to_array_info(savepoint_diffusion_init.theta_v().ndarray)
    rho = test_utils.array_to_array_info(savepoint_diffusion_init.rho().ndarray)
    dtime = savepoint_diffusion_init.get_metadata("dtime")["dtime"]

    # --- Expected objects that form inputs into init and run functions
    expected_icon_grid = icon_grid
    expected_dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    expected_edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()
    expected_cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    expected_interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=interpolation_savepoint.geofac_div(),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    expected_metric_state = diffusion_states.DiffusionMetricState(
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        zd_intcoef=metrics_savepoint.zd_intcoef(),
        zd_vertoffset=metrics_savepoint.zd_vertoffset(),
        zd_diffcoef=metrics_savepoint.zd_diffcoef(),
    )
    expected_diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )
    expected_prognostic_state = savepoint_diffusion_init.construct_prognostics()
    expected_config = experiment.config.diffusion
    expected_additional_parameters = diffusion.DiffusionParams(expected_config)

    # --- Mock and Test Diffusion.init ---
    with mock.patch(
        "icon4py.model.atmosphere.diffusion.diffusion.Diffusion.__init__", return_value=None
    ) as mock_init:
        diffusion_wrapper.diffusion_init(
            ffi=cffi.FFI(),
            perf_counters=None,
            theta_ref_mc=theta_ref_mc,
            wgtfac_c=wgtfac_c,
            e_bln_c_s=e_bln_c_s,
            geofac_div=geofac_div,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            geofac_n2s=geofac_n2s,
            nudgecoeff_e=nudgecoeff_e,
            rbf_vec_coeff_v=rbf_vec_coeff_v,
            zd_cellidx=zd_cellidx,
            zd_vertidx=zd_vertidx,
            zd_intcoef=zd_intcoef,
            zd_diffcoef=zd_diffcoef,
            ndyn_substeps=cfg.ndyn_substeps,
            diffusion_type=cfg.diffusion_type,
            hdiff_w=cfg.apply_to_vertical_wind,
            hdiff_vn=cfg.apply_to_horizontal_wind,
            hdiff_smag_w=cfg.apply_smag_diff_to_vertical_wind,
            zdiffu_t=cfg.apply_zdiffusion_t,
            type_t_diffu=cfg.type_t_diffu,
            type_vn_diffu=cfg.type_vn_diffu,
            hdiff_efdt_ratio=cfg.hdiff_efdt_ratio,
            hdiff_w_efdt_ratio=cfg.hdiff_w_efdt_ratio,
            smagorinski_scaling_factor=cfg.smagorinski_scaling_factor,
            smagorinski_scaling_factor2=cfg.smagorinski_scaling_factor2,
            smagorinski_scaling_factor3=cfg.smagorinski_scaling_factor3,
            smagorinski_scaling_factor4=cfg.smagorinski_scaling_factor4,
            smagorinski_scaling_height=cfg.smagorinski_scaling_height,
            smagorinski_scaling_height2=cfg.smagorinski_scaling_height2,
            smagorinski_scaling_height3=cfg.smagorinski_scaling_height3,
            smagorinski_scaling_height4=cfg.smagorinski_scaling_height4,
            hdiff_temp=cfg.apply_to_temperature,
            denom_diffu_v=cfg.velocity_boundary_diffusion_denominator,
            nudge_max_coeff=cfg.max_nudging_coefficient,
            itype_sher=cfg.shear_type.value,
            iforcing=cfg.iforcing.value,
            a_hshr=cfg.a_hshr,
            loutshs=cfg.loutshs,
            backend=wrapper_common.BackendIntEnum.DEFAULT,
        )

        # Check input arguments to Diffusion.init
        _, captured_kwargs = mock_init.call_args

        # special case of grid._id as we do not use this arg in the wrapper as we cant pass strings from Fortran to the wrapper
        try:
            result, error_message = utils.compare_objects(
                captured_kwargs["grid"], expected_icon_grid
            )
            assert result, f"Grid comparison failed: {error_message}"
        except AssertionError as e:
            error_message = str(e)
            if "icon_grid != " not in error_message:
                raise
            else:
                pass

        result, error_message = utils.compare_objects(captured_kwargs["config"], expected_config)
        assert result, f"Config comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["params"], expected_additional_parameters
        )
        assert result, f"Params comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["metric_state"], expected_metric_state
        )
        assert result, f"Metric State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["interpolation_state"], expected_interpolation_state
        )
        assert result, f"Interpolation State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["edge_params"], expected_edge_geometry
        )
        assert result, f"Edge Params comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["cell_params"], expected_cell_geometry
        )
        assert result, f"Cell Params comparison failed: {error_message}"

    # --- Mock and Test Diffusion.run ---
    with mock.patch("icon4py.model.atmosphere.diffusion.diffusion.Diffusion.run") as mock_run:
        diffusion_wrapper.diffusion_run(
            ffi=cffi.FFI(),
            perf_counters=None,
            w=w,
            vn=vn,
            exner=exner,
            theta_v=theta_v,
            rho=rho,
            hdef_ic=hdef_ic,
            div_ic=div_ic,
            dwdx=dwdx,
            dwdy=dwdy,
            dtime=dtime,
            linit=False,
        )

        # Check input arguments to Diffusion.run
        _, captured_kwargs = mock_run.call_args
        assert utils.compare_objects(captured_kwargs["diagnostic_state"], expected_diagnostic_state)
        assert utils.compare_objects(captured_kwargs["prognostic_state"], expected_prognostic_state)
        assert captured_kwargs["dtime"] == expected_dtime


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
    ],
)
@pytest.mark.parametrize("backend", [None])  # TODO(havogt): consider parametrizing over backends
def test_diffusion_wrapper_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    savepoint_diffusion_init,
    savepoint_diffusion_exit,
    interpolation_savepoint,
    metrics_savepoint,
    grid_savepoint,
    grid_init,  # initializes the grid as side-effect
    experiment,
    step_date_init,
    step_date_exit,
):
    cfg = experiment.config.diffusion

    # Metric state parameters
    theta_ref_mc = test_utils.array_to_array_info(metrics_savepoint.theta_ref_mc().ndarray)
    wgtfac_c = test_utils.array_to_array_info(metrics_savepoint.wgtfac_c().ndarray)

    # The wrapper expects [cellidx, c2e2c_ids] and then extracts `zd_cellidx[0,:]` because it only needs the cellidxs
    # (this is because slicing causes issue in the bindings, but not for serialization)
    zd_cellidx = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_cellidx", metrics_savepoint.savepoint))[
            np.newaxis, :
        ]
    )
    zd_vertidx = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_vertidx", metrics_savepoint.savepoint))
    )
    zd_intcoef = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_intcoef", metrics_savepoint.savepoint))
    )
    zd_diffcoef = test_utils.array_to_array_info(
        np.squeeze(metrics_savepoint.serializer.read("zd_diffcoef", metrics_savepoint.savepoint))
    )

    # Interpolation state parameters
    e_bln_c_s = test_utils.array_to_array_info(interpolation_savepoint.e_bln_c_s().ndarray)
    geofac_div = test_utils.array_to_array_info(interpolation_savepoint.geofac_div().ndarray)
    geofac_grg_x_field, geofac_grg_y_field = interpolation_savepoint.geofac_grg()
    geofac_grg_x = test_utils.array_to_array_info(geofac_grg_x_field.ndarray)
    geofac_grg_y = test_utils.array_to_array_info(geofac_grg_y_field.ndarray)
    geofac_n2s = test_utils.array_to_array_info(interpolation_savepoint.geofac_n2s().ndarray)
    nudgecoeff_e = test_utils.array_to_array_info(interpolation_savepoint.nudgecoeff_e().ndarray)
    # we need the raw Fortran data instead of the postprocessed GT4Py field, see dycore_wrapper.solve_nh_init
    rbf_vec_coeff_v_array = np.squeeze(
        interpolation_savepoint.serializer.read(
            "rbf_vec_coeff_v", interpolation_savepoint.savepoint
        ).astype(float)
    )
    rbf_vec_coeff_v_array = interpolation_savepoint._reduce_to_dim_size(
        rbf_vec_coeff_v_array, [dims.V2EDim, dims.V2EDim, dims.VertexDim]
    )
    rbf_vec_coeff_v = test_utils.array_to_array_info(rbf_vec_coeff_v_array)

    # Diagnostic state parameters
    hdef_ic = test_utils.array_to_array_info(savepoint_diffusion_init.hdef_ic().ndarray)
    div_ic = test_utils.array_to_array_info(savepoint_diffusion_init.div_ic().ndarray)
    dwdx = test_utils.array_to_array_info(savepoint_diffusion_init.dwdx().ndarray)
    dwdy = test_utils.array_to_array_info(savepoint_diffusion_init.dwdy().ndarray)

    # Prognostic state parameters
    w = test_utils.array_to_array_info(savepoint_diffusion_init.w().ndarray)
    vn = test_utils.array_to_array_info(savepoint_diffusion_init.vn().ndarray)
    exner = test_utils.array_to_array_info(savepoint_diffusion_init.exner().ndarray)
    theta_v = test_utils.array_to_array_info(savepoint_diffusion_init.theta_v().ndarray)
    rho = test_utils.array_to_array_info(savepoint_diffusion_init.rho().ndarray)
    dtime = savepoint_diffusion_init.get_metadata("dtime")["dtime"]

    ffi = cffi.FFI()
    # Call diffusion_init
    diffusion_wrapper.diffusion_init(
        ffi=ffi,
        perf_counters=None,
        theta_ref_mc=theta_ref_mc,
        wgtfac_c=wgtfac_c,
        e_bln_c_s=e_bln_c_s,
        geofac_div=geofac_div,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        geofac_n2s=geofac_n2s,
        nudgecoeff_e=nudgecoeff_e,
        rbf_vec_coeff_v=rbf_vec_coeff_v,
        zd_cellidx=zd_cellidx,
        zd_vertidx=zd_vertidx,
        zd_intcoef=zd_intcoef,
        zd_diffcoef=zd_diffcoef,
        ndyn_substeps=cfg.ndyn_substeps,
        diffusion_type=cfg.diffusion_type,
        hdiff_w=cfg.apply_to_vertical_wind,
        hdiff_vn=cfg.apply_to_horizontal_wind,
        hdiff_smag_w=cfg.apply_smag_diff_to_vertical_wind,
        zdiffu_t=cfg.apply_zdiffusion_t,
        type_t_diffu=cfg.type_t_diffu,
        type_vn_diffu=cfg.type_vn_diffu,
        hdiff_efdt_ratio=cfg.hdiff_efdt_ratio,
        hdiff_w_efdt_ratio=cfg.hdiff_w_efdt_ratio,
        smagorinski_scaling_factor=cfg.smagorinski_scaling_factor,
        smagorinski_scaling_factor2=cfg.smagorinski_scaling_factor2,
        smagorinski_scaling_factor3=cfg.smagorinski_scaling_factor3,
        smagorinski_scaling_factor4=cfg.smagorinski_scaling_factor4,
        smagorinski_scaling_height=cfg.smagorinski_scaling_height,
        smagorinski_scaling_height2=cfg.smagorinski_scaling_height2,
        smagorinski_scaling_height3=cfg.smagorinski_scaling_height3,
        smagorinski_scaling_height4=cfg.smagorinski_scaling_height4,
        hdiff_temp=cfg.apply_to_temperature,
        denom_diffu_v=cfg.velocity_boundary_diffusion_denominator,
        nudge_max_coeff=cfg.max_nudging_coefficient,
        itype_sher=cfg.shear_type.value,
        iforcing=cfg.iforcing.value,
        a_hshr=cfg.a_hshr,
        loutshs=cfg.loutshs,
        backend=wrapper_common.BackendIntEnum.DEFAULT,
    )

    # Call diffusion_run
    diffusion_wrapper.diffusion_run(
        ffi=ffi,
        perf_counters=None,
        w=w,
        vn=vn,
        exner=exner,
        theta_v=theta_v,
        rho=rho,
        hdef_ic=hdef_ic,
        div_ic=div_ic,
        dwdx=dwdx,
        dwdy=dwdy,
        dtime=dtime,
        linit=False,
    )

    # Assertions comparing the serialized output with computed output fields
    w_ = savepoint_diffusion_exit.w()
    vn_ = savepoint_diffusion_exit.vn()
    exner_ = savepoint_diffusion_exit.exner()
    theta_v_ = savepoint_diffusion_exit.theta_v()
    hdef_ic_ = savepoint_diffusion_exit.hdef_ic()
    div_ic_ = savepoint_diffusion_exit.div_ic()
    dwdx_ = savepoint_diffusion_exit.dwdx()
    dwdy_ = savepoint_diffusion_exit.dwdy()

    assert testing_test_utils.dallclose(py2fgen.as_array(ffi, w), w_.asnumpy(), atol=1e-12)
    assert testing_test_utils.dallclose(py2fgen.as_array(ffi, vn), vn_.asnumpy(), atol=1e-12)
    assert testing_test_utils.dallclose(py2fgen.as_array(ffi, exner), exner_.asnumpy(), atol=1e-12)
    assert testing_test_utils.dallclose(
        py2fgen.as_array(ffi, theta_v), theta_v_.asnumpy(), atol=1e-12
    )
    assert testing_test_utils.dallclose(
        py2fgen.as_array(ffi, hdef_ic), hdef_ic_.asnumpy(), atol=1e-12
    )
    assert testing_test_utils.dallclose(
        py2fgen.as_array(ffi, div_ic), div_ic_.asnumpy(), atol=1e-12
    )
    assert testing_test_utils.dallclose(py2fgen.as_array(ffi, dwdx), dwdx_.asnumpy(), atol=1e-12)
    assert testing_test_utils.dallclose(py2fgen.as_array(ffi, dwdy), dwdy_.asnumpy(), atol=1e-12)
