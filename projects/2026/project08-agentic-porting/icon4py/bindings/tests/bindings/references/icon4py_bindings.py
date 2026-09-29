import pkgutil
from icon4py.tools.py2fgen import runtime_config

for callable_name in runtime_config.EXTRA_CALLABLES:
    pkgutil.resolve_name(callable_name)()

import logging
from icon4py_bindings import ffi
from icon4py.tools.py2fgen import _runtime, _conversion

logger = logging.getLogger(__name__)
log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
logging.basicConfig(
    level=getattr(logging, runtime_config.LOG_LEVEL),
    format=log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# embedded function imports
from icon4py.bindings.diffusion_wrapper import diffusion_init
from icon4py.bindings.diffusion_wrapper import diffusion_run
from icon4py.bindings.grid_wrapper import grid_init
from icon4py.bindings.dycore_wrapper import solve_nh_init
from icon4py.bindings.dycore_wrapper import solve_nh_run


@ffi.def_extern(error=2)
def diffusion_init_wrapper(
    theta_ref_mc,
    theta_ref_mc_size_0,
    theta_ref_mc_size_1,
    wgtfac_c,
    wgtfac_c_size_0,
    wgtfac_c_size_1,
    e_bln_c_s,
    e_bln_c_s_size_0,
    e_bln_c_s_size_1,
    geofac_div,
    geofac_div_size_0,
    geofac_div_size_1,
    geofac_grg_x,
    geofac_grg_x_size_0,
    geofac_grg_x_size_1,
    geofac_grg_y,
    geofac_grg_y_size_0,
    geofac_grg_y_size_1,
    geofac_n2s,
    geofac_n2s_size_0,
    geofac_n2s_size_1,
    nudgecoeff_e,
    nudgecoeff_e_size_0,
    rbf_vec_coeff_v,
    rbf_vec_coeff_v_size_0,
    rbf_vec_coeff_v_size_1,
    rbf_vec_coeff_v_size_2,
    zd_cellidx,
    zd_cellidx_size_0,
    zd_cellidx_size_1,
    zd_vertidx,
    zd_vertidx_size_0,
    zd_vertidx_size_1,
    zd_intcoef,
    zd_intcoef_size_0,
    zd_intcoef_size_1,
    zd_diffcoef,
    zd_diffcoef_size_0,
    ndyn_substeps,
    diffusion_type,
    hdiff_w,
    hdiff_vn,
    hdiff_smag_w,
    zdiffu_t,
    type_t_diffu,
    type_vn_diffu,
    hdiff_efdt_ratio,
    hdiff_w_efdt_ratio,
    smagorinski_scaling_factor,
    smagorinski_scaling_factor2,
    smagorinski_scaling_factor3,
    smagorinski_scaling_factor4,
    smagorinski_scaling_height,
    smagorinski_scaling_height2,
    smagorinski_scaling_height3,
    smagorinski_scaling_height4,
    hdiff_temp,
    denom_diffu_v,
    nudge_max_coeff,
    itype_sher,
    iforcing,
    a_hshr,
    loutshs,
    backend,
    on_gpu,
):
    with runtime_config.HOOK_BINDINGS_FUNCTION["diffusion_init"]:
        try:
            if __debug__:
                logger.info("Python execution of diffusion_init started.")

            if __debug__:
                if runtime_config.PROFILING:
                    unpack_start_time = _runtime.perf_counter()

            # ArrayInfos

            theta_ref_mc = (
                theta_ref_mc,
                (
                    theta_ref_mc_size_0,
                    theta_ref_mc_size_1,
                ),
                on_gpu,
                False,
            )

            wgtfac_c = (
                wgtfac_c,
                (
                    wgtfac_c_size_0,
                    wgtfac_c_size_1,
                ),
                on_gpu,
                False,
            )

            e_bln_c_s = (
                e_bln_c_s,
                (
                    e_bln_c_s_size_0,
                    e_bln_c_s_size_1,
                ),
                on_gpu,
                False,
            )

            geofac_div = (
                geofac_div,
                (
                    geofac_div_size_0,
                    geofac_div_size_1,
                ),
                on_gpu,
                False,
            )

            geofac_grg_x = (
                geofac_grg_x,
                (
                    geofac_grg_x_size_0,
                    geofac_grg_x_size_1,
                ),
                on_gpu,
                False,
            )

            geofac_grg_y = (
                geofac_grg_y,
                (
                    geofac_grg_y_size_0,
                    geofac_grg_y_size_1,
                ),
                on_gpu,
                False,
            )

            geofac_n2s = (
                geofac_n2s,
                (
                    geofac_n2s_size_0,
                    geofac_n2s_size_1,
                ),
                on_gpu,
                False,
            )

            nudgecoeff_e = (nudgecoeff_e, (nudgecoeff_e_size_0,), on_gpu, False)

            rbf_vec_coeff_v = (
                rbf_vec_coeff_v,
                (
                    rbf_vec_coeff_v_size_0,
                    rbf_vec_coeff_v_size_1,
                    rbf_vec_coeff_v_size_2,
                ),
                on_gpu,
                False,
            )

            zd_cellidx = (
                zd_cellidx,
                (
                    zd_cellidx_size_0,
                    zd_cellidx_size_1,
                ),
                on_gpu,
                True,
            )

            zd_vertidx = (
                zd_vertidx,
                (
                    zd_vertidx_size_0,
                    zd_vertidx_size_1,
                ),
                on_gpu,
                True,
            )

            zd_intcoef = (
                zd_intcoef,
                (
                    zd_intcoef_size_0,
                    zd_intcoef_size_1,
                ),
                on_gpu,
                True,
            )

            zd_diffcoef = (zd_diffcoef, (zd_diffcoef_size_0,), on_gpu, True)

            if __debug__:
                if runtime_config.PROFILING:
                    allocate_end_time = _runtime.perf_counter()
                    logger.info(
                        "diffusion_init constructing `ArrayInfos` time: %s"
                        % str(allocate_end_time - unpack_start_time)
                    )

                    func_start_time = _runtime.perf_counter()

            if __debug__ and runtime_config.PROFILING:
                perf_counters = {}
            else:
                perf_counters = None
            diffusion_init(
                ffi=ffi,
                perf_counters=perf_counters,
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
                ndyn_substeps=ndyn_substeps,
                diffusion_type=diffusion_type,
                hdiff_w=hdiff_w,
                hdiff_vn=hdiff_vn,
                hdiff_smag_w=hdiff_smag_w,
                zdiffu_t=zdiffu_t,
                type_t_diffu=type_t_diffu,
                type_vn_diffu=type_vn_diffu,
                hdiff_efdt_ratio=hdiff_efdt_ratio,
                hdiff_w_efdt_ratio=hdiff_w_efdt_ratio,
                smagorinski_scaling_factor=smagorinski_scaling_factor,
                smagorinski_scaling_factor2=smagorinski_scaling_factor2,
                smagorinski_scaling_factor3=smagorinski_scaling_factor3,
                smagorinski_scaling_factor4=smagorinski_scaling_factor4,
                smagorinski_scaling_height=smagorinski_scaling_height,
                smagorinski_scaling_height2=smagorinski_scaling_height2,
                smagorinski_scaling_height3=smagorinski_scaling_height3,
                smagorinski_scaling_height4=smagorinski_scaling_height4,
                hdiff_temp=hdiff_temp,
                denom_diffu_v=denom_diffu_v,
                nudge_max_coeff=nudge_max_coeff,
                itype_sher=itype_sher,
                iforcing=iforcing,
                a_hshr=a_hshr,
                loutshs=loutshs,
                backend=backend,
            )

            if __debug__:
                if runtime_config.PROFILING:
                    func_end_time = _runtime.perf_counter()
                    logger.info(
                        "diffusion_init convert time: %s"
                        % str(
                            perf_counters["convert_end_time"] - perf_counters["convert_start_time"]
                        )
                    )
                    logger.info(
                        "diffusion_init execution time: %s" % str(func_end_time - func_start_time)
                    )

            if __debug__:
                if logger.isEnabledFor(logging.DEBUG):

                    theta_ref_mc_arr = (
                        _conversion.as_array(ffi, theta_ref_mc)
                        if theta_ref_mc is not None
                        else None
                    )
                    msg = "shape of theta_ref_mc after computation = %s" % str(
                        theta_ref_mc_arr.shape if theta_ref_mc is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "theta_ref_mc after computation: %s" % str(theta_ref_mc_arr)
                        if theta_ref_mc is not None
                        else "None"
                    )
                    logger.debug(msg)

                    wgtfac_c_arr = (
                        _conversion.as_array(ffi, wgtfac_c) if wgtfac_c is not None else None
                    )
                    msg = "shape of wgtfac_c after computation = %s" % str(
                        wgtfac_c_arr.shape if wgtfac_c is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "wgtfac_c after computation: %s" % str(wgtfac_c_arr)
                        if wgtfac_c is not None
                        else "None"
                    )
                    logger.debug(msg)

                    e_bln_c_s_arr = (
                        _conversion.as_array(ffi, e_bln_c_s) if e_bln_c_s is not None else None
                    )
                    msg = "shape of e_bln_c_s after computation = %s" % str(
                        e_bln_c_s_arr.shape if e_bln_c_s is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "e_bln_c_s after computation: %s" % str(e_bln_c_s_arr)
                        if e_bln_c_s is not None
                        else "None"
                    )
                    logger.debug(msg)

                    geofac_div_arr = (
                        _conversion.as_array(ffi, geofac_div) if geofac_div is not None else None
                    )
                    msg = "shape of geofac_div after computation = %s" % str(
                        geofac_div_arr.shape if geofac_div is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "geofac_div after computation: %s" % str(geofac_div_arr)
                        if geofac_div is not None
                        else "None"
                    )
                    logger.debug(msg)

                    geofac_grg_x_arr = (
                        _conversion.as_array(ffi, geofac_grg_x)
                        if geofac_grg_x is not None
                        else None
                    )
                    msg = "shape of geofac_grg_x after computation = %s" % str(
                        geofac_grg_x_arr.shape if geofac_grg_x is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "geofac_grg_x after computation: %s" % str(geofac_grg_x_arr)
                        if geofac_grg_x is not None
                        else "None"
                    )
                    logger.debug(msg)

                    geofac_grg_y_arr = (
                        _conversion.as_array(ffi, geofac_grg_y)
                        if geofac_grg_y is not None
                        else None
                    )
                    msg = "shape of geofac_grg_y after computation = %s" % str(
                        geofac_grg_y_arr.shape if geofac_grg_y is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "geofac_grg_y after computation: %s" % str(geofac_grg_y_arr)
                        if geofac_grg_y is not None
                        else "None"
                    )
                    logger.debug(msg)

                    geofac_n2s_arr = (
                        _conversion.as_array(ffi, geofac_n2s) if geofac_n2s is not None else None
                    )
                    msg = "shape of geofac_n2s after computation = %s" % str(
                        geofac_n2s_arr.shape if geofac_n2s is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "geofac_n2s after computation: %s" % str(geofac_n2s_arr)
                        if geofac_n2s is not None
                        else "None"
                    )
                    logger.debug(msg)

                    nudgecoeff_e_arr = (
                        _conversion.as_array(ffi, nudgecoeff_e)
                        if nudgecoeff_e is not None
                        else None
                    )
                    msg = "shape of nudgecoeff_e after computation = %s" % str(
                        nudgecoeff_e_arr.shape if nudgecoeff_e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "nudgecoeff_e after computation: %s" % str(nudgecoeff_e_arr)
                        if nudgecoeff_e is not None
                        else "None"
                    )
                    logger.debug(msg)

                    rbf_vec_coeff_v_arr = (
                        _conversion.as_array(ffi, rbf_vec_coeff_v)
                        if rbf_vec_coeff_v is not None
                        else None
                    )
                    msg = "shape of rbf_vec_coeff_v after computation = %s" % str(
                        rbf_vec_coeff_v_arr.shape if rbf_vec_coeff_v is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "rbf_vec_coeff_v after computation: %s" % str(rbf_vec_coeff_v_arr)
                        if rbf_vec_coeff_v is not None
                        else "None"
                    )
                    logger.debug(msg)

                    zd_cellidx_arr = (
                        _conversion.as_array(ffi, zd_cellidx) if zd_cellidx is not None else None
                    )
                    msg = "shape of zd_cellidx after computation = %s" % str(
                        zd_cellidx_arr.shape if zd_cellidx is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "zd_cellidx after computation: %s" % str(zd_cellidx_arr)
                        if zd_cellidx is not None
                        else "None"
                    )
                    logger.debug(msg)

                    zd_vertidx_arr = (
                        _conversion.as_array(ffi, zd_vertidx) if zd_vertidx is not None else None
                    )
                    msg = "shape of zd_vertidx after computation = %s" % str(
                        zd_vertidx_arr.shape if zd_vertidx is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "zd_vertidx after computation: %s" % str(zd_vertidx_arr)
                        if zd_vertidx is not None
                        else "None"
                    )
                    logger.debug(msg)

                    zd_intcoef_arr = (
                        _conversion.as_array(ffi, zd_intcoef) if zd_intcoef is not None else None
                    )
                    msg = "shape of zd_intcoef after computation = %s" % str(
                        zd_intcoef_arr.shape if zd_intcoef is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "zd_intcoef after computation: %s" % str(zd_intcoef_arr)
                        if zd_intcoef is not None
                        else "None"
                    )
                    logger.debug(msg)

                    zd_diffcoef_arr = (
                        _conversion.as_array(ffi, zd_diffcoef) if zd_diffcoef is not None else None
                    )
                    msg = "shape of zd_diffcoef after computation = %s" % str(
                        zd_diffcoef_arr.shape if zd_diffcoef is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "zd_diffcoef after computation: %s" % str(zd_diffcoef_arr)
                        if zd_diffcoef is not None
                        else "None"
                    )
                    logger.debug(msg)

            if __debug__:
                logger.info("Python execution of diffusion_init completed.")

        except Exception as e:
            logger.exception(f"A Python error occurred: {e}")
            return 2

    return 1


@ffi.def_extern(error=2)
def diffusion_run_wrapper(
    w,
    w_size_0,
    w_size_1,
    vn,
    vn_size_0,
    vn_size_1,
    exner,
    exner_size_0,
    exner_size_1,
    theta_v,
    theta_v_size_0,
    theta_v_size_1,
    rho,
    rho_size_0,
    rho_size_1,
    hdef_ic,
    hdef_ic_size_0,
    hdef_ic_size_1,
    div_ic,
    div_ic_size_0,
    div_ic_size_1,
    dwdx,
    dwdx_size_0,
    dwdx_size_1,
    dwdy,
    dwdy_size_0,
    dwdy_size_1,
    dtime,
    linit,
    on_gpu,
):
    with runtime_config.HOOK_BINDINGS_FUNCTION["diffusion_run"]:
        try:
            if __debug__:
                logger.info("Python execution of diffusion_run started.")

            if __debug__:
                if runtime_config.PROFILING:
                    unpack_start_time = _runtime.perf_counter()

            # ArrayInfos

            w = (
                w,
                (
                    w_size_0,
                    w_size_1,
                ),
                on_gpu,
                False,
            )

            vn = (
                vn,
                (
                    vn_size_0,
                    vn_size_1,
                ),
                on_gpu,
                False,
            )

            exner = (
                exner,
                (
                    exner_size_0,
                    exner_size_1,
                ),
                on_gpu,
                False,
            )

            theta_v = (
                theta_v,
                (
                    theta_v_size_0,
                    theta_v_size_1,
                ),
                on_gpu,
                False,
            )

            rho = (
                rho,
                (
                    rho_size_0,
                    rho_size_1,
                ),
                on_gpu,
                False,
            )

            hdef_ic = (
                hdef_ic,
                (
                    hdef_ic_size_0,
                    hdef_ic_size_1,
                ),
                on_gpu,
                True,
            )

            div_ic = (
                div_ic,
                (
                    div_ic_size_0,
                    div_ic_size_1,
                ),
                on_gpu,
                True,
            )

            dwdx = (
                dwdx,
                (
                    dwdx_size_0,
                    dwdx_size_1,
                ),
                on_gpu,
                True,
            )

            dwdy = (
                dwdy,
                (
                    dwdy_size_0,
                    dwdy_size_1,
                ),
                on_gpu,
                True,
            )

            if __debug__:
                if runtime_config.PROFILING:
                    allocate_end_time = _runtime.perf_counter()
                    logger.info(
                        "diffusion_run constructing `ArrayInfos` time: %s"
                        % str(allocate_end_time - unpack_start_time)
                    )

                    func_start_time = _runtime.perf_counter()

            if __debug__ and runtime_config.PROFILING:
                perf_counters = {}
            else:
                perf_counters = None
            diffusion_run(
                ffi=ffi,
                perf_counters=perf_counters,
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
                linit=linit,
            )

            if __debug__:
                if runtime_config.PROFILING:
                    func_end_time = _runtime.perf_counter()
                    logger.info(
                        "diffusion_run convert time: %s"
                        % str(
                            perf_counters["convert_end_time"] - perf_counters["convert_start_time"]
                        )
                    )
                    logger.info(
                        "diffusion_run execution time: %s" % str(func_end_time - func_start_time)
                    )

            if __debug__:
                if logger.isEnabledFor(logging.DEBUG):

                    w_arr = _conversion.as_array(ffi, w) if w is not None else None
                    msg = "shape of w after computation = %s" % str(
                        w_arr.shape if w is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "w after computation: %s" % str(w_arr) if w is not None else "None"
                    logger.debug(msg)

                    vn_arr = _conversion.as_array(ffi, vn) if vn is not None else None
                    msg = "shape of vn after computation = %s" % str(
                        vn_arr.shape if vn is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "vn after computation: %s" % str(vn_arr) if vn is not None else "None"
                    logger.debug(msg)

                    exner_arr = _conversion.as_array(ffi, exner) if exner is not None else None
                    msg = "shape of exner after computation = %s" % str(
                        exner_arr.shape if exner is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "exner after computation: %s" % str(exner_arr)
                        if exner is not None
                        else "None"
                    )
                    logger.debug(msg)

                    theta_v_arr = (
                        _conversion.as_array(ffi, theta_v) if theta_v is not None else None
                    )
                    msg = "shape of theta_v after computation = %s" % str(
                        theta_v_arr.shape if theta_v is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "theta_v after computation: %s" % str(theta_v_arr)
                        if theta_v is not None
                        else "None"
                    )
                    logger.debug(msg)

                    rho_arr = _conversion.as_array(ffi, rho) if rho is not None else None
                    msg = "shape of rho after computation = %s" % str(
                        rho_arr.shape if rho is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "rho after computation: %s" % str(rho_arr) if rho is not None else "None"
                    logger.debug(msg)

                    hdef_ic_arr = (
                        _conversion.as_array(ffi, hdef_ic) if hdef_ic is not None else None
                    )
                    msg = "shape of hdef_ic after computation = %s" % str(
                        hdef_ic_arr.shape if hdef_ic is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "hdef_ic after computation: %s" % str(hdef_ic_arr)
                        if hdef_ic is not None
                        else "None"
                    )
                    logger.debug(msg)

                    div_ic_arr = _conversion.as_array(ffi, div_ic) if div_ic is not None else None
                    msg = "shape of div_ic after computation = %s" % str(
                        div_ic_arr.shape if div_ic is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "div_ic after computation: %s" % str(div_ic_arr)
                        if div_ic is not None
                        else "None"
                    )
                    logger.debug(msg)

                    dwdx_arr = _conversion.as_array(ffi, dwdx) if dwdx is not None else None
                    msg = "shape of dwdx after computation = %s" % str(
                        dwdx_arr.shape if dwdx is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "dwdx after computation: %s" % str(dwdx_arr) if dwdx is not None else "None"
                    )
                    logger.debug(msg)

                    dwdy_arr = _conversion.as_array(ffi, dwdy) if dwdy is not None else None
                    msg = "shape of dwdy after computation = %s" % str(
                        dwdy_arr.shape if dwdy is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "dwdy after computation: %s" % str(dwdy_arr) if dwdy is not None else "None"
                    )
                    logger.debug(msg)

            if __debug__:
                logger.info("Python execution of diffusion_run completed.")

        except Exception as e:
            logger.exception(f"A Python error occurred: {e}")
            return 2

    return 1


@ffi.def_extern(error=2)
def grid_init_wrapper(
    cell_starts,
    cell_starts_size_0,
    cell_ends,
    cell_ends_size_0,
    vertex_starts,
    vertex_starts_size_0,
    vertex_ends,
    vertex_ends_size_0,
    edge_starts,
    edge_starts_size_0,
    edge_ends,
    edge_ends_size_0,
    c2e,
    c2e_size_0,
    c2e_size_1,
    e2c,
    e2c_size_0,
    e2c_size_1,
    c2e2c,
    c2e2c_size_0,
    c2e2c_size_1,
    e2c2e,
    e2c2e_size_0,
    e2c2e_size_1,
    e2v,
    e2v_size_0,
    e2v_size_1,
    v2e,
    v2e_size_0,
    v2e_size_1,
    v2c,
    v2c_size_0,
    v2c_size_1,
    e2c2v,
    e2c2v_size_0,
    e2c2v_size_1,
    c2v,
    c2v_size_0,
    c2v_size_1,
    c_owner_mask,
    c_owner_mask_size_0,
    e_owner_mask,
    e_owner_mask_size_0,
    v_owner_mask,
    v_owner_mask_size_0,
    c_glb_index,
    c_glb_index_size_0,
    e_glb_index,
    e_glb_index_size_0,
    v_glb_index,
    v_glb_index_size_0,
    tangent_orientation,
    tangent_orientation_size_0,
    inverse_primal_edge_lengths,
    inverse_primal_edge_lengths_size_0,
    inv_dual_edge_length,
    inv_dual_edge_length_size_0,
    inv_vert_vert_length,
    inv_vert_vert_length_size_0,
    edge_areas,
    edge_areas_size_0,
    f_e,
    f_e_size_0,
    cell_center_lat,
    cell_center_lat_size_0,
    cell_center_lon,
    cell_center_lon_size_0,
    cell_areas,
    cell_areas_size_0,
    primal_normal_vert_x,
    primal_normal_vert_x_size_0,
    primal_normal_vert_x_size_1,
    primal_normal_vert_y,
    primal_normal_vert_y_size_0,
    primal_normal_vert_y_size_1,
    dual_normal_vert_x,
    dual_normal_vert_x_size_0,
    dual_normal_vert_x_size_1,
    dual_normal_vert_y,
    dual_normal_vert_y_size_0,
    dual_normal_vert_y_size_1,
    primal_normal_cell_x,
    primal_normal_cell_x_size_0,
    primal_normal_cell_x_size_1,
    primal_normal_cell_y,
    primal_normal_cell_y_size_0,
    primal_normal_cell_y_size_1,
    dual_normal_cell_x,
    dual_normal_cell_x_size_0,
    dual_normal_cell_x_size_1,
    dual_normal_cell_y,
    dual_normal_cell_y_size_0,
    dual_normal_cell_y_size_1,
    edge_center_lat,
    edge_center_lat_size_0,
    edge_center_lon,
    edge_center_lon_size_0,
    primal_normal_x,
    primal_normal_x_size_0,
    primal_normal_y,
    primal_normal_y_size_0,
    vct_a,
    vct_a_size_0,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    flat_height,
    rayleigh_damping_height,
    mean_cell_area,
    comm_id,
    num_vertices,
    num_cells,
    num_edges,
    vertical_size,
    limited_area,
    backend,
    on_gpu,
):
    with runtime_config.HOOK_BINDINGS_FUNCTION["grid_init"]:
        try:
            if __debug__:
                logger.info("Python execution of grid_init started.")

            if __debug__:
                if runtime_config.PROFILING:
                    unpack_start_time = _runtime.perf_counter()

            # ArrayInfos

            cell_starts = (cell_starts, (cell_starts_size_0,), False, False)

            cell_ends = (cell_ends, (cell_ends_size_0,), False, False)

            vertex_starts = (vertex_starts, (vertex_starts_size_0,), False, False)

            vertex_ends = (vertex_ends, (vertex_ends_size_0,), False, False)

            edge_starts = (edge_starts, (edge_starts_size_0,), False, False)

            edge_ends = (edge_ends, (edge_ends_size_0,), False, False)

            c2e = (
                c2e,
                (
                    c2e_size_0,
                    c2e_size_1,
                ),
                on_gpu,
                False,
            )

            e2c = (
                e2c,
                (
                    e2c_size_0,
                    e2c_size_1,
                ),
                on_gpu,
                False,
            )

            c2e2c = (
                c2e2c,
                (
                    c2e2c_size_0,
                    c2e2c_size_1,
                ),
                on_gpu,
                False,
            )

            e2c2e = (
                e2c2e,
                (
                    e2c2e_size_0,
                    e2c2e_size_1,
                ),
                on_gpu,
                False,
            )

            e2v = (
                e2v,
                (
                    e2v_size_0,
                    e2v_size_1,
                ),
                on_gpu,
                False,
            )

            v2e = (
                v2e,
                (
                    v2e_size_0,
                    v2e_size_1,
                ),
                on_gpu,
                False,
            )

            v2c = (
                v2c,
                (
                    v2c_size_0,
                    v2c_size_1,
                ),
                on_gpu,
                False,
            )

            e2c2v = (
                e2c2v,
                (
                    e2c2v_size_0,
                    e2c2v_size_1,
                ),
                on_gpu,
                False,
            )

            c2v = (
                c2v,
                (
                    c2v_size_0,
                    c2v_size_1,
                ),
                on_gpu,
                False,
            )

            c_owner_mask = (c_owner_mask, (c_owner_mask_size_0,), False, False)

            e_owner_mask = (e_owner_mask, (e_owner_mask_size_0,), False, False)

            v_owner_mask = (v_owner_mask, (v_owner_mask_size_0,), False, False)

            c_glb_index = (c_glb_index, (c_glb_index_size_0,), False, False)

            e_glb_index = (e_glb_index, (e_glb_index_size_0,), False, False)

            v_glb_index = (v_glb_index, (v_glb_index_size_0,), False, False)

            tangent_orientation = (
                tangent_orientation,
                (tangent_orientation_size_0,),
                on_gpu,
                False,
            )

            inverse_primal_edge_lengths = (
                inverse_primal_edge_lengths,
                (inverse_primal_edge_lengths_size_0,),
                on_gpu,
                False,
            )

            inv_dual_edge_length = (
                inv_dual_edge_length,
                (inv_dual_edge_length_size_0,),
                on_gpu,
                False,
            )

            inv_vert_vert_length = (
                inv_vert_vert_length,
                (inv_vert_vert_length_size_0,),
                on_gpu,
                False,
            )

            edge_areas = (edge_areas, (edge_areas_size_0,), on_gpu, False)

            f_e = (f_e, (f_e_size_0,), on_gpu, False)

            cell_center_lat = (cell_center_lat, (cell_center_lat_size_0,), on_gpu, False)

            cell_center_lon = (cell_center_lon, (cell_center_lon_size_0,), on_gpu, False)

            cell_areas = (cell_areas, (cell_areas_size_0,), on_gpu, False)

            primal_normal_vert_x = (
                primal_normal_vert_x,
                (
                    primal_normal_vert_x_size_0,
                    primal_normal_vert_x_size_1,
                ),
                on_gpu,
                False,
            )

            primal_normal_vert_y = (
                primal_normal_vert_y,
                (
                    primal_normal_vert_y_size_0,
                    primal_normal_vert_y_size_1,
                ),
                on_gpu,
                False,
            )

            dual_normal_vert_x = (
                dual_normal_vert_x,
                (
                    dual_normal_vert_x_size_0,
                    dual_normal_vert_x_size_1,
                ),
                on_gpu,
                False,
            )

            dual_normal_vert_y = (
                dual_normal_vert_y,
                (
                    dual_normal_vert_y_size_0,
                    dual_normal_vert_y_size_1,
                ),
                on_gpu,
                False,
            )

            primal_normal_cell_x = (
                primal_normal_cell_x,
                (
                    primal_normal_cell_x_size_0,
                    primal_normal_cell_x_size_1,
                ),
                on_gpu,
                False,
            )

            primal_normal_cell_y = (
                primal_normal_cell_y,
                (
                    primal_normal_cell_y_size_0,
                    primal_normal_cell_y_size_1,
                ),
                on_gpu,
                False,
            )

            dual_normal_cell_x = (
                dual_normal_cell_x,
                (
                    dual_normal_cell_x_size_0,
                    dual_normal_cell_x_size_1,
                ),
                on_gpu,
                False,
            )

            dual_normal_cell_y = (
                dual_normal_cell_y,
                (
                    dual_normal_cell_y_size_0,
                    dual_normal_cell_y_size_1,
                ),
                on_gpu,
                False,
            )

            edge_center_lat = (edge_center_lat, (edge_center_lat_size_0,), on_gpu, False)

            edge_center_lon = (edge_center_lon, (edge_center_lon_size_0,), on_gpu, False)

            primal_normal_x = (primal_normal_x, (primal_normal_x_size_0,), on_gpu, False)

            primal_normal_y = (primal_normal_y, (primal_normal_y_size_0,), on_gpu, False)

            vct_a = (vct_a, (vct_a_size_0,), on_gpu, False)

            if __debug__:
                if runtime_config.PROFILING:
                    allocate_end_time = _runtime.perf_counter()
                    logger.info(
                        "grid_init constructing `ArrayInfos` time: %s"
                        % str(allocate_end_time - unpack_start_time)
                    )

                    func_start_time = _runtime.perf_counter()

            if __debug__ and runtime_config.PROFILING:
                perf_counters = {}
            else:
                perf_counters = None
            grid_init(
                ffi=ffi,
                perf_counters=perf_counters,
                cell_starts=cell_starts,
                cell_ends=cell_ends,
                vertex_starts=vertex_starts,
                vertex_ends=vertex_ends,
                edge_starts=edge_starts,
                edge_ends=edge_ends,
                c2e=c2e,
                e2c=e2c,
                c2e2c=c2e2c,
                e2c2e=e2c2e,
                e2v=e2v,
                v2e=v2e,
                v2c=v2c,
                e2c2v=e2c2v,
                c2v=c2v,
                c_owner_mask=c_owner_mask,
                e_owner_mask=e_owner_mask,
                v_owner_mask=v_owner_mask,
                c_glb_index=c_glb_index,
                e_glb_index=e_glb_index,
                v_glb_index=v_glb_index,
                tangent_orientation=tangent_orientation,
                inverse_primal_edge_lengths=inverse_primal_edge_lengths,
                inv_dual_edge_length=inv_dual_edge_length,
                inv_vert_vert_length=inv_vert_vert_length,
                edge_areas=edge_areas,
                f_e=f_e,
                cell_center_lat=cell_center_lat,
                cell_center_lon=cell_center_lon,
                cell_areas=cell_areas,
                primal_normal_vert_x=primal_normal_vert_x,
                primal_normal_vert_y=primal_normal_vert_y,
                dual_normal_vert_x=dual_normal_vert_x,
                dual_normal_vert_y=dual_normal_vert_y,
                primal_normal_cell_x=primal_normal_cell_x,
                primal_normal_cell_y=primal_normal_cell_y,
                dual_normal_cell_x=dual_normal_cell_x,
                dual_normal_cell_y=dual_normal_cell_y,
                edge_center_lat=edge_center_lat,
                edge_center_lon=edge_center_lon,
                primal_normal_x=primal_normal_x,
                primal_normal_y=primal_normal_y,
                vct_a=vct_a,
                lowest_layer_thickness=lowest_layer_thickness,
                model_top_height=model_top_height,
                stretch_factor=stretch_factor,
                flat_height=flat_height,
                rayleigh_damping_height=rayleigh_damping_height,
                mean_cell_area=mean_cell_area,
                comm_id=comm_id,
                num_vertices=num_vertices,
                num_cells=num_cells,
                num_edges=num_edges,
                vertical_size=vertical_size,
                limited_area=limited_area,
                backend=backend,
            )

            if __debug__:
                if runtime_config.PROFILING:
                    func_end_time = _runtime.perf_counter()
                    logger.info(
                        "grid_init convert time: %s"
                        % str(
                            perf_counters["convert_end_time"] - perf_counters["convert_start_time"]
                        )
                    )
                    logger.info(
                        "grid_init execution time: %s" % str(func_end_time - func_start_time)
                    )

            if __debug__:
                if logger.isEnabledFor(logging.DEBUG):

                    cell_starts_arr = (
                        _conversion.as_array(ffi, cell_starts) if cell_starts is not None else None
                    )
                    msg = "shape of cell_starts after computation = %s" % str(
                        cell_starts_arr.shape if cell_starts is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "cell_starts after computation: %s" % str(cell_starts_arr)
                        if cell_starts is not None
                        else "None"
                    )
                    logger.debug(msg)

                    cell_ends_arr = (
                        _conversion.as_array(ffi, cell_ends) if cell_ends is not None else None
                    )
                    msg = "shape of cell_ends after computation = %s" % str(
                        cell_ends_arr.shape if cell_ends is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "cell_ends after computation: %s" % str(cell_ends_arr)
                        if cell_ends is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vertex_starts_arr = (
                        _conversion.as_array(ffi, vertex_starts)
                        if vertex_starts is not None
                        else None
                    )
                    msg = "shape of vertex_starts after computation = %s" % str(
                        vertex_starts_arr.shape if vertex_starts is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vertex_starts after computation: %s" % str(vertex_starts_arr)
                        if vertex_starts is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vertex_ends_arr = (
                        _conversion.as_array(ffi, vertex_ends) if vertex_ends is not None else None
                    )
                    msg = "shape of vertex_ends after computation = %s" % str(
                        vertex_ends_arr.shape if vertex_ends is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vertex_ends after computation: %s" % str(vertex_ends_arr)
                        if vertex_ends is not None
                        else "None"
                    )
                    logger.debug(msg)

                    edge_starts_arr = (
                        _conversion.as_array(ffi, edge_starts) if edge_starts is not None else None
                    )
                    msg = "shape of edge_starts after computation = %s" % str(
                        edge_starts_arr.shape if edge_starts is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "edge_starts after computation: %s" % str(edge_starts_arr)
                        if edge_starts is not None
                        else "None"
                    )
                    logger.debug(msg)

                    edge_ends_arr = (
                        _conversion.as_array(ffi, edge_ends) if edge_ends is not None else None
                    )
                    msg = "shape of edge_ends after computation = %s" % str(
                        edge_ends_arr.shape if edge_ends is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "edge_ends after computation: %s" % str(edge_ends_arr)
                        if edge_ends is not None
                        else "None"
                    )
                    logger.debug(msg)

                    c2e_arr = _conversion.as_array(ffi, c2e) if c2e is not None else None
                    msg = "shape of c2e after computation = %s" % str(
                        c2e_arr.shape if c2e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "c2e after computation: %s" % str(c2e_arr) if c2e is not None else "None"
                    logger.debug(msg)

                    e2c_arr = _conversion.as_array(ffi, e2c) if e2c is not None else None
                    msg = "shape of e2c after computation = %s" % str(
                        e2c_arr.shape if e2c is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "e2c after computation: %s" % str(e2c_arr) if e2c is not None else "None"
                    logger.debug(msg)

                    c2e2c_arr = _conversion.as_array(ffi, c2e2c) if c2e2c is not None else None
                    msg = "shape of c2e2c after computation = %s" % str(
                        c2e2c_arr.shape if c2e2c is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "c2e2c after computation: %s" % str(c2e2c_arr)
                        if c2e2c is not None
                        else "None"
                    )
                    logger.debug(msg)

                    e2c2e_arr = _conversion.as_array(ffi, e2c2e) if e2c2e is not None else None
                    msg = "shape of e2c2e after computation = %s" % str(
                        e2c2e_arr.shape if e2c2e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "e2c2e after computation: %s" % str(e2c2e_arr)
                        if e2c2e is not None
                        else "None"
                    )
                    logger.debug(msg)

                    e2v_arr = _conversion.as_array(ffi, e2v) if e2v is not None else None
                    msg = "shape of e2v after computation = %s" % str(
                        e2v_arr.shape if e2v is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "e2v after computation: %s" % str(e2v_arr) if e2v is not None else "None"
                    logger.debug(msg)

                    v2e_arr = _conversion.as_array(ffi, v2e) if v2e is not None else None
                    msg = "shape of v2e after computation = %s" % str(
                        v2e_arr.shape if v2e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "v2e after computation: %s" % str(v2e_arr) if v2e is not None else "None"
                    logger.debug(msg)

                    v2c_arr = _conversion.as_array(ffi, v2c) if v2c is not None else None
                    msg = "shape of v2c after computation = %s" % str(
                        v2c_arr.shape if v2c is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "v2c after computation: %s" % str(v2c_arr) if v2c is not None else "None"
                    logger.debug(msg)

                    e2c2v_arr = _conversion.as_array(ffi, e2c2v) if e2c2v is not None else None
                    msg = "shape of e2c2v after computation = %s" % str(
                        e2c2v_arr.shape if e2c2v is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "e2c2v after computation: %s" % str(e2c2v_arr)
                        if e2c2v is not None
                        else "None"
                    )
                    logger.debug(msg)

                    c2v_arr = _conversion.as_array(ffi, c2v) if c2v is not None else None
                    msg = "shape of c2v after computation = %s" % str(
                        c2v_arr.shape if c2v is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "c2v after computation: %s" % str(c2v_arr) if c2v is not None else "None"
                    logger.debug(msg)

                    c_owner_mask_arr = (
                        _conversion.as_array(ffi, c_owner_mask)
                        if c_owner_mask is not None
                        else None
                    )
                    msg = "shape of c_owner_mask after computation = %s" % str(
                        c_owner_mask_arr.shape if c_owner_mask is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "c_owner_mask after computation: %s" % str(c_owner_mask_arr)
                        if c_owner_mask is not None
                        else "None"
                    )
                    logger.debug(msg)

                    e_owner_mask_arr = (
                        _conversion.as_array(ffi, e_owner_mask)
                        if e_owner_mask is not None
                        else None
                    )
                    msg = "shape of e_owner_mask after computation = %s" % str(
                        e_owner_mask_arr.shape if e_owner_mask is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "e_owner_mask after computation: %s" % str(e_owner_mask_arr)
                        if e_owner_mask is not None
                        else "None"
                    )
                    logger.debug(msg)

                    v_owner_mask_arr = (
                        _conversion.as_array(ffi, v_owner_mask)
                        if v_owner_mask is not None
                        else None
                    )
                    msg = "shape of v_owner_mask after computation = %s" % str(
                        v_owner_mask_arr.shape if v_owner_mask is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "v_owner_mask after computation: %s" % str(v_owner_mask_arr)
                        if v_owner_mask is not None
                        else "None"
                    )
                    logger.debug(msg)

                    c_glb_index_arr = (
                        _conversion.as_array(ffi, c_glb_index) if c_glb_index is not None else None
                    )
                    msg = "shape of c_glb_index after computation = %s" % str(
                        c_glb_index_arr.shape if c_glb_index is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "c_glb_index after computation: %s" % str(c_glb_index_arr)
                        if c_glb_index is not None
                        else "None"
                    )
                    logger.debug(msg)

                    e_glb_index_arr = (
                        _conversion.as_array(ffi, e_glb_index) if e_glb_index is not None else None
                    )
                    msg = "shape of e_glb_index after computation = %s" % str(
                        e_glb_index_arr.shape if e_glb_index is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "e_glb_index after computation: %s" % str(e_glb_index_arr)
                        if e_glb_index is not None
                        else "None"
                    )
                    logger.debug(msg)

                    v_glb_index_arr = (
                        _conversion.as_array(ffi, v_glb_index) if v_glb_index is not None else None
                    )
                    msg = "shape of v_glb_index after computation = %s" % str(
                        v_glb_index_arr.shape if v_glb_index is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "v_glb_index after computation: %s" % str(v_glb_index_arr)
                        if v_glb_index is not None
                        else "None"
                    )
                    logger.debug(msg)

                    tangent_orientation_arr = (
                        _conversion.as_array(ffi, tangent_orientation)
                        if tangent_orientation is not None
                        else None
                    )
                    msg = "shape of tangent_orientation after computation = %s" % str(
                        tangent_orientation_arr.shape if tangent_orientation is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "tangent_orientation after computation: %s" % str(tangent_orientation_arr)
                        if tangent_orientation is not None
                        else "None"
                    )
                    logger.debug(msg)

                    inverse_primal_edge_lengths_arr = (
                        _conversion.as_array(ffi, inverse_primal_edge_lengths)
                        if inverse_primal_edge_lengths is not None
                        else None
                    )
                    msg = "shape of inverse_primal_edge_lengths after computation = %s" % str(
                        inverse_primal_edge_lengths_arr.shape
                        if inverse_primal_edge_lengths is not None
                        else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "inverse_primal_edge_lengths after computation: %s"
                        % str(inverse_primal_edge_lengths_arr)
                        if inverse_primal_edge_lengths is not None
                        else "None"
                    )
                    logger.debug(msg)

                    inv_dual_edge_length_arr = (
                        _conversion.as_array(ffi, inv_dual_edge_length)
                        if inv_dual_edge_length is not None
                        else None
                    )
                    msg = "shape of inv_dual_edge_length after computation = %s" % str(
                        inv_dual_edge_length_arr.shape
                        if inv_dual_edge_length is not None
                        else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "inv_dual_edge_length after computation: %s" % str(inv_dual_edge_length_arr)
                        if inv_dual_edge_length is not None
                        else "None"
                    )
                    logger.debug(msg)

                    inv_vert_vert_length_arr = (
                        _conversion.as_array(ffi, inv_vert_vert_length)
                        if inv_vert_vert_length is not None
                        else None
                    )
                    msg = "shape of inv_vert_vert_length after computation = %s" % str(
                        inv_vert_vert_length_arr.shape
                        if inv_vert_vert_length is not None
                        else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "inv_vert_vert_length after computation: %s" % str(inv_vert_vert_length_arr)
                        if inv_vert_vert_length is not None
                        else "None"
                    )
                    logger.debug(msg)

                    edge_areas_arr = (
                        _conversion.as_array(ffi, edge_areas) if edge_areas is not None else None
                    )
                    msg = "shape of edge_areas after computation = %s" % str(
                        edge_areas_arr.shape if edge_areas is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "edge_areas after computation: %s" % str(edge_areas_arr)
                        if edge_areas is not None
                        else "None"
                    )
                    logger.debug(msg)

                    f_e_arr = _conversion.as_array(ffi, f_e) if f_e is not None else None
                    msg = "shape of f_e after computation = %s" % str(
                        f_e_arr.shape if f_e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "f_e after computation: %s" % str(f_e_arr) if f_e is not None else "None"
                    logger.debug(msg)

                    cell_center_lat_arr = (
                        _conversion.as_array(ffi, cell_center_lat)
                        if cell_center_lat is not None
                        else None
                    )
                    msg = "shape of cell_center_lat after computation = %s" % str(
                        cell_center_lat_arr.shape if cell_center_lat is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "cell_center_lat after computation: %s" % str(cell_center_lat_arr)
                        if cell_center_lat is not None
                        else "None"
                    )
                    logger.debug(msg)

                    cell_center_lon_arr = (
                        _conversion.as_array(ffi, cell_center_lon)
                        if cell_center_lon is not None
                        else None
                    )
                    msg = "shape of cell_center_lon after computation = %s" % str(
                        cell_center_lon_arr.shape if cell_center_lon is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "cell_center_lon after computation: %s" % str(cell_center_lon_arr)
                        if cell_center_lon is not None
                        else "None"
                    )
                    logger.debug(msg)

                    cell_areas_arr = (
                        _conversion.as_array(ffi, cell_areas) if cell_areas is not None else None
                    )
                    msg = "shape of cell_areas after computation = %s" % str(
                        cell_areas_arr.shape if cell_areas is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "cell_areas after computation: %s" % str(cell_areas_arr)
                        if cell_areas is not None
                        else "None"
                    )
                    logger.debug(msg)

                    primal_normal_vert_x_arr = (
                        _conversion.as_array(ffi, primal_normal_vert_x)
                        if primal_normal_vert_x is not None
                        else None
                    )
                    msg = "shape of primal_normal_vert_x after computation = %s" % str(
                        primal_normal_vert_x_arr.shape
                        if primal_normal_vert_x is not None
                        else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "primal_normal_vert_x after computation: %s" % str(primal_normal_vert_x_arr)
                        if primal_normal_vert_x is not None
                        else "None"
                    )
                    logger.debug(msg)

                    primal_normal_vert_y_arr = (
                        _conversion.as_array(ffi, primal_normal_vert_y)
                        if primal_normal_vert_y is not None
                        else None
                    )
                    msg = "shape of primal_normal_vert_y after computation = %s" % str(
                        primal_normal_vert_y_arr.shape
                        if primal_normal_vert_y is not None
                        else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "primal_normal_vert_y after computation: %s" % str(primal_normal_vert_y_arr)
                        if primal_normal_vert_y is not None
                        else "None"
                    )
                    logger.debug(msg)

                    dual_normal_vert_x_arr = (
                        _conversion.as_array(ffi, dual_normal_vert_x)
                        if dual_normal_vert_x is not None
                        else None
                    )
                    msg = "shape of dual_normal_vert_x after computation = %s" % str(
                        dual_normal_vert_x_arr.shape if dual_normal_vert_x is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "dual_normal_vert_x after computation: %s" % str(dual_normal_vert_x_arr)
                        if dual_normal_vert_x is not None
                        else "None"
                    )
                    logger.debug(msg)

                    dual_normal_vert_y_arr = (
                        _conversion.as_array(ffi, dual_normal_vert_y)
                        if dual_normal_vert_y is not None
                        else None
                    )
                    msg = "shape of dual_normal_vert_y after computation = %s" % str(
                        dual_normal_vert_y_arr.shape if dual_normal_vert_y is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "dual_normal_vert_y after computation: %s" % str(dual_normal_vert_y_arr)
                        if dual_normal_vert_y is not None
                        else "None"
                    )
                    logger.debug(msg)

                    primal_normal_cell_x_arr = (
                        _conversion.as_array(ffi, primal_normal_cell_x)
                        if primal_normal_cell_x is not None
                        else None
                    )
                    msg = "shape of primal_normal_cell_x after computation = %s" % str(
                        primal_normal_cell_x_arr.shape
                        if primal_normal_cell_x is not None
                        else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "primal_normal_cell_x after computation: %s" % str(primal_normal_cell_x_arr)
                        if primal_normal_cell_x is not None
                        else "None"
                    )
                    logger.debug(msg)

                    primal_normal_cell_y_arr = (
                        _conversion.as_array(ffi, primal_normal_cell_y)
                        if primal_normal_cell_y is not None
                        else None
                    )
                    msg = "shape of primal_normal_cell_y after computation = %s" % str(
                        primal_normal_cell_y_arr.shape
                        if primal_normal_cell_y is not None
                        else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "primal_normal_cell_y after computation: %s" % str(primal_normal_cell_y_arr)
                        if primal_normal_cell_y is not None
                        else "None"
                    )
                    logger.debug(msg)

                    dual_normal_cell_x_arr = (
                        _conversion.as_array(ffi, dual_normal_cell_x)
                        if dual_normal_cell_x is not None
                        else None
                    )
                    msg = "shape of dual_normal_cell_x after computation = %s" % str(
                        dual_normal_cell_x_arr.shape if dual_normal_cell_x is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "dual_normal_cell_x after computation: %s" % str(dual_normal_cell_x_arr)
                        if dual_normal_cell_x is not None
                        else "None"
                    )
                    logger.debug(msg)

                    dual_normal_cell_y_arr = (
                        _conversion.as_array(ffi, dual_normal_cell_y)
                        if dual_normal_cell_y is not None
                        else None
                    )
                    msg = "shape of dual_normal_cell_y after computation = %s" % str(
                        dual_normal_cell_y_arr.shape if dual_normal_cell_y is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "dual_normal_cell_y after computation: %s" % str(dual_normal_cell_y_arr)
                        if dual_normal_cell_y is not None
                        else "None"
                    )
                    logger.debug(msg)

                    edge_center_lat_arr = (
                        _conversion.as_array(ffi, edge_center_lat)
                        if edge_center_lat is not None
                        else None
                    )
                    msg = "shape of edge_center_lat after computation = %s" % str(
                        edge_center_lat_arr.shape if edge_center_lat is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "edge_center_lat after computation: %s" % str(edge_center_lat_arr)
                        if edge_center_lat is not None
                        else "None"
                    )
                    logger.debug(msg)

                    edge_center_lon_arr = (
                        _conversion.as_array(ffi, edge_center_lon)
                        if edge_center_lon is not None
                        else None
                    )
                    msg = "shape of edge_center_lon after computation = %s" % str(
                        edge_center_lon_arr.shape if edge_center_lon is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "edge_center_lon after computation: %s" % str(edge_center_lon_arr)
                        if edge_center_lon is not None
                        else "None"
                    )
                    logger.debug(msg)

                    primal_normal_x_arr = (
                        _conversion.as_array(ffi, primal_normal_x)
                        if primal_normal_x is not None
                        else None
                    )
                    msg = "shape of primal_normal_x after computation = %s" % str(
                        primal_normal_x_arr.shape if primal_normal_x is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "primal_normal_x after computation: %s" % str(primal_normal_x_arr)
                        if primal_normal_x is not None
                        else "None"
                    )
                    logger.debug(msg)

                    primal_normal_y_arr = (
                        _conversion.as_array(ffi, primal_normal_y)
                        if primal_normal_y is not None
                        else None
                    )
                    msg = "shape of primal_normal_y after computation = %s" % str(
                        primal_normal_y_arr.shape if primal_normal_y is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "primal_normal_y after computation: %s" % str(primal_normal_y_arr)
                        if primal_normal_y is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vct_a_arr = _conversion.as_array(ffi, vct_a) if vct_a is not None else None
                    msg = "shape of vct_a after computation = %s" % str(
                        vct_a_arr.shape if vct_a is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vct_a after computation: %s" % str(vct_a_arr)
                        if vct_a is not None
                        else "None"
                    )
                    logger.debug(msg)

            if __debug__:
                logger.info("Python execution of grid_init completed.")

        except Exception as e:
            logger.exception(f"A Python error occurred: {e}")
            return 2

    return 1


@ffi.def_extern(error=2)
def solve_nh_init_wrapper(
    c_lin_e,
    c_lin_e_size_0,
    c_lin_e_size_1,
    c_intp,
    c_intp_size_0,
    c_intp_size_1,
    e_flx_avg,
    e_flx_avg_size_0,
    e_flx_avg_size_1,
    geofac_grdiv,
    geofac_grdiv_size_0,
    geofac_grdiv_size_1,
    geofac_rot,
    geofac_rot_size_0,
    geofac_rot_size_1,
    pos_on_tplane_e_1,
    pos_on_tplane_e_1_size_0,
    pos_on_tplane_e_1_size_1,
    pos_on_tplane_e_2,
    pos_on_tplane_e_2_size_0,
    pos_on_tplane_e_2_size_1,
    rbf_vec_coeff_e,
    rbf_vec_coeff_e_size_0,
    rbf_vec_coeff_e_size_1,
    e_bln_c_s,
    e_bln_c_s_size_0,
    e_bln_c_s_size_1,
    rbf_vec_coeff_v,
    rbf_vec_coeff_v_size_0,
    rbf_vec_coeff_v_size_1,
    rbf_vec_coeff_v_size_2,
    geofac_div,
    geofac_div_size_0,
    geofac_div_size_1,
    geofac_n2s,
    geofac_n2s_size_0,
    geofac_n2s_size_1,
    geofac_grg_x,
    geofac_grg_x_size_0,
    geofac_grg_x_size_1,
    geofac_grg_y,
    geofac_grg_y_size_0,
    geofac_grg_y_size_1,
    nudgecoeff_e,
    nudgecoeff_e_size_0,
    mask_prog_halo_c,
    mask_prog_halo_c_size_0,
    rayleigh_w,
    rayleigh_w_size_0,
    exner_exfac,
    exner_exfac_size_0,
    exner_exfac_size_1,
    exner_ref_mc,
    exner_ref_mc_size_0,
    exner_ref_mc_size_1,
    wgtfac_c,
    wgtfac_c_size_0,
    wgtfac_c_size_1,
    wgtfacq_c,
    wgtfacq_c_size_0,
    wgtfacq_c_size_1,
    inv_ddqz_z_full,
    inv_ddqz_z_full_size_0,
    inv_ddqz_z_full_size_1,
    rho_ref_mc,
    rho_ref_mc_size_0,
    rho_ref_mc_size_1,
    theta_ref_mc,
    theta_ref_mc_size_0,
    theta_ref_mc_size_1,
    vwind_expl_wgt,
    vwind_expl_wgt_size_0,
    d_exner_dz_ref_ic,
    d_exner_dz_ref_ic_size_0,
    d_exner_dz_ref_ic_size_1,
    ddqz_z_half,
    ddqz_z_half_size_0,
    ddqz_z_half_size_1,
    theta_ref_ic,
    theta_ref_ic_size_0,
    theta_ref_ic_size_1,
    d2dexdz2_fac1_mc,
    d2dexdz2_fac1_mc_size_0,
    d2dexdz2_fac1_mc_size_1,
    d2dexdz2_fac2_mc,
    d2dexdz2_fac2_mc_size_0,
    d2dexdz2_fac2_mc_size_1,
    rho_ref_me,
    rho_ref_me_size_0,
    rho_ref_me_size_1,
    theta_ref_me,
    theta_ref_me_size_0,
    theta_ref_me_size_1,
    ddxn_z_full,
    ddxn_z_full_size_0,
    ddxn_z_full_size_1,
    zdiff_gradp,
    zdiff_gradp_size_0,
    zdiff_gradp_size_1,
    zdiff_gradp_size_2,
    vertidx_gradp,
    vertidx_gradp_size_0,
    vertidx_gradp_size_1,
    vertidx_gradp_size_2,
    pg_edgeidx,
    pg_edgeidx_size_0,
    pg_vertidx,
    pg_vertidx_size_0,
    pg_exdist,
    pg_exdist_size_0,
    ddqz_z_full_e,
    ddqz_z_full_e_size_0,
    ddqz_z_full_e_size_1,
    ddxt_z_full,
    ddxt_z_full_size_0,
    ddxt_z_full_size_1,
    wgtfac_e,
    wgtfac_e_size_0,
    wgtfac_e_size_1,
    wgtfacq_e,
    wgtfacq_e_size_0,
    wgtfacq_e_size_1,
    vwind_impl_wgt,
    vwind_impl_wgt_size_0,
    hmask_dd3d,
    hmask_dd3d_size_0,
    scalfac_dd3d,
    scalfac_dd3d_size_0,
    coeff1_dwdz,
    coeff1_dwdz_size_0,
    coeff1_dwdz_size_1,
    coeff2_dwdz,
    coeff2_dwdz_size_0,
    coeff2_dwdz_size_1,
    coeff_gradekin,
    coeff_gradekin_size_0,
    coeff_gradekin_size_1,
    c_owner_mask,
    c_owner_mask_size_0,
    itime_scheme,
    iadv_rhotheta,
    igradp_method,
    rayleigh_type,
    divdamp_order,
    divdamp_type,
    l_vert_nested,
    ldeepatmo,
    iau_init,
    extra_diffu,
    rhotheta_offctr,
    veladv_offctr,
    nudge_max_coeff,
    divdamp_fac,
    divdamp_fac2,
    divdamp_fac3,
    divdamp_fac4,
    divdamp_z,
    divdamp_z2,
    divdamp_z3,
    divdamp_z4,
    nflat_gradp,
    backend,
    on_gpu,
):
    with runtime_config.HOOK_BINDINGS_FUNCTION["solve_nh_init"]:
        try:
            if __debug__:
                logger.info("Python execution of solve_nh_init started.")

            if __debug__:
                if runtime_config.PROFILING:
                    unpack_start_time = _runtime.perf_counter()

            # ArrayInfos

            c_lin_e = (
                c_lin_e,
                (
                    c_lin_e_size_0,
                    c_lin_e_size_1,
                ),
                on_gpu,
                False,
            )

            c_intp = (
                c_intp,
                (
                    c_intp_size_0,
                    c_intp_size_1,
                ),
                on_gpu,
                False,
            )

            e_flx_avg = (
                e_flx_avg,
                (
                    e_flx_avg_size_0,
                    e_flx_avg_size_1,
                ),
                on_gpu,
                False,
            )

            geofac_grdiv = (
                geofac_grdiv,
                (
                    geofac_grdiv_size_0,
                    geofac_grdiv_size_1,
                ),
                on_gpu,
                False,
            )

            geofac_rot = (
                geofac_rot,
                (
                    geofac_rot_size_0,
                    geofac_rot_size_1,
                ),
                on_gpu,
                False,
            )

            pos_on_tplane_e_1 = (
                pos_on_tplane_e_1,
                (
                    pos_on_tplane_e_1_size_0,
                    pos_on_tplane_e_1_size_1,
                ),
                on_gpu,
                False,
            )

            pos_on_tplane_e_2 = (
                pos_on_tplane_e_2,
                (
                    pos_on_tplane_e_2_size_0,
                    pos_on_tplane_e_2_size_1,
                ),
                on_gpu,
                False,
            )

            rbf_vec_coeff_e = (
                rbf_vec_coeff_e,
                (
                    rbf_vec_coeff_e_size_0,
                    rbf_vec_coeff_e_size_1,
                ),
                on_gpu,
                False,
            )

            e_bln_c_s = (
                e_bln_c_s,
                (
                    e_bln_c_s_size_0,
                    e_bln_c_s_size_1,
                ),
                on_gpu,
                False,
            )

            rbf_vec_coeff_v = (
                rbf_vec_coeff_v,
                (
                    rbf_vec_coeff_v_size_0,
                    rbf_vec_coeff_v_size_1,
                    rbf_vec_coeff_v_size_2,
                ),
                on_gpu,
                False,
            )

            geofac_div = (
                geofac_div,
                (
                    geofac_div_size_0,
                    geofac_div_size_1,
                ),
                on_gpu,
                False,
            )

            geofac_n2s = (
                geofac_n2s,
                (
                    geofac_n2s_size_0,
                    geofac_n2s_size_1,
                ),
                on_gpu,
                False,
            )

            geofac_grg_x = (
                geofac_grg_x,
                (
                    geofac_grg_x_size_0,
                    geofac_grg_x_size_1,
                ),
                on_gpu,
                False,
            )

            geofac_grg_y = (
                geofac_grg_y,
                (
                    geofac_grg_y_size_0,
                    geofac_grg_y_size_1,
                ),
                on_gpu,
                False,
            )

            nudgecoeff_e = (nudgecoeff_e, (nudgecoeff_e_size_0,), on_gpu, False)

            mask_prog_halo_c = (mask_prog_halo_c, (mask_prog_halo_c_size_0,), on_gpu, False)

            rayleigh_w = (rayleigh_w, (rayleigh_w_size_0,), on_gpu, False)

            exner_exfac = (
                exner_exfac,
                (
                    exner_exfac_size_0,
                    exner_exfac_size_1,
                ),
                on_gpu,
                False,
            )

            exner_ref_mc = (
                exner_ref_mc,
                (
                    exner_ref_mc_size_0,
                    exner_ref_mc_size_1,
                ),
                on_gpu,
                False,
            )

            wgtfac_c = (
                wgtfac_c,
                (
                    wgtfac_c_size_0,
                    wgtfac_c_size_1,
                ),
                on_gpu,
                False,
            )

            wgtfacq_c = (
                wgtfacq_c,
                (
                    wgtfacq_c_size_0,
                    wgtfacq_c_size_1,
                ),
                on_gpu,
                False,
            )

            inv_ddqz_z_full = (
                inv_ddqz_z_full,
                (
                    inv_ddqz_z_full_size_0,
                    inv_ddqz_z_full_size_1,
                ),
                on_gpu,
                False,
            )

            rho_ref_mc = (
                rho_ref_mc,
                (
                    rho_ref_mc_size_0,
                    rho_ref_mc_size_1,
                ),
                on_gpu,
                False,
            )

            theta_ref_mc = (
                theta_ref_mc,
                (
                    theta_ref_mc_size_0,
                    theta_ref_mc_size_1,
                ),
                on_gpu,
                False,
            )

            vwind_expl_wgt = (vwind_expl_wgt, (vwind_expl_wgt_size_0,), on_gpu, False)

            d_exner_dz_ref_ic = (
                d_exner_dz_ref_ic,
                (
                    d_exner_dz_ref_ic_size_0,
                    d_exner_dz_ref_ic_size_1,
                ),
                on_gpu,
                False,
            )

            ddqz_z_half = (
                ddqz_z_half,
                (
                    ddqz_z_half_size_0,
                    ddqz_z_half_size_1,
                ),
                on_gpu,
                False,
            )

            theta_ref_ic = (
                theta_ref_ic,
                (
                    theta_ref_ic_size_0,
                    theta_ref_ic_size_1,
                ),
                on_gpu,
                False,
            )

            d2dexdz2_fac1_mc = (
                d2dexdz2_fac1_mc,
                (
                    d2dexdz2_fac1_mc_size_0,
                    d2dexdz2_fac1_mc_size_1,
                ),
                on_gpu,
                False,
            )

            d2dexdz2_fac2_mc = (
                d2dexdz2_fac2_mc,
                (
                    d2dexdz2_fac2_mc_size_0,
                    d2dexdz2_fac2_mc_size_1,
                ),
                on_gpu,
                False,
            )

            rho_ref_me = (
                rho_ref_me,
                (
                    rho_ref_me_size_0,
                    rho_ref_me_size_1,
                ),
                on_gpu,
                False,
            )

            theta_ref_me = (
                theta_ref_me,
                (
                    theta_ref_me_size_0,
                    theta_ref_me_size_1,
                ),
                on_gpu,
                False,
            )

            ddxn_z_full = (
                ddxn_z_full,
                (
                    ddxn_z_full_size_0,
                    ddxn_z_full_size_1,
                ),
                on_gpu,
                False,
            )

            zdiff_gradp = (
                zdiff_gradp,
                (
                    zdiff_gradp_size_0,
                    zdiff_gradp_size_1,
                    zdiff_gradp_size_2,
                ),
                on_gpu,
                False,
            )

            vertidx_gradp = (
                vertidx_gradp,
                (
                    vertidx_gradp_size_0,
                    vertidx_gradp_size_1,
                    vertidx_gradp_size_2,
                ),
                on_gpu,
                False,
            )

            pg_edgeidx = (pg_edgeidx, (pg_edgeidx_size_0,), on_gpu, True)

            pg_vertidx = (pg_vertidx, (pg_vertidx_size_0,), on_gpu, True)

            pg_exdist = (pg_exdist, (pg_exdist_size_0,), on_gpu, True)

            ddqz_z_full_e = (
                ddqz_z_full_e,
                (
                    ddqz_z_full_e_size_0,
                    ddqz_z_full_e_size_1,
                ),
                on_gpu,
                False,
            )

            ddxt_z_full = (
                ddxt_z_full,
                (
                    ddxt_z_full_size_0,
                    ddxt_z_full_size_1,
                ),
                on_gpu,
                False,
            )

            wgtfac_e = (
                wgtfac_e,
                (
                    wgtfac_e_size_0,
                    wgtfac_e_size_1,
                ),
                on_gpu,
                False,
            )

            wgtfacq_e = (
                wgtfacq_e,
                (
                    wgtfacq_e_size_0,
                    wgtfacq_e_size_1,
                ),
                on_gpu,
                False,
            )

            vwind_impl_wgt = (vwind_impl_wgt, (vwind_impl_wgt_size_0,), on_gpu, False)

            hmask_dd3d = (hmask_dd3d, (hmask_dd3d_size_0,), on_gpu, False)

            scalfac_dd3d = (scalfac_dd3d, (scalfac_dd3d_size_0,), on_gpu, False)

            coeff1_dwdz = (
                coeff1_dwdz,
                (
                    coeff1_dwdz_size_0,
                    coeff1_dwdz_size_1,
                ),
                on_gpu,
                False,
            )

            coeff2_dwdz = (
                coeff2_dwdz,
                (
                    coeff2_dwdz_size_0,
                    coeff2_dwdz_size_1,
                ),
                on_gpu,
                False,
            )

            coeff_gradekin = (
                coeff_gradekin,
                (
                    coeff_gradekin_size_0,
                    coeff_gradekin_size_1,
                ),
                on_gpu,
                False,
            )

            c_owner_mask = (c_owner_mask, (c_owner_mask_size_0,), on_gpu, False)

            if __debug__:
                if runtime_config.PROFILING:
                    allocate_end_time = _runtime.perf_counter()
                    logger.info(
                        "solve_nh_init constructing `ArrayInfos` time: %s"
                        % str(allocate_end_time - unpack_start_time)
                    )

                    func_start_time = _runtime.perf_counter()

            if __debug__ and runtime_config.PROFILING:
                perf_counters = {}
            else:
                perf_counters = None
            solve_nh_init(
                ffi=ffi,
                perf_counters=perf_counters,
                c_lin_e=c_lin_e,
                c_intp=c_intp,
                e_flx_avg=e_flx_avg,
                geofac_grdiv=geofac_grdiv,
                geofac_rot=geofac_rot,
                pos_on_tplane_e_1=pos_on_tplane_e_1,
                pos_on_tplane_e_2=pos_on_tplane_e_2,
                rbf_vec_coeff_e=rbf_vec_coeff_e,
                e_bln_c_s=e_bln_c_s,
                rbf_vec_coeff_v=rbf_vec_coeff_v,
                geofac_div=geofac_div,
                geofac_n2s=geofac_n2s,
                geofac_grg_x=geofac_grg_x,
                geofac_grg_y=geofac_grg_y,
                nudgecoeff_e=nudgecoeff_e,
                mask_prog_halo_c=mask_prog_halo_c,
                rayleigh_w=rayleigh_w,
                exner_exfac=exner_exfac,
                exner_ref_mc=exner_ref_mc,
                wgtfac_c=wgtfac_c,
                wgtfacq_c=wgtfacq_c,
                inv_ddqz_z_full=inv_ddqz_z_full,
                rho_ref_mc=rho_ref_mc,
                theta_ref_mc=theta_ref_mc,
                vwind_expl_wgt=vwind_expl_wgt,
                d_exner_dz_ref_ic=d_exner_dz_ref_ic,
                ddqz_z_half=ddqz_z_half,
                theta_ref_ic=theta_ref_ic,
                d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
                d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
                rho_ref_me=rho_ref_me,
                theta_ref_me=theta_ref_me,
                ddxn_z_full=ddxn_z_full,
                zdiff_gradp=zdiff_gradp,
                vertidx_gradp=vertidx_gradp,
                pg_edgeidx=pg_edgeidx,
                pg_vertidx=pg_vertidx,
                pg_exdist=pg_exdist,
                ddqz_z_full_e=ddqz_z_full_e,
                ddxt_z_full=ddxt_z_full,
                wgtfac_e=wgtfac_e,
                wgtfacq_e=wgtfacq_e,
                vwind_impl_wgt=vwind_impl_wgt,
                hmask_dd3d=hmask_dd3d,
                scalfac_dd3d=scalfac_dd3d,
                coeff1_dwdz=coeff1_dwdz,
                coeff2_dwdz=coeff2_dwdz,
                coeff_gradekin=coeff_gradekin,
                c_owner_mask=c_owner_mask,
                itime_scheme=itime_scheme,
                iadv_rhotheta=iadv_rhotheta,
                igradp_method=igradp_method,
                rayleigh_type=rayleigh_type,
                divdamp_order=divdamp_order,
                divdamp_type=divdamp_type,
                l_vert_nested=l_vert_nested,
                ldeepatmo=ldeepatmo,
                iau_init=iau_init,
                extra_diffu=extra_diffu,
                rhotheta_offctr=rhotheta_offctr,
                veladv_offctr=veladv_offctr,
                nudge_max_coeff=nudge_max_coeff,
                divdamp_fac=divdamp_fac,
                divdamp_fac2=divdamp_fac2,
                divdamp_fac3=divdamp_fac3,
                divdamp_fac4=divdamp_fac4,
                divdamp_z=divdamp_z,
                divdamp_z2=divdamp_z2,
                divdamp_z3=divdamp_z3,
                divdamp_z4=divdamp_z4,
                nflat_gradp=nflat_gradp,
                backend=backend,
            )

            if __debug__:
                if runtime_config.PROFILING:
                    func_end_time = _runtime.perf_counter()
                    logger.info(
                        "solve_nh_init convert time: %s"
                        % str(
                            perf_counters["convert_end_time"] - perf_counters["convert_start_time"]
                        )
                    )
                    logger.info(
                        "solve_nh_init execution time: %s" % str(func_end_time - func_start_time)
                    )

            if __debug__:
                if logger.isEnabledFor(logging.DEBUG):

                    c_lin_e_arr = (
                        _conversion.as_array(ffi, c_lin_e) if c_lin_e is not None else None
                    )
                    msg = "shape of c_lin_e after computation = %s" % str(
                        c_lin_e_arr.shape if c_lin_e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "c_lin_e after computation: %s" % str(c_lin_e_arr)
                        if c_lin_e is not None
                        else "None"
                    )
                    logger.debug(msg)

                    c_intp_arr = _conversion.as_array(ffi, c_intp) if c_intp is not None else None
                    msg = "shape of c_intp after computation = %s" % str(
                        c_intp_arr.shape if c_intp is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "c_intp after computation: %s" % str(c_intp_arr)
                        if c_intp is not None
                        else "None"
                    )
                    logger.debug(msg)

                    e_flx_avg_arr = (
                        _conversion.as_array(ffi, e_flx_avg) if e_flx_avg is not None else None
                    )
                    msg = "shape of e_flx_avg after computation = %s" % str(
                        e_flx_avg_arr.shape if e_flx_avg is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "e_flx_avg after computation: %s" % str(e_flx_avg_arr)
                        if e_flx_avg is not None
                        else "None"
                    )
                    logger.debug(msg)

                    geofac_grdiv_arr = (
                        _conversion.as_array(ffi, geofac_grdiv)
                        if geofac_grdiv is not None
                        else None
                    )
                    msg = "shape of geofac_grdiv after computation = %s" % str(
                        geofac_grdiv_arr.shape if geofac_grdiv is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "geofac_grdiv after computation: %s" % str(geofac_grdiv_arr)
                        if geofac_grdiv is not None
                        else "None"
                    )
                    logger.debug(msg)

                    geofac_rot_arr = (
                        _conversion.as_array(ffi, geofac_rot) if geofac_rot is not None else None
                    )
                    msg = "shape of geofac_rot after computation = %s" % str(
                        geofac_rot_arr.shape if geofac_rot is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "geofac_rot after computation: %s" % str(geofac_rot_arr)
                        if geofac_rot is not None
                        else "None"
                    )
                    logger.debug(msg)

                    pos_on_tplane_e_1_arr = (
                        _conversion.as_array(ffi, pos_on_tplane_e_1)
                        if pos_on_tplane_e_1 is not None
                        else None
                    )
                    msg = "shape of pos_on_tplane_e_1 after computation = %s" % str(
                        pos_on_tplane_e_1_arr.shape if pos_on_tplane_e_1 is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "pos_on_tplane_e_1 after computation: %s" % str(pos_on_tplane_e_1_arr)
                        if pos_on_tplane_e_1 is not None
                        else "None"
                    )
                    logger.debug(msg)

                    pos_on_tplane_e_2_arr = (
                        _conversion.as_array(ffi, pos_on_tplane_e_2)
                        if pos_on_tplane_e_2 is not None
                        else None
                    )
                    msg = "shape of pos_on_tplane_e_2 after computation = %s" % str(
                        pos_on_tplane_e_2_arr.shape if pos_on_tplane_e_2 is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "pos_on_tplane_e_2 after computation: %s" % str(pos_on_tplane_e_2_arr)
                        if pos_on_tplane_e_2 is not None
                        else "None"
                    )
                    logger.debug(msg)

                    rbf_vec_coeff_e_arr = (
                        _conversion.as_array(ffi, rbf_vec_coeff_e)
                        if rbf_vec_coeff_e is not None
                        else None
                    )
                    msg = "shape of rbf_vec_coeff_e after computation = %s" % str(
                        rbf_vec_coeff_e_arr.shape if rbf_vec_coeff_e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "rbf_vec_coeff_e after computation: %s" % str(rbf_vec_coeff_e_arr)
                        if rbf_vec_coeff_e is not None
                        else "None"
                    )
                    logger.debug(msg)

                    e_bln_c_s_arr = (
                        _conversion.as_array(ffi, e_bln_c_s) if e_bln_c_s is not None else None
                    )
                    msg = "shape of e_bln_c_s after computation = %s" % str(
                        e_bln_c_s_arr.shape if e_bln_c_s is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "e_bln_c_s after computation: %s" % str(e_bln_c_s_arr)
                        if e_bln_c_s is not None
                        else "None"
                    )
                    logger.debug(msg)

                    rbf_vec_coeff_v_arr = (
                        _conversion.as_array(ffi, rbf_vec_coeff_v)
                        if rbf_vec_coeff_v is not None
                        else None
                    )
                    msg = "shape of rbf_vec_coeff_v after computation = %s" % str(
                        rbf_vec_coeff_v_arr.shape if rbf_vec_coeff_v is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "rbf_vec_coeff_v after computation: %s" % str(rbf_vec_coeff_v_arr)
                        if rbf_vec_coeff_v is not None
                        else "None"
                    )
                    logger.debug(msg)

                    geofac_div_arr = (
                        _conversion.as_array(ffi, geofac_div) if geofac_div is not None else None
                    )
                    msg = "shape of geofac_div after computation = %s" % str(
                        geofac_div_arr.shape if geofac_div is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "geofac_div after computation: %s" % str(geofac_div_arr)
                        if geofac_div is not None
                        else "None"
                    )
                    logger.debug(msg)

                    geofac_n2s_arr = (
                        _conversion.as_array(ffi, geofac_n2s) if geofac_n2s is not None else None
                    )
                    msg = "shape of geofac_n2s after computation = %s" % str(
                        geofac_n2s_arr.shape if geofac_n2s is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "geofac_n2s after computation: %s" % str(geofac_n2s_arr)
                        if geofac_n2s is not None
                        else "None"
                    )
                    logger.debug(msg)

                    geofac_grg_x_arr = (
                        _conversion.as_array(ffi, geofac_grg_x)
                        if geofac_grg_x is not None
                        else None
                    )
                    msg = "shape of geofac_grg_x after computation = %s" % str(
                        geofac_grg_x_arr.shape if geofac_grg_x is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "geofac_grg_x after computation: %s" % str(geofac_grg_x_arr)
                        if geofac_grg_x is not None
                        else "None"
                    )
                    logger.debug(msg)

                    geofac_grg_y_arr = (
                        _conversion.as_array(ffi, geofac_grg_y)
                        if geofac_grg_y is not None
                        else None
                    )
                    msg = "shape of geofac_grg_y after computation = %s" % str(
                        geofac_grg_y_arr.shape if geofac_grg_y is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "geofac_grg_y after computation: %s" % str(geofac_grg_y_arr)
                        if geofac_grg_y is not None
                        else "None"
                    )
                    logger.debug(msg)

                    nudgecoeff_e_arr = (
                        _conversion.as_array(ffi, nudgecoeff_e)
                        if nudgecoeff_e is not None
                        else None
                    )
                    msg = "shape of nudgecoeff_e after computation = %s" % str(
                        nudgecoeff_e_arr.shape if nudgecoeff_e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "nudgecoeff_e after computation: %s" % str(nudgecoeff_e_arr)
                        if nudgecoeff_e is not None
                        else "None"
                    )
                    logger.debug(msg)

                    mask_prog_halo_c_arr = (
                        _conversion.as_array(ffi, mask_prog_halo_c)
                        if mask_prog_halo_c is not None
                        else None
                    )
                    msg = "shape of mask_prog_halo_c after computation = %s" % str(
                        mask_prog_halo_c_arr.shape if mask_prog_halo_c is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "mask_prog_halo_c after computation: %s" % str(mask_prog_halo_c_arr)
                        if mask_prog_halo_c is not None
                        else "None"
                    )
                    logger.debug(msg)

                    rayleigh_w_arr = (
                        _conversion.as_array(ffi, rayleigh_w) if rayleigh_w is not None else None
                    )
                    msg = "shape of rayleigh_w after computation = %s" % str(
                        rayleigh_w_arr.shape if rayleigh_w is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "rayleigh_w after computation: %s" % str(rayleigh_w_arr)
                        if rayleigh_w is not None
                        else "None"
                    )
                    logger.debug(msg)

                    exner_exfac_arr = (
                        _conversion.as_array(ffi, exner_exfac) if exner_exfac is not None else None
                    )
                    msg = "shape of exner_exfac after computation = %s" % str(
                        exner_exfac_arr.shape if exner_exfac is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "exner_exfac after computation: %s" % str(exner_exfac_arr)
                        if exner_exfac is not None
                        else "None"
                    )
                    logger.debug(msg)

                    exner_ref_mc_arr = (
                        _conversion.as_array(ffi, exner_ref_mc)
                        if exner_ref_mc is not None
                        else None
                    )
                    msg = "shape of exner_ref_mc after computation = %s" % str(
                        exner_ref_mc_arr.shape if exner_ref_mc is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "exner_ref_mc after computation: %s" % str(exner_ref_mc_arr)
                        if exner_ref_mc is not None
                        else "None"
                    )
                    logger.debug(msg)

                    wgtfac_c_arr = (
                        _conversion.as_array(ffi, wgtfac_c) if wgtfac_c is not None else None
                    )
                    msg = "shape of wgtfac_c after computation = %s" % str(
                        wgtfac_c_arr.shape if wgtfac_c is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "wgtfac_c after computation: %s" % str(wgtfac_c_arr)
                        if wgtfac_c is not None
                        else "None"
                    )
                    logger.debug(msg)

                    wgtfacq_c_arr = (
                        _conversion.as_array(ffi, wgtfacq_c) if wgtfacq_c is not None else None
                    )
                    msg = "shape of wgtfacq_c after computation = %s" % str(
                        wgtfacq_c_arr.shape if wgtfacq_c is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "wgtfacq_c after computation: %s" % str(wgtfacq_c_arr)
                        if wgtfacq_c is not None
                        else "None"
                    )
                    logger.debug(msg)

                    inv_ddqz_z_full_arr = (
                        _conversion.as_array(ffi, inv_ddqz_z_full)
                        if inv_ddqz_z_full is not None
                        else None
                    )
                    msg = "shape of inv_ddqz_z_full after computation = %s" % str(
                        inv_ddqz_z_full_arr.shape if inv_ddqz_z_full is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "inv_ddqz_z_full after computation: %s" % str(inv_ddqz_z_full_arr)
                        if inv_ddqz_z_full is not None
                        else "None"
                    )
                    logger.debug(msg)

                    rho_ref_mc_arr = (
                        _conversion.as_array(ffi, rho_ref_mc) if rho_ref_mc is not None else None
                    )
                    msg = "shape of rho_ref_mc after computation = %s" % str(
                        rho_ref_mc_arr.shape if rho_ref_mc is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "rho_ref_mc after computation: %s" % str(rho_ref_mc_arr)
                        if rho_ref_mc is not None
                        else "None"
                    )
                    logger.debug(msg)

                    theta_ref_mc_arr = (
                        _conversion.as_array(ffi, theta_ref_mc)
                        if theta_ref_mc is not None
                        else None
                    )
                    msg = "shape of theta_ref_mc after computation = %s" % str(
                        theta_ref_mc_arr.shape if theta_ref_mc is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "theta_ref_mc after computation: %s" % str(theta_ref_mc_arr)
                        if theta_ref_mc is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vwind_expl_wgt_arr = (
                        _conversion.as_array(ffi, vwind_expl_wgt)
                        if vwind_expl_wgt is not None
                        else None
                    )
                    msg = "shape of vwind_expl_wgt after computation = %s" % str(
                        vwind_expl_wgt_arr.shape if vwind_expl_wgt is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vwind_expl_wgt after computation: %s" % str(vwind_expl_wgt_arr)
                        if vwind_expl_wgt is not None
                        else "None"
                    )
                    logger.debug(msg)

                    d_exner_dz_ref_ic_arr = (
                        _conversion.as_array(ffi, d_exner_dz_ref_ic)
                        if d_exner_dz_ref_ic is not None
                        else None
                    )
                    msg = "shape of d_exner_dz_ref_ic after computation = %s" % str(
                        d_exner_dz_ref_ic_arr.shape if d_exner_dz_ref_ic is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "d_exner_dz_ref_ic after computation: %s" % str(d_exner_dz_ref_ic_arr)
                        if d_exner_dz_ref_ic is not None
                        else "None"
                    )
                    logger.debug(msg)

                    ddqz_z_half_arr = (
                        _conversion.as_array(ffi, ddqz_z_half) if ddqz_z_half is not None else None
                    )
                    msg = "shape of ddqz_z_half after computation = %s" % str(
                        ddqz_z_half_arr.shape if ddqz_z_half is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "ddqz_z_half after computation: %s" % str(ddqz_z_half_arr)
                        if ddqz_z_half is not None
                        else "None"
                    )
                    logger.debug(msg)

                    theta_ref_ic_arr = (
                        _conversion.as_array(ffi, theta_ref_ic)
                        if theta_ref_ic is not None
                        else None
                    )
                    msg = "shape of theta_ref_ic after computation = %s" % str(
                        theta_ref_ic_arr.shape if theta_ref_ic is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "theta_ref_ic after computation: %s" % str(theta_ref_ic_arr)
                        if theta_ref_ic is not None
                        else "None"
                    )
                    logger.debug(msg)

                    d2dexdz2_fac1_mc_arr = (
                        _conversion.as_array(ffi, d2dexdz2_fac1_mc)
                        if d2dexdz2_fac1_mc is not None
                        else None
                    )
                    msg = "shape of d2dexdz2_fac1_mc after computation = %s" % str(
                        d2dexdz2_fac1_mc_arr.shape if d2dexdz2_fac1_mc is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "d2dexdz2_fac1_mc after computation: %s" % str(d2dexdz2_fac1_mc_arr)
                        if d2dexdz2_fac1_mc is not None
                        else "None"
                    )
                    logger.debug(msg)

                    d2dexdz2_fac2_mc_arr = (
                        _conversion.as_array(ffi, d2dexdz2_fac2_mc)
                        if d2dexdz2_fac2_mc is not None
                        else None
                    )
                    msg = "shape of d2dexdz2_fac2_mc after computation = %s" % str(
                        d2dexdz2_fac2_mc_arr.shape if d2dexdz2_fac2_mc is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "d2dexdz2_fac2_mc after computation: %s" % str(d2dexdz2_fac2_mc_arr)
                        if d2dexdz2_fac2_mc is not None
                        else "None"
                    )
                    logger.debug(msg)

                    rho_ref_me_arr = (
                        _conversion.as_array(ffi, rho_ref_me) if rho_ref_me is not None else None
                    )
                    msg = "shape of rho_ref_me after computation = %s" % str(
                        rho_ref_me_arr.shape if rho_ref_me is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "rho_ref_me after computation: %s" % str(rho_ref_me_arr)
                        if rho_ref_me is not None
                        else "None"
                    )
                    logger.debug(msg)

                    theta_ref_me_arr = (
                        _conversion.as_array(ffi, theta_ref_me)
                        if theta_ref_me is not None
                        else None
                    )
                    msg = "shape of theta_ref_me after computation = %s" % str(
                        theta_ref_me_arr.shape if theta_ref_me is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "theta_ref_me after computation: %s" % str(theta_ref_me_arr)
                        if theta_ref_me is not None
                        else "None"
                    )
                    logger.debug(msg)

                    ddxn_z_full_arr = (
                        _conversion.as_array(ffi, ddxn_z_full) if ddxn_z_full is not None else None
                    )
                    msg = "shape of ddxn_z_full after computation = %s" % str(
                        ddxn_z_full_arr.shape if ddxn_z_full is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "ddxn_z_full after computation: %s" % str(ddxn_z_full_arr)
                        if ddxn_z_full is not None
                        else "None"
                    )
                    logger.debug(msg)

                    zdiff_gradp_arr = (
                        _conversion.as_array(ffi, zdiff_gradp) if zdiff_gradp is not None else None
                    )
                    msg = "shape of zdiff_gradp after computation = %s" % str(
                        zdiff_gradp_arr.shape if zdiff_gradp is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "zdiff_gradp after computation: %s" % str(zdiff_gradp_arr)
                        if zdiff_gradp is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vertidx_gradp_arr = (
                        _conversion.as_array(ffi, vertidx_gradp)
                        if vertidx_gradp is not None
                        else None
                    )
                    msg = "shape of vertidx_gradp after computation = %s" % str(
                        vertidx_gradp_arr.shape if vertidx_gradp is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vertidx_gradp after computation: %s" % str(vertidx_gradp_arr)
                        if vertidx_gradp is not None
                        else "None"
                    )
                    logger.debug(msg)

                    pg_edgeidx_arr = (
                        _conversion.as_array(ffi, pg_edgeidx) if pg_edgeidx is not None else None
                    )
                    msg = "shape of pg_edgeidx after computation = %s" % str(
                        pg_edgeidx_arr.shape if pg_edgeidx is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "pg_edgeidx after computation: %s" % str(pg_edgeidx_arr)
                        if pg_edgeidx is not None
                        else "None"
                    )
                    logger.debug(msg)

                    pg_vertidx_arr = (
                        _conversion.as_array(ffi, pg_vertidx) if pg_vertidx is not None else None
                    )
                    msg = "shape of pg_vertidx after computation = %s" % str(
                        pg_vertidx_arr.shape if pg_vertidx is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "pg_vertidx after computation: %s" % str(pg_vertidx_arr)
                        if pg_vertidx is not None
                        else "None"
                    )
                    logger.debug(msg)

                    pg_exdist_arr = (
                        _conversion.as_array(ffi, pg_exdist) if pg_exdist is not None else None
                    )
                    msg = "shape of pg_exdist after computation = %s" % str(
                        pg_exdist_arr.shape if pg_exdist is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "pg_exdist after computation: %s" % str(pg_exdist_arr)
                        if pg_exdist is not None
                        else "None"
                    )
                    logger.debug(msg)

                    ddqz_z_full_e_arr = (
                        _conversion.as_array(ffi, ddqz_z_full_e)
                        if ddqz_z_full_e is not None
                        else None
                    )
                    msg = "shape of ddqz_z_full_e after computation = %s" % str(
                        ddqz_z_full_e_arr.shape if ddqz_z_full_e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "ddqz_z_full_e after computation: %s" % str(ddqz_z_full_e_arr)
                        if ddqz_z_full_e is not None
                        else "None"
                    )
                    logger.debug(msg)

                    ddxt_z_full_arr = (
                        _conversion.as_array(ffi, ddxt_z_full) if ddxt_z_full is not None else None
                    )
                    msg = "shape of ddxt_z_full after computation = %s" % str(
                        ddxt_z_full_arr.shape if ddxt_z_full is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "ddxt_z_full after computation: %s" % str(ddxt_z_full_arr)
                        if ddxt_z_full is not None
                        else "None"
                    )
                    logger.debug(msg)

                    wgtfac_e_arr = (
                        _conversion.as_array(ffi, wgtfac_e) if wgtfac_e is not None else None
                    )
                    msg = "shape of wgtfac_e after computation = %s" % str(
                        wgtfac_e_arr.shape if wgtfac_e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "wgtfac_e after computation: %s" % str(wgtfac_e_arr)
                        if wgtfac_e is not None
                        else "None"
                    )
                    logger.debug(msg)

                    wgtfacq_e_arr = (
                        _conversion.as_array(ffi, wgtfacq_e) if wgtfacq_e is not None else None
                    )
                    msg = "shape of wgtfacq_e after computation = %s" % str(
                        wgtfacq_e_arr.shape if wgtfacq_e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "wgtfacq_e after computation: %s" % str(wgtfacq_e_arr)
                        if wgtfacq_e is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vwind_impl_wgt_arr = (
                        _conversion.as_array(ffi, vwind_impl_wgt)
                        if vwind_impl_wgt is not None
                        else None
                    )
                    msg = "shape of vwind_impl_wgt after computation = %s" % str(
                        vwind_impl_wgt_arr.shape if vwind_impl_wgt is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vwind_impl_wgt after computation: %s" % str(vwind_impl_wgt_arr)
                        if vwind_impl_wgt is not None
                        else "None"
                    )
                    logger.debug(msg)

                    hmask_dd3d_arr = (
                        _conversion.as_array(ffi, hmask_dd3d) if hmask_dd3d is not None else None
                    )
                    msg = "shape of hmask_dd3d after computation = %s" % str(
                        hmask_dd3d_arr.shape if hmask_dd3d is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "hmask_dd3d after computation: %s" % str(hmask_dd3d_arr)
                        if hmask_dd3d is not None
                        else "None"
                    )
                    logger.debug(msg)

                    scalfac_dd3d_arr = (
                        _conversion.as_array(ffi, scalfac_dd3d)
                        if scalfac_dd3d is not None
                        else None
                    )
                    msg = "shape of scalfac_dd3d after computation = %s" % str(
                        scalfac_dd3d_arr.shape if scalfac_dd3d is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "scalfac_dd3d after computation: %s" % str(scalfac_dd3d_arr)
                        if scalfac_dd3d is not None
                        else "None"
                    )
                    logger.debug(msg)

                    coeff1_dwdz_arr = (
                        _conversion.as_array(ffi, coeff1_dwdz) if coeff1_dwdz is not None else None
                    )
                    msg = "shape of coeff1_dwdz after computation = %s" % str(
                        coeff1_dwdz_arr.shape if coeff1_dwdz is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "coeff1_dwdz after computation: %s" % str(coeff1_dwdz_arr)
                        if coeff1_dwdz is not None
                        else "None"
                    )
                    logger.debug(msg)

                    coeff2_dwdz_arr = (
                        _conversion.as_array(ffi, coeff2_dwdz) if coeff2_dwdz is not None else None
                    )
                    msg = "shape of coeff2_dwdz after computation = %s" % str(
                        coeff2_dwdz_arr.shape if coeff2_dwdz is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "coeff2_dwdz after computation: %s" % str(coeff2_dwdz_arr)
                        if coeff2_dwdz is not None
                        else "None"
                    )
                    logger.debug(msg)

                    coeff_gradekin_arr = (
                        _conversion.as_array(ffi, coeff_gradekin)
                        if coeff_gradekin is not None
                        else None
                    )
                    msg = "shape of coeff_gradekin after computation = %s" % str(
                        coeff_gradekin_arr.shape if coeff_gradekin is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "coeff_gradekin after computation: %s" % str(coeff_gradekin_arr)
                        if coeff_gradekin is not None
                        else "None"
                    )
                    logger.debug(msg)

                    c_owner_mask_arr = (
                        _conversion.as_array(ffi, c_owner_mask)
                        if c_owner_mask is not None
                        else None
                    )
                    msg = "shape of c_owner_mask after computation = %s" % str(
                        c_owner_mask_arr.shape if c_owner_mask is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "c_owner_mask after computation: %s" % str(c_owner_mask_arr)
                        if c_owner_mask is not None
                        else "None"
                    )
                    logger.debug(msg)

            if __debug__:
                logger.info("Python execution of solve_nh_init completed.")

        except Exception as e:
            logger.exception(f"A Python error occurred: {e}")
            return 2

    return 1


@ffi.def_extern(error=2)
def solve_nh_run_wrapper(
    rho_now,
    rho_now_size_0,
    rho_now_size_1,
    rho_new,
    rho_new_size_0,
    rho_new_size_1,
    exner_now,
    exner_now_size_0,
    exner_now_size_1,
    exner_new,
    exner_new_size_0,
    exner_new_size_1,
    w_now,
    w_now_size_0,
    w_now_size_1,
    w_new,
    w_new_size_0,
    w_new_size_1,
    theta_v_now,
    theta_v_now_size_0,
    theta_v_now_size_1,
    theta_v_new,
    theta_v_new_size_0,
    theta_v_new_size_1,
    vn_now,
    vn_now_size_0,
    vn_now_size_1,
    vn_new,
    vn_new_size_0,
    vn_new_size_1,
    w_concorr_c,
    w_concorr_c_size_0,
    w_concorr_c_size_1,
    ddt_vn_apc_ntl1,
    ddt_vn_apc_ntl1_size_0,
    ddt_vn_apc_ntl1_size_1,
    ddt_vn_apc_ntl2,
    ddt_vn_apc_ntl2_size_0,
    ddt_vn_apc_ntl2_size_1,
    ddt_w_adv_ntl1,
    ddt_w_adv_ntl1_size_0,
    ddt_w_adv_ntl1_size_1,
    ddt_w_adv_ntl2,
    ddt_w_adv_ntl2_size_0,
    ddt_w_adv_ntl2_size_1,
    theta_v_ic,
    theta_v_ic_size_0,
    theta_v_ic_size_1,
    rho_ic,
    rho_ic_size_0,
    rho_ic_size_1,
    exner_pr,
    exner_pr_size_0,
    exner_pr_size_1,
    exner_dyn_incr,
    exner_dyn_incr_size_0,
    exner_dyn_incr_size_1,
    ddt_exner_phy,
    ddt_exner_phy_size_0,
    ddt_exner_phy_size_1,
    grf_tend_rho,
    grf_tend_rho_size_0,
    grf_tend_rho_size_1,
    grf_tend_thv,
    grf_tend_thv_size_0,
    grf_tend_thv_size_1,
    grf_tend_w,
    grf_tend_w_size_0,
    grf_tend_w_size_1,
    mass_fl_e,
    mass_fl_e_size_0,
    mass_fl_e_size_1,
    ddt_vn_phy,
    ddt_vn_phy_size_0,
    ddt_vn_phy_size_1,
    grf_tend_vn,
    grf_tend_vn_size_0,
    grf_tend_vn_size_1,
    vn_ie,
    vn_ie_size_0,
    vn_ie_size_1,
    vt,
    vt_size_0,
    vt_size_1,
    vn_incr,
    vn_incr_size_0,
    vn_incr_size_1,
    rho_incr,
    rho_incr_size_0,
    rho_incr_size_1,
    exner_incr,
    exner_incr_size_0,
    exner_incr_size_1,
    mass_flx_me,
    mass_flx_me_size_0,
    mass_flx_me_size_1,
    mass_flx_ic,
    mass_flx_ic_size_0,
    mass_flx_ic_size_1,
    vol_flx_ic,
    vol_flx_ic_size_0,
    vol_flx_ic_size_1,
    vn_traj,
    vn_traj_size_0,
    vn_traj_size_1,
    dtime,
    max_vcfl_size1_array,
    max_vcfl_size1_array_size_0,
    lprep_adv,
    at_initial_timestep,
    divdamp_fac_o2,
    ndyn_substeps_var,
    idyn_timestep,
    is_iau_active,
    iau_wgt_dyn,
    on_gpu,
):
    with runtime_config.HOOK_BINDINGS_FUNCTION["solve_nh_run"]:
        try:
            if __debug__:
                logger.info("Python execution of solve_nh_run started.")

            if __debug__:
                if runtime_config.PROFILING:
                    unpack_start_time = _runtime.perf_counter()

            # ArrayInfos

            rho_now = (
                rho_now,
                (
                    rho_now_size_0,
                    rho_now_size_1,
                ),
                on_gpu,
                False,
            )

            rho_new = (
                rho_new,
                (
                    rho_new_size_0,
                    rho_new_size_1,
                ),
                on_gpu,
                False,
            )

            exner_now = (
                exner_now,
                (
                    exner_now_size_0,
                    exner_now_size_1,
                ),
                on_gpu,
                False,
            )

            exner_new = (
                exner_new,
                (
                    exner_new_size_0,
                    exner_new_size_1,
                ),
                on_gpu,
                False,
            )

            w_now = (
                w_now,
                (
                    w_now_size_0,
                    w_now_size_1,
                ),
                on_gpu,
                False,
            )

            w_new = (
                w_new,
                (
                    w_new_size_0,
                    w_new_size_1,
                ),
                on_gpu,
                False,
            )

            theta_v_now = (
                theta_v_now,
                (
                    theta_v_now_size_0,
                    theta_v_now_size_1,
                ),
                on_gpu,
                False,
            )

            theta_v_new = (
                theta_v_new,
                (
                    theta_v_new_size_0,
                    theta_v_new_size_1,
                ),
                on_gpu,
                False,
            )

            vn_now = (
                vn_now,
                (
                    vn_now_size_0,
                    vn_now_size_1,
                ),
                on_gpu,
                False,
            )

            vn_new = (
                vn_new,
                (
                    vn_new_size_0,
                    vn_new_size_1,
                ),
                on_gpu,
                False,
            )

            w_concorr_c = (
                w_concorr_c,
                (
                    w_concorr_c_size_0,
                    w_concorr_c_size_1,
                ),
                on_gpu,
                False,
            )

            ddt_vn_apc_ntl1 = (
                ddt_vn_apc_ntl1,
                (
                    ddt_vn_apc_ntl1_size_0,
                    ddt_vn_apc_ntl1_size_1,
                ),
                on_gpu,
                False,
            )

            ddt_vn_apc_ntl2 = (
                ddt_vn_apc_ntl2,
                (
                    ddt_vn_apc_ntl2_size_0,
                    ddt_vn_apc_ntl2_size_1,
                ),
                on_gpu,
                False,
            )

            ddt_w_adv_ntl1 = (
                ddt_w_adv_ntl1,
                (
                    ddt_w_adv_ntl1_size_0,
                    ddt_w_adv_ntl1_size_1,
                ),
                on_gpu,
                False,
            )

            ddt_w_adv_ntl2 = (
                ddt_w_adv_ntl2,
                (
                    ddt_w_adv_ntl2_size_0,
                    ddt_w_adv_ntl2_size_1,
                ),
                on_gpu,
                False,
            )

            theta_v_ic = (
                theta_v_ic,
                (
                    theta_v_ic_size_0,
                    theta_v_ic_size_1,
                ),
                on_gpu,
                False,
            )

            rho_ic = (
                rho_ic,
                (
                    rho_ic_size_0,
                    rho_ic_size_1,
                ),
                on_gpu,
                False,
            )

            exner_pr = (
                exner_pr,
                (
                    exner_pr_size_0,
                    exner_pr_size_1,
                ),
                on_gpu,
                False,
            )

            exner_dyn_incr = (
                exner_dyn_incr,
                (
                    exner_dyn_incr_size_0,
                    exner_dyn_incr_size_1,
                ),
                on_gpu,
                False,
            )

            ddt_exner_phy = (
                ddt_exner_phy,
                (
                    ddt_exner_phy_size_0,
                    ddt_exner_phy_size_1,
                ),
                on_gpu,
                False,
            )

            grf_tend_rho = (
                grf_tend_rho,
                (
                    grf_tend_rho_size_0,
                    grf_tend_rho_size_1,
                ),
                on_gpu,
                False,
            )

            grf_tend_thv = (
                grf_tend_thv,
                (
                    grf_tend_thv_size_0,
                    grf_tend_thv_size_1,
                ),
                on_gpu,
                False,
            )

            grf_tend_w = (
                grf_tend_w,
                (
                    grf_tend_w_size_0,
                    grf_tend_w_size_1,
                ),
                on_gpu,
                False,
            )

            mass_fl_e = (
                mass_fl_e,
                (
                    mass_fl_e_size_0,
                    mass_fl_e_size_1,
                ),
                on_gpu,
                False,
            )

            ddt_vn_phy = (
                ddt_vn_phy,
                (
                    ddt_vn_phy_size_0,
                    ddt_vn_phy_size_1,
                ),
                on_gpu,
                False,
            )

            grf_tend_vn = (
                grf_tend_vn,
                (
                    grf_tend_vn_size_0,
                    grf_tend_vn_size_1,
                ),
                on_gpu,
                False,
            )

            vn_ie = (
                vn_ie,
                (
                    vn_ie_size_0,
                    vn_ie_size_1,
                ),
                on_gpu,
                False,
            )

            vt = (
                vt,
                (
                    vt_size_0,
                    vt_size_1,
                ),
                on_gpu,
                False,
            )

            vn_incr = (
                vn_incr,
                (
                    vn_incr_size_0,
                    vn_incr_size_1,
                ),
                on_gpu,
                True,
            )

            rho_incr = (
                rho_incr,
                (
                    rho_incr_size_0,
                    rho_incr_size_1,
                ),
                on_gpu,
                True,
            )

            exner_incr = (
                exner_incr,
                (
                    exner_incr_size_0,
                    exner_incr_size_1,
                ),
                on_gpu,
                True,
            )

            mass_flx_me = (
                mass_flx_me,
                (
                    mass_flx_me_size_0,
                    mass_flx_me_size_1,
                ),
                on_gpu,
                False,
            )

            mass_flx_ic = (
                mass_flx_ic,
                (
                    mass_flx_ic_size_0,
                    mass_flx_ic_size_1,
                ),
                on_gpu,
                False,
            )

            vol_flx_ic = (
                vol_flx_ic,
                (
                    vol_flx_ic_size_0,
                    vol_flx_ic_size_1,
                ),
                on_gpu,
                False,
            )

            vn_traj = (
                vn_traj,
                (
                    vn_traj_size_0,
                    vn_traj_size_1,
                ),
                on_gpu,
                False,
            )

            max_vcfl_size1_array = (
                max_vcfl_size1_array,
                (max_vcfl_size1_array_size_0,),
                False,
                False,
            )

            if __debug__:
                if runtime_config.PROFILING:
                    allocate_end_time = _runtime.perf_counter()
                    logger.info(
                        "solve_nh_run constructing `ArrayInfos` time: %s"
                        % str(allocate_end_time - unpack_start_time)
                    )

                    func_start_time = _runtime.perf_counter()

            if __debug__ and runtime_config.PROFILING:
                perf_counters = {}
            else:
                perf_counters = None
            solve_nh_run(
                ffi=ffi,
                perf_counters=perf_counters,
                rho_now=rho_now,
                rho_new=rho_new,
                exner_now=exner_now,
                exner_new=exner_new,
                w_now=w_now,
                w_new=w_new,
                theta_v_now=theta_v_now,
                theta_v_new=theta_v_new,
                vn_now=vn_now,
                vn_new=vn_new,
                w_concorr_c=w_concorr_c,
                ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
                ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
                ddt_w_adv_ntl1=ddt_w_adv_ntl1,
                ddt_w_adv_ntl2=ddt_w_adv_ntl2,
                theta_v_ic=theta_v_ic,
                rho_ic=rho_ic,
                exner_pr=exner_pr,
                exner_dyn_incr=exner_dyn_incr,
                ddt_exner_phy=ddt_exner_phy,
                grf_tend_rho=grf_tend_rho,
                grf_tend_thv=grf_tend_thv,
                grf_tend_w=grf_tend_w,
                mass_fl_e=mass_fl_e,
                ddt_vn_phy=ddt_vn_phy,
                grf_tend_vn=grf_tend_vn,
                vn_ie=vn_ie,
                vt=vt,
                vn_incr=vn_incr,
                rho_incr=rho_incr,
                exner_incr=exner_incr,
                mass_flx_me=mass_flx_me,
                mass_flx_ic=mass_flx_ic,
                vol_flx_ic=vol_flx_ic,
                vn_traj=vn_traj,
                dtime=dtime,
                max_vcfl_size1_array=max_vcfl_size1_array,
                lprep_adv=lprep_adv,
                at_initial_timestep=at_initial_timestep,
                divdamp_fac_o2=divdamp_fac_o2,
                ndyn_substeps_var=ndyn_substeps_var,
                idyn_timestep=idyn_timestep,
                is_iau_active=is_iau_active,
                iau_wgt_dyn=iau_wgt_dyn,
            )

            if __debug__:
                if runtime_config.PROFILING:
                    func_end_time = _runtime.perf_counter()
                    logger.info(
                        "solve_nh_run convert time: %s"
                        % str(
                            perf_counters["convert_end_time"] - perf_counters["convert_start_time"]
                        )
                    )
                    logger.info(
                        "solve_nh_run execution time: %s" % str(func_end_time - func_start_time)
                    )

            if __debug__:
                if logger.isEnabledFor(logging.DEBUG):

                    rho_now_arr = (
                        _conversion.as_array(ffi, rho_now) if rho_now is not None else None
                    )
                    msg = "shape of rho_now after computation = %s" % str(
                        rho_now_arr.shape if rho_now is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "rho_now after computation: %s" % str(rho_now_arr)
                        if rho_now is not None
                        else "None"
                    )
                    logger.debug(msg)

                    rho_new_arr = (
                        _conversion.as_array(ffi, rho_new) if rho_new is not None else None
                    )
                    msg = "shape of rho_new after computation = %s" % str(
                        rho_new_arr.shape if rho_new is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "rho_new after computation: %s" % str(rho_new_arr)
                        if rho_new is not None
                        else "None"
                    )
                    logger.debug(msg)

                    exner_now_arr = (
                        _conversion.as_array(ffi, exner_now) if exner_now is not None else None
                    )
                    msg = "shape of exner_now after computation = %s" % str(
                        exner_now_arr.shape if exner_now is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "exner_now after computation: %s" % str(exner_now_arr)
                        if exner_now is not None
                        else "None"
                    )
                    logger.debug(msg)

                    exner_new_arr = (
                        _conversion.as_array(ffi, exner_new) if exner_new is not None else None
                    )
                    msg = "shape of exner_new after computation = %s" % str(
                        exner_new_arr.shape if exner_new is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "exner_new after computation: %s" % str(exner_new_arr)
                        if exner_new is not None
                        else "None"
                    )
                    logger.debug(msg)

                    w_now_arr = _conversion.as_array(ffi, w_now) if w_now is not None else None
                    msg = "shape of w_now after computation = %s" % str(
                        w_now_arr.shape if w_now is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "w_now after computation: %s" % str(w_now_arr)
                        if w_now is not None
                        else "None"
                    )
                    logger.debug(msg)

                    w_new_arr = _conversion.as_array(ffi, w_new) if w_new is not None else None
                    msg = "shape of w_new after computation = %s" % str(
                        w_new_arr.shape if w_new is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "w_new after computation: %s" % str(w_new_arr)
                        if w_new is not None
                        else "None"
                    )
                    logger.debug(msg)

                    theta_v_now_arr = (
                        _conversion.as_array(ffi, theta_v_now) if theta_v_now is not None else None
                    )
                    msg = "shape of theta_v_now after computation = %s" % str(
                        theta_v_now_arr.shape if theta_v_now is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "theta_v_now after computation: %s" % str(theta_v_now_arr)
                        if theta_v_now is not None
                        else "None"
                    )
                    logger.debug(msg)

                    theta_v_new_arr = (
                        _conversion.as_array(ffi, theta_v_new) if theta_v_new is not None else None
                    )
                    msg = "shape of theta_v_new after computation = %s" % str(
                        theta_v_new_arr.shape if theta_v_new is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "theta_v_new after computation: %s" % str(theta_v_new_arr)
                        if theta_v_new is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vn_now_arr = _conversion.as_array(ffi, vn_now) if vn_now is not None else None
                    msg = "shape of vn_now after computation = %s" % str(
                        vn_now_arr.shape if vn_now is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vn_now after computation: %s" % str(vn_now_arr)
                        if vn_now is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vn_new_arr = _conversion.as_array(ffi, vn_new) if vn_new is not None else None
                    msg = "shape of vn_new after computation = %s" % str(
                        vn_new_arr.shape if vn_new is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vn_new after computation: %s" % str(vn_new_arr)
                        if vn_new is not None
                        else "None"
                    )
                    logger.debug(msg)

                    w_concorr_c_arr = (
                        _conversion.as_array(ffi, w_concorr_c) if w_concorr_c is not None else None
                    )
                    msg = "shape of w_concorr_c after computation = %s" % str(
                        w_concorr_c_arr.shape if w_concorr_c is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "w_concorr_c after computation: %s" % str(w_concorr_c_arr)
                        if w_concorr_c is not None
                        else "None"
                    )
                    logger.debug(msg)

                    ddt_vn_apc_ntl1_arr = (
                        _conversion.as_array(ffi, ddt_vn_apc_ntl1)
                        if ddt_vn_apc_ntl1 is not None
                        else None
                    )
                    msg = "shape of ddt_vn_apc_ntl1 after computation = %s" % str(
                        ddt_vn_apc_ntl1_arr.shape if ddt_vn_apc_ntl1 is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "ddt_vn_apc_ntl1 after computation: %s" % str(ddt_vn_apc_ntl1_arr)
                        if ddt_vn_apc_ntl1 is not None
                        else "None"
                    )
                    logger.debug(msg)

                    ddt_vn_apc_ntl2_arr = (
                        _conversion.as_array(ffi, ddt_vn_apc_ntl2)
                        if ddt_vn_apc_ntl2 is not None
                        else None
                    )
                    msg = "shape of ddt_vn_apc_ntl2 after computation = %s" % str(
                        ddt_vn_apc_ntl2_arr.shape if ddt_vn_apc_ntl2 is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "ddt_vn_apc_ntl2 after computation: %s" % str(ddt_vn_apc_ntl2_arr)
                        if ddt_vn_apc_ntl2 is not None
                        else "None"
                    )
                    logger.debug(msg)

                    ddt_w_adv_ntl1_arr = (
                        _conversion.as_array(ffi, ddt_w_adv_ntl1)
                        if ddt_w_adv_ntl1 is not None
                        else None
                    )
                    msg = "shape of ddt_w_adv_ntl1 after computation = %s" % str(
                        ddt_w_adv_ntl1_arr.shape if ddt_w_adv_ntl1 is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "ddt_w_adv_ntl1 after computation: %s" % str(ddt_w_adv_ntl1_arr)
                        if ddt_w_adv_ntl1 is not None
                        else "None"
                    )
                    logger.debug(msg)

                    ddt_w_adv_ntl2_arr = (
                        _conversion.as_array(ffi, ddt_w_adv_ntl2)
                        if ddt_w_adv_ntl2 is not None
                        else None
                    )
                    msg = "shape of ddt_w_adv_ntl2 after computation = %s" % str(
                        ddt_w_adv_ntl2_arr.shape if ddt_w_adv_ntl2 is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "ddt_w_adv_ntl2 after computation: %s" % str(ddt_w_adv_ntl2_arr)
                        if ddt_w_adv_ntl2 is not None
                        else "None"
                    )
                    logger.debug(msg)

                    theta_v_ic_arr = (
                        _conversion.as_array(ffi, theta_v_ic) if theta_v_ic is not None else None
                    )
                    msg = "shape of theta_v_ic after computation = %s" % str(
                        theta_v_ic_arr.shape if theta_v_ic is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "theta_v_ic after computation: %s" % str(theta_v_ic_arr)
                        if theta_v_ic is not None
                        else "None"
                    )
                    logger.debug(msg)

                    rho_ic_arr = _conversion.as_array(ffi, rho_ic) if rho_ic is not None else None
                    msg = "shape of rho_ic after computation = %s" % str(
                        rho_ic_arr.shape if rho_ic is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "rho_ic after computation: %s" % str(rho_ic_arr)
                        if rho_ic is not None
                        else "None"
                    )
                    logger.debug(msg)

                    exner_pr_arr = (
                        _conversion.as_array(ffi, exner_pr) if exner_pr is not None else None
                    )
                    msg = "shape of exner_pr after computation = %s" % str(
                        exner_pr_arr.shape if exner_pr is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "exner_pr after computation: %s" % str(exner_pr_arr)
                        if exner_pr is not None
                        else "None"
                    )
                    logger.debug(msg)

                    exner_dyn_incr_arr = (
                        _conversion.as_array(ffi, exner_dyn_incr)
                        if exner_dyn_incr is not None
                        else None
                    )
                    msg = "shape of exner_dyn_incr after computation = %s" % str(
                        exner_dyn_incr_arr.shape if exner_dyn_incr is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "exner_dyn_incr after computation: %s" % str(exner_dyn_incr_arr)
                        if exner_dyn_incr is not None
                        else "None"
                    )
                    logger.debug(msg)

                    ddt_exner_phy_arr = (
                        _conversion.as_array(ffi, ddt_exner_phy)
                        if ddt_exner_phy is not None
                        else None
                    )
                    msg = "shape of ddt_exner_phy after computation = %s" % str(
                        ddt_exner_phy_arr.shape if ddt_exner_phy is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "ddt_exner_phy after computation: %s" % str(ddt_exner_phy_arr)
                        if ddt_exner_phy is not None
                        else "None"
                    )
                    logger.debug(msg)

                    grf_tend_rho_arr = (
                        _conversion.as_array(ffi, grf_tend_rho)
                        if grf_tend_rho is not None
                        else None
                    )
                    msg = "shape of grf_tend_rho after computation = %s" % str(
                        grf_tend_rho_arr.shape if grf_tend_rho is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "grf_tend_rho after computation: %s" % str(grf_tend_rho_arr)
                        if grf_tend_rho is not None
                        else "None"
                    )
                    logger.debug(msg)

                    grf_tend_thv_arr = (
                        _conversion.as_array(ffi, grf_tend_thv)
                        if grf_tend_thv is not None
                        else None
                    )
                    msg = "shape of grf_tend_thv after computation = %s" % str(
                        grf_tend_thv_arr.shape if grf_tend_thv is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "grf_tend_thv after computation: %s" % str(grf_tend_thv_arr)
                        if grf_tend_thv is not None
                        else "None"
                    )
                    logger.debug(msg)

                    grf_tend_w_arr = (
                        _conversion.as_array(ffi, grf_tend_w) if grf_tend_w is not None else None
                    )
                    msg = "shape of grf_tend_w after computation = %s" % str(
                        grf_tend_w_arr.shape if grf_tend_w is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "grf_tend_w after computation: %s" % str(grf_tend_w_arr)
                        if grf_tend_w is not None
                        else "None"
                    )
                    logger.debug(msg)

                    mass_fl_e_arr = (
                        _conversion.as_array(ffi, mass_fl_e) if mass_fl_e is not None else None
                    )
                    msg = "shape of mass_fl_e after computation = %s" % str(
                        mass_fl_e_arr.shape if mass_fl_e is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "mass_fl_e after computation: %s" % str(mass_fl_e_arr)
                        if mass_fl_e is not None
                        else "None"
                    )
                    logger.debug(msg)

                    ddt_vn_phy_arr = (
                        _conversion.as_array(ffi, ddt_vn_phy) if ddt_vn_phy is not None else None
                    )
                    msg = "shape of ddt_vn_phy after computation = %s" % str(
                        ddt_vn_phy_arr.shape if ddt_vn_phy is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "ddt_vn_phy after computation: %s" % str(ddt_vn_phy_arr)
                        if ddt_vn_phy is not None
                        else "None"
                    )
                    logger.debug(msg)

                    grf_tend_vn_arr = (
                        _conversion.as_array(ffi, grf_tend_vn) if grf_tend_vn is not None else None
                    )
                    msg = "shape of grf_tend_vn after computation = %s" % str(
                        grf_tend_vn_arr.shape if grf_tend_vn is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "grf_tend_vn after computation: %s" % str(grf_tend_vn_arr)
                        if grf_tend_vn is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vn_ie_arr = _conversion.as_array(ffi, vn_ie) if vn_ie is not None else None
                    msg = "shape of vn_ie after computation = %s" % str(
                        vn_ie_arr.shape if vn_ie is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vn_ie after computation: %s" % str(vn_ie_arr)
                        if vn_ie is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vt_arr = _conversion.as_array(ffi, vt) if vt is not None else None
                    msg = "shape of vt after computation = %s" % str(
                        vt_arr.shape if vt is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "vt after computation: %s" % str(vt_arr) if vt is not None else "None"
                    logger.debug(msg)

                    vn_incr_arr = (
                        _conversion.as_array(ffi, vn_incr) if vn_incr is not None else None
                    )
                    msg = "shape of vn_incr after computation = %s" % str(
                        vn_incr_arr.shape if vn_incr is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vn_incr after computation: %s" % str(vn_incr_arr)
                        if vn_incr is not None
                        else "None"
                    )
                    logger.debug(msg)

                    rho_incr_arr = (
                        _conversion.as_array(ffi, rho_incr) if rho_incr is not None else None
                    )
                    msg = "shape of rho_incr after computation = %s" % str(
                        rho_incr_arr.shape if rho_incr is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "rho_incr after computation: %s" % str(rho_incr_arr)
                        if rho_incr is not None
                        else "None"
                    )
                    logger.debug(msg)

                    exner_incr_arr = (
                        _conversion.as_array(ffi, exner_incr) if exner_incr is not None else None
                    )
                    msg = "shape of exner_incr after computation = %s" % str(
                        exner_incr_arr.shape if exner_incr is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "exner_incr after computation: %s" % str(exner_incr_arr)
                        if exner_incr is not None
                        else "None"
                    )
                    logger.debug(msg)

                    mass_flx_me_arr = (
                        _conversion.as_array(ffi, mass_flx_me) if mass_flx_me is not None else None
                    )
                    msg = "shape of mass_flx_me after computation = %s" % str(
                        mass_flx_me_arr.shape if mass_flx_me is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "mass_flx_me after computation: %s" % str(mass_flx_me_arr)
                        if mass_flx_me is not None
                        else "None"
                    )
                    logger.debug(msg)

                    mass_flx_ic_arr = (
                        _conversion.as_array(ffi, mass_flx_ic) if mass_flx_ic is not None else None
                    )
                    msg = "shape of mass_flx_ic after computation = %s" % str(
                        mass_flx_ic_arr.shape if mass_flx_ic is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "mass_flx_ic after computation: %s" % str(mass_flx_ic_arr)
                        if mass_flx_ic is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vol_flx_ic_arr = (
                        _conversion.as_array(ffi, vol_flx_ic) if vol_flx_ic is not None else None
                    )
                    msg = "shape of vol_flx_ic after computation = %s" % str(
                        vol_flx_ic_arr.shape if vol_flx_ic is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vol_flx_ic after computation: %s" % str(vol_flx_ic_arr)
                        if vol_flx_ic is not None
                        else "None"
                    )
                    logger.debug(msg)

                    vn_traj_arr = (
                        _conversion.as_array(ffi, vn_traj) if vn_traj is not None else None
                    )
                    msg = "shape of vn_traj after computation = %s" % str(
                        vn_traj_arr.shape if vn_traj is not None else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "vn_traj after computation: %s" % str(vn_traj_arr)
                        if vn_traj is not None
                        else "None"
                    )
                    logger.debug(msg)

                    max_vcfl_size1_array_arr = (
                        _conversion.as_array(ffi, max_vcfl_size1_array)
                        if max_vcfl_size1_array is not None
                        else None
                    )
                    msg = "shape of max_vcfl_size1_array after computation = %s" % str(
                        max_vcfl_size1_array_arr.shape
                        if max_vcfl_size1_array is not None
                        else "None"
                    )
                    logger.debug(msg)
                    msg = (
                        "max_vcfl_size1_array after computation: %s" % str(max_vcfl_size1_array_arr)
                        if max_vcfl_size1_array is not None
                        else "None"
                    )
                    logger.debug(msg)

            if __debug__:
                logger.info("Python execution of solve_nh_run completed.")

        except Exception as e:
            logger.exception(f"A Python error occurred: {e}")
            return 2

    return 1
