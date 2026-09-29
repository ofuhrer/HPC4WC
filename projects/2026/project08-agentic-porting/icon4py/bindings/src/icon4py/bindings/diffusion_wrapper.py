# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


"""
Wrapper module for diffusion granule.

Module contains a diffusion_init and diffusion_run function that follow the architecture of
Fortran granule interfaces:
- all arguments needed from external sources are passed.
- passing of scalar types or fields of simple types
"""

import dataclasses
import logging
from collections.abc import Callable

import gt4py.next as gtx
import numpy as np

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.bindings import (
    common as wrapper_common,
    config as wrapper_config,
    grid_wrapper,
    icon4py_export,
)
from icon4py.model.atmosphere.diffusion.diffusion import (
    Diffusion,
    DiffusionConfig,
    DiffusionParams,
    DiffusionType,
    ForcingType,
    SmagorinskyStencilType,
    TemperatureDiscretizationType,
    TurbulenceShearForcingType,
)
from icon4py.model.atmosphere.diffusion.diffusion_states import (
    DiffusionDiagnosticState,
    DiffusionInterpolationState,
    DiffusionMetricState,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, model_backends
from icon4py.model.common.states.prognostic_state import PrognosticState
from icon4py.model.common.type_alias import wpfloat


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DiffusionGranule:
    diffusion: Diffusion
    dummy_field_factory: Callable


granule: DiffusionGranule | None = None


@icon4py_export.export
def diffusion_init(  # noqa: PLR0917 [too-many-positional-arguments]
    theta_ref_mc: fa.CellKField[wpfloat],
    wgtfac_c: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.float64],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.float64],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], gtx.float64],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    rbf_vec_coeff_v: wrapper_common.Float64Array3D,
    zd_cellidx: wrapper_common.OptionalInt32Array2D,
    zd_vertidx: wrapper_common.OptionalInt32Array2D,
    zd_intcoef: wrapper_common.OptionalFloat64Array2D,
    zd_diffcoef: wrapper_common.OptionalFloat64Array1D,
    ndyn_substeps: gtx.int32,
    diffusion_type: gtx.int32,
    hdiff_w: bool,
    hdiff_vn: bool,
    hdiff_smag_w: bool,
    zdiffu_t: bool,
    type_t_diffu: gtx.int32,
    type_vn_diffu: gtx.int32,
    hdiff_efdt_ratio: gtx.float64,
    hdiff_w_efdt_ratio: gtx.float64,
    smagorinski_scaling_factor: gtx.float64,
    smagorinski_scaling_factor2: gtx.float64,
    smagorinski_scaling_factor3: gtx.float64,
    smagorinski_scaling_factor4: gtx.float64,
    smagorinski_scaling_height: gtx.float64,
    smagorinski_scaling_height2: gtx.float64,
    smagorinski_scaling_height3: gtx.float64,
    smagorinski_scaling_height4: gtx.float64,
    hdiff_temp: bool,
    denom_diffu_v: float,
    nudge_max_coeff: float,  # note: this is the scaled ICON value, i.e. not the namelist value
    itype_sher: gtx.int32,
    iforcing: gtx.int32,
    a_hshr: gtx.float64,
    loutshs: bool,
    backend: gtx.int32,
):
    if grid_wrapper.grid_state is None:
        raise Exception(
            "Need to initialise grid using 'grid_init' before running 'diffusion_init'."
        )

    xp = theta_ref_mc.array_ns
    on_gpu = xp != np  # TODO(havogt): expose `on_gpu` from py2fgen
    actual_backend = wrapper_common.select_backend(
        wrapper_common.BackendIntEnum(backend), on_gpu=on_gpu
    )
    backend_name = actual_backend.name if hasattr(actual_backend, "name") else actual_backend
    logger.info(f"Using Backend {backend_name} with on_gpu={on_gpu}")
    allocator = model_backends.get_allocator(actual_backend)

    # Diffusion parameters
    config = DiffusionConfig(
        diffusion_type=DiffusionType(diffusion_type),
        apply_to_vertical_wind=hdiff_w,
        apply_to_horizontal_wind=hdiff_vn,
        apply_smag_diff_to_vertical_wind=hdiff_smag_w,
        apply_zdiffusion_t=zdiffu_t,
        type_t_diffu=TemperatureDiscretizationType(type_t_diffu),
        type_vn_diffu=SmagorinskyStencilType(type_vn_diffu),
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
        apply_to_temperature=hdiff_temp,
        ndyn_substeps=int(ndyn_substeps),
        velocity_boundary_diffusion_denominator=denom_diffu_v,
        max_nudging_coefficient=nudge_max_coeff,
        shear_type=TurbulenceShearForcingType(itype_sher),
        iforcing=ForcingType(iforcing),
        a_hshr=a_hshr,
        loutshs=loutshs,
    )

    diffusion_params = DiffusionParams(config)

    nlev = wgtfac_c.domain[dims.KDim].unit_range.stop - 1  # wgtfac_c has nlevp1 levels
    cell_k_domain = gtx.domain(
        {dims.CellDim: wgtfac_c.domain[dims.CellDim].unit_range, dims.KDim: nlev}
    )
    c2e2c_size = geofac_grg_x.domain[dims.C2E2CODim].unit_range.stop - 1
    cell_c2e2c_k_domain = gtx.domain(
        {
            dims.CellDim: wgtfac_c.domain[dims.CellDim].unit_range,
            dims.C2E2CDim: c2e2c_size,
            dims.KDim: nlev,
        }
    )

    if zd_cellidx is None:
        # then zdiffu_t is False or the list on that rank is empty, then all of the following are not initialized
        assert zd_vertidx is None and zd_intcoef is None and zd_diffcoef is None
        zd_diffcoef = gtx.zeros(cell_k_domain, dtype=theta_ref_mc.dtype, allocator=allocator)
        zd_intcoef = gtx.zeros(cell_c2e2c_k_domain, dtype=wgtfac_c.dtype, allocator=allocator)
        zd_vertoffset = gtx.zeros(cell_c2e2c_k_domain, dtype=xp.int32, allocator=allocator)
    else:
        # transform lists to fields

        # only the first row is needed, the others are for C2E2C neighbors, but slicing in fortran causes issues
        zd_cellidx = zd_cellidx[0, :]
        # these are the three k offsets for the C2E2C neighbors
        zd_vertoffset = zd_vertidx[1:, :] - zd_vertidx[0, :]
        # this is the k list (with fortran 1-based indexing) for the central point of the C2E2C stencil
        zd_vertidx = zd_vertidx[0, :]

        zd_diffcoef = data_alloc.list2field(
            domain=cell_k_domain,
            values=zd_diffcoef,
            indices=(
                data_alloc.adjust_fortran_indices(zd_cellidx),
                data_alloc.adjust_fortran_indices(zd_vertidx),
            ),
            default_value=gtx.float64(0.0),
            allocator=allocator,
        )
        zd_intcoef = data_alloc.list2field(
            domain=cell_c2e2c_k_domain,
            values=zd_intcoef.T,
            indices=(
                data_alloc.adjust_fortran_indices(zd_cellidx),
                slice(None),
                data_alloc.adjust_fortran_indices(zd_vertidx),
            ),
            default_value=gtx.float64(0.0),
            allocator=allocator,
        )
        zd_vertoffset = data_alloc.list2field(
            domain=cell_c2e2c_k_domain,
            values=zd_vertoffset.T,
            indices=(
                data_alloc.adjust_fortran_indices(zd_cellidx),
                slice(None),
                data_alloc.adjust_fortran_indices(zd_vertidx),
            ),
            default_value=gtx.int32(0),
            allocator=allocator,
        )

    # Metric state
    metric_state = DiffusionMetricState(
        theta_ref_mc=theta_ref_mc,
        wgtfac_c=wgtfac_c,
        zd_intcoef=zd_intcoef,
        zd_vertoffset=zd_vertoffset,
        zd_diffcoef=zd_diffcoef,
    )

    # Create separate fields for the two components of the RBF vector coefficients and swap.
    # TODO(havogt): we could use GT4Py's named collections.
    rbf_coeff_1 = gtx.as_field(
        [dims.VertexDim, dims.V2EDim], xp.transpose(rbf_vec_coeff_v[:, 0, :]), allocator=allocator
    )
    rbf_coeff_2 = gtx.as_field(
        [dims.VertexDim, dims.V2EDim], xp.transpose(rbf_vec_coeff_v[:, 1, :]), allocator=allocator
    )

    # Interpolation state
    interpolation_state = DiffusionInterpolationState(
        e_bln_c_s=e_bln_c_s,
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        geofac_div=geofac_div,
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        nudgecoeff_e=nudgecoeff_e,
    )

    # Initialize the diffusion granule
    global granule  # noqa: PLW0603 [global-statement]
    granule = DiffusionGranule(
        diffusion=Diffusion(
            grid=grid_wrapper.grid_state.grid,
            config=config,
            params=diffusion_params,
            vertical_grid=grid_wrapper.grid_state.vertical_grid,
            metric_state=metric_state,
            interpolation_state=interpolation_state,
            edge_params=grid_wrapper.grid_state.edge_geometry,
            cell_params=grid_wrapper.grid_state.cell_geometry,
            backend=actual_backend,
            exchange=grid_wrapper.grid_state.exchange_runtime,
        ),
        dummy_field_factory=wrapper_common.cached_dummy_field_factory(allocator),
    )
    if wrapper_config.WAIT_FOR_COMPILATION:
        gtx.wait_for_compilation()


@icon4py_export.export
def diffusion_run(  # noqa: PLR0917 [too-many-positional-arguments]
    w: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64],
    vn: fa.EdgeKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    rho: fa.CellKField[wpfloat],
    hdef_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    div_ic: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    dwdx: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    dwdy: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], gtx.float64] | None,
    dtime: gtx.float64,
    linit: bool,
):
    if granule is None:
        raise RuntimeError("Diffusion granule not initialized. Call 'diffusion_init' first.")

    # prognostic and diagnostic variables
    prognostic_state = PrognosticState(
        w=w,
        vn=vn,
        exner=exner,
        theta_v=theta_v,
        rho=rho,
    )

    if hdef_ic is None:
        hdef_ic = granule.dummy_field_factory("hdef_ic", domain=w.domain, dtype=w.dtype)
    if div_ic is None:
        div_ic = granule.dummy_field_factory("div_ic", domain=w.domain, dtype=w.dtype)
    if dwdx is None:
        dwdx = granule.dummy_field_factory("dwdx", domain=w.domain, dtype=w.dtype)
    if dwdy is None:
        dwdy = granule.dummy_field_factory("dwdy", domain=w.domain, dtype=w.dtype)
    diagnostic_state = DiffusionDiagnosticState(
        hdef_ic=hdef_ic,
        div_ic=div_ic,
        dwdx=dwdx,
        dwdy=dwdy,
    )

    granule.diffusion.run(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        dtime=dtime,
        initial_run=linit,
    )
