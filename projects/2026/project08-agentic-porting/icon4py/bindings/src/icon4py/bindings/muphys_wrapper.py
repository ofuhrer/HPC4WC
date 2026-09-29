# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from gt4py import next as gtx
from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.bindings import icon4py_export
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import run_graupel_only
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations import graupel
from icon4py.model.common import dimension as dims, model_backends, type_alias as ta


graupel_program = None


@icon4py_export.export
def graupel_run(  # noqa: PLR0917 [too-many-positional-arguments]
    ke: gtx.int32,
    ivstart: gtx.int32,
    ivend: gtx.int32,
    kstart: gtx.int32,
    dt: ta.wpfloat,
    dz: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    t: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    rho: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    p: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qv: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qc: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qi: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qr: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qs: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qg: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    qnc: ta.wpfloat,
    prr_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    pri_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    prs_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    prg_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    pflx: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    pre_gsp: gtx.Field[gtx.Dims[dims.CellDim, dims.KDim], ta.wpfloat],
    enable_masking: bool,
    enable_dace_hooks: bool,
    wait_for_completion: bool,
) -> None:
    global graupel_program  # noqa: PLW0603 [global-statement]
    if graupel_program is None:
        backend_descriptor = {
            "backend_factory": model_backends.make_custom_dace_backend,
            "device": model_backends.CPU if t.array_ns == np else model_backends.GPU,  # type: ignore[attr-defined]  # to be fixed in gt4py
            "async_sdfg_call": not wait_for_completion,
        }
        graupel_program = run_graupel_only.setup_graupel(
            dt=dt,
            qnc=qnc,
            backend=backend_descriptor,
            horizontal_start=ivstart,
            horizontal_end=ivend,
            vertical_start=kstart,
            vertical_end=ke,
            enable_masking=enable_masking,
            enable_dace_hooks=enable_dace_hooks,
        )

    q = graupel.Q(qv, qc, qr, qs, qi, qg)

    # The precipitation fields (pr, ps, pi, pg, pre) are defined as 1D-fields with
    # horizontal domain, in the driver, because they represent precipitation at the
    # surface level. The graupel program requires a 2D-domain, with a single vertical
    # level corresponding to the surface level. The vertcal shift below is to set
    # the field origin at the surface level, as required by the program.
    graupel_program(
        dz=dz,
        te=t,
        p=p,
        rho=rho,
        q_in=q,
        t_out=t,
        q_out=q,
        pflx=pflx,
        pr=prr_gsp(dims.KDim - (ke - 1)),
        ps=prs_gsp(dims.KDim - (ke - 1)),
        pi=pri_gsp(dims.KDim - (ke - 1)),
        pg=prg_gsp(dims.KDim - (ke - 1)),
        pre=pre_gsp(dims.KDim - (ke - 1)),
    )


@icon4py_export.export
def graupel_finalize() -> None:
    # The atexit function is not called when embedding cpython into another application
    # with cffi, so we call it explicitly here to dump the metrics.
    gtx_metrics._dump_metrics_at_exit()
