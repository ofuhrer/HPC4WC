#!/usr/bin/env python
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import copy
import pathlib
import time

from gt4py import next as gtx
from gt4py.next import config as gtx_config
from gt4py.next.instrumentation import metrics as gtx_metrics
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import common, utils
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations import (
    graupel,
    graupel_dace_hooks,
)
from icon4py.model.common import dimension as dims, model_backends, model_options, type_alias as ta
from icon4py.model.common.utils import device_utils


# TODO(havogt): make similar to icon4py driver structure


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", metavar="output_file", dest="output_file", help="output filename", default="output.nc"
    )
    parser.add_argument(
        "-b", metavar="backend", dest="backend", help="gt4py backend", default="gtfn_cpu"
    )
    parser.add_argument("input_file", help="input data file")
    parser.add_argument("itime", help="time-index", nargs="?", default=0)
    parser.add_argument("dt", help="timestep", nargs="?", default=30.0)
    parser.add_argument("qnc", help="Water number concentration", nargs="?", default=100.0)
    parser.add_argument(
        "-m",
        "--masking",
        dest="enable_masking",
        choices=[True, False],
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Enable compatibility with reference implementation.",
    )

    return parser.parse_args()


def setup_graupel(
    *,
    dt: float,
    qnc: float,
    backend: model_backends.BackendLike,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
    vertical_end: int,
    enable_masking: bool = True,
    enable_dace_hooks: bool = True,
):
    if enable_dace_hooks:
        assert model_backends.is_backend_descriptor(backend)
        backend = copy.deepcopy(backend)
        if "optimization_args" not in backend:
            backend["optimization_args"] = {}
        backend["optimization_args"]["optimization_hooks"] = {
            gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPre: graupel_dace_hooks.remove_self_copy_inside_scan,
            gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowPost: graupel_dace_hooks.rename_intermediate_access_nodes,
        }
    with utils.recursion_limit(10**4):  # TODO(havogt): make an option in gt4py?
        graupel_run_program = model_options.setup_program(
            backend=backend,
            program=graupel.graupel_run,
            constant_args={
                "dt": ta.wpfloat(dt),
                "qnc": ta.wpfloat(qnc),
                "enable_masking": enable_masking,
            },
            horizontal_sizes={
                "horizontal_start": horizontal_start,
                "horizontal_end": horizontal_end,
            },
            vertical_sizes={
                "vertical_start": vertical_start,
                "vertical_end": vertical_end,
            },
            offset_provider={},
        )
        gtx.wait_for_compilation()
        return graupel_run_program


def main():
    args = get_args()

    backend = model_backends.BACKENDS[args.backend]
    allocator = model_backends.get_allocator(backend)
    dtype = gtx.float32 if ta.precision == "single" else gtx.float64

    inp = common.GraupelInput.load(
        filename=pathlib.Path(args.input_file), allocator=allocator, dtype=dtype
    )

    use_inout_buffers = True  # Set to True to reuse input buffers for output.
    if use_inout_buffers:
        # We are passing the same buffers for `Q` as input and output. This is not best GT4Py practice,
        # but should be safe in this case as we are not reading the input with an offset.
        references = {
            "qv": inp.qv,
            "qc": inp.qc,
            "qi": inp.qi,
            "qr": inp.qr,
            "qs": inp.qs,
            "qg": inp.qg,
            "t": inp.t,
        }
    else:
        references = None

    out = common.GraupelOutput.allocate(
        allocator=allocator,
        domain=gtx.domain({dims.CellDim: inp.ncells, dims.KDim: inp.nlev}),
        dtype=dtype,
        references=references,
    )

    graupel_run_program = setup_graupel(
        dt=args.dt,
        qnc=args.qnc,
        backend=backend,
        horizontal_start=0,
        horizontal_end=inp.ncells,
        vertical_start=0,
        vertical_end=inp.nlev,
        enable_masking=args.enable_masking,
    )

    start_time = None
    for _x in range(int(args.itime) + 1):
        if _x == 1:  # Only start timing second iteration
            device_utils.sync(allocator)
            start_time = time.time()

        graupel_run_program(
            dz=inp.dz,
            te=inp.t,
            p=inp.p,
            rho=inp.rho,
            q_in=inp.q,
            t_out=out.t,
            q_out=out.q,
            pflx=out.pflx,
            pr=out.pr,
            ps=out.ps,
            pi=out.pi,
            pg=out.pg,
            pre=out.pre,
        )
    device_utils.sync(allocator)
    end_time = time.time()

    if start_time is not None:
        elapsed_time = end_time - start_time
        print("For", int(args.itime), "iterations it took", elapsed_time, "seconds!")

    if gtx_config.COLLECT_METRICS_LEVEL > 0:
        print(gtx_metrics.dumps())
        gtx_metrics.dump_json("gt4py_metrics.json")

    out.write(args.output_file)


if __name__ == "__main__":
    main()
