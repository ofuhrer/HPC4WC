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
import functools
import pathlib
import time
from collections.abc import Callable

from gt4py import next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core import saturation_adjustment
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import (
    common,
    run_graupel_only,
    utils,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations import muphys
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    model_options,
    type_alias as ta,
)
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

    return parser.parse_args()


def _muphys_step_separate(
    *,
    graupel_program: Callable,
    saturation_adjustment_program: Callable,
    dz: fa.CellKField[ta.wpfloat],
    te: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Pressure
    rho: fa.CellKField[ta.wpfloat],  # Density containing dry air and water constituents
    q_in: common.Q,
    q_out: common.Q,
    t_out: fa.CellKField[ta.wpfloat],  # Revised temperature
    pflx: fa.CellKField[ta.wpfloat],  # Total precipitation flux
    pr: fa.CellKField[ta.wpfloat],  # Precipitation of rain
    ps: fa.CellKField[ta.wpfloat],  # Precipitation of snow
    pi: fa.CellKField[ta.wpfloat],  # Precipitation of ice
    pg: fa.CellKField[ta.wpfloat],  # Precipitation of graupel
    pre: fa.CellKField[ta.wpfloat],  # Precipitation of graupel
):
    # In-place update ok since saturation_adjustment is fully point-wise,
    # but not recommended. TODO
    saturation_adjustment_program(
        te=te,
        q_in=q_in,
        rho=rho,
        te_out=te,
        qve_out=q_in.v,
        qce_out=q_in.c,
    )

    graupel_program(
        dz=dz,
        te=te,
        p=p,
        rho=rho,
        q_in=q_in,
        t_out=t_out,
        q_out=q_out,
        pflx=pflx,
        pr=pr,
        ps=ps,
        pi=pi,
        pg=pg,
        pre=pre,
    )

    saturation_adjustment_program(
        te=t_out,
        q_in=q_out,
        rho=rho,
        te_out=t_out,
        qve_out=q_out.v,
        qce_out=q_out.c,
    )


def setup_muphys(
    inp: common.GraupelInput,
    dt: float,
    qnc: float,
    backend: model_backends.BackendLike,
    *,
    single_program: bool = False,
):
    if single_program:
        # TODO(havogt): make an option in gt4py for thread-safety?
        with utils.recursion_limit(10**5):
            muphys_program = model_options.setup_program(
                backend=backend,
                program=muphys.muphys_run,
                constant_args={"dt": ta.wpfloat(dt), "qnc": ta.wpfloat(qnc)},
                horizontal_sizes={
                    "horizontal_start": gtx.int32(0),
                    "horizontal_end": inp.ncells,
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(0),
                    "vertical_end": gtx.int32(inp.nlev),
                },
                offset_provider={},
            )
            gtx.wait_for_compilation()
            return muphys_program
    else:
        graupel_run_program = run_graupel_only.setup_graupel(
            dt=dt,
            qnc=qnc,
            backend=backend,
            horizontal_start=0,
            horizontal_end=inp.ncells,
            vertical_start=0,
            vertical_end=inp.nlev,
            enable_masking=True,
        )
        with utils.recursion_limit(10**5):  # TODO(havogt): make an option in gt4py?
            saturation_adjustment_program = model_options.setup_program(
                backend=backend,
                program=saturation_adjustment.saturation_adjustment,
                horizontal_sizes={
                    "horizontal_start": gtx.int32(0),
                    "horizontal_end": inp.ncells,
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(0),
                    "vertical_end": gtx.int32(inp.nlev),
                },
            )
            gtx.wait_for_compilation()

        return functools.partial(
            _muphys_step_separate,
            graupel_program=graupel_run_program,
            saturation_adjustment_program=saturation_adjustment_program,
        )


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
        domain=gtx.domain({dims.CellDim: inp.ncells, dims.KDim: inp.nlev}),
        allocator=allocator,
        dtype=dtype,
        references=references,
    )

    # TODO(havogt): once we see single program being equally fast, remove the other implementation
    muphys_step = setup_muphys(
        inp=inp, dt=args.dt, qnc=args.qnc, backend=backend, single_program=False
    )

    start_time = None
    for _x in range(int(args.itime) + 1):
        if _x == 1:  # Only start timing second iteration
            device_utils.sync(allocator)
            start_time = time.time()

        muphys_step(
            dz=inp.dz,
            te=inp.t,
            p=inp.p,
            rho=inp.rho,
            q_in=inp.q,
            q_out=out.q,
            t_out=out.t,
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

    out.write(args.output_file)


if __name__ == "__main__":
    main()
