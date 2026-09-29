# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver import common, run_full_muphys
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.testing import test_utils
from icon4py.model.testing.fixtures.datatest import backend_like

from . import utils
from .utils import download_test_data


class Experiments:
    # TODO(havogt): the following references need to be checked (and moved to the shared directory),
    # currently they are not verifying
    # muphys/full_muphys/r2b04.tar.gz
    # muphys/full_muphys/r2b04_maxfrac.tar.gz
    # muphys/full_muphys/r2b05.tar.gz
    # Note: don't use the 'tiny' experiment from graupel_only,
    # as it is not sensitive to saturation adjustment
    # TODO(havogt): double-check that all other experiments actually are sensitive,
    # i.e. reference of full_muphys and graupel_only differ significantly.
    MINI: Final = utils.MuphysExperiment(
        name="mini",
        type=utils.ExperimentType.FULL_MUPHYS,
    )


@pytest.mark.uses_concat_where
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        Experiments.MINI,
        # TODO(havogt): references need to be checked, currently they are not verifying
        # Experiments.R2B04,
        # Experiments.R2B04_MAXFRAC,
        # Experiments.R2B05,
    ],
    ids=lambda exp: exp.name,
)
@pytest.mark.parametrize("single_program", [True, False], ids=lambda sp: f"single_program={sp}")
def test_full_muphys(
    backend_like: model_backends.BackendLike,
    experiment: utils.MuphysExperiment,
    single_program: bool,
) -> None:
    assert experiment.type == utils.ExperimentType.FULL_MUPHYS

    if single_program:
        pytest.xfail("Single program version currently fails verification. Needs investigation.")

    inp = common.GraupelInput.load(
        filename=experiment.input_file, allocator=model_backends.get_allocator(backend_like)
    )

    muphys_program = run_full_muphys.setup_muphys(
        inp=inp,
        dt=experiment.dt,
        qnc=experiment.qnc,
        backend=backend_like,
        single_program=single_program,
    )

    # We are passing the same buffers for `Q` as input and output. This is not best GT4Py practice,
    # but save in this case as we are not reading the input with an offset.
    out = common.GraupelOutput.allocate(
        allocator=model_backends.get_allocator(backend_like),
        domain=gtx.domain({dims.CellDim: inp.ncells, dims.KDim: inp.nlev}),
        references={
            "qv": inp.qv,
            "qc": inp.qc,
            "qi": inp.qi,
            "qr": inp.qr,
            "qs": inp.qs,
            "qg": inp.qg,
            "t": inp.t,
        },
    )

    muphys_program(
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

    ref = common.GraupelOutput.load(
        filename=experiment.reference_file, allocator=model_backends.get_allocator(backend_like)
    )

    rtol = 1e-14
    atol = 1e-16

    test_utils.assert_dallclose(ref.qv.asnumpy(), out.qv.asnumpy(), atol=atol, rtol=rtol)
    test_utils.assert_dallclose(ref.qc.asnumpy(), out.qc.asnumpy(), atol=atol, rtol=rtol)
    test_utils.assert_dallclose(ref.qi.asnumpy(), out.qi.asnumpy(), atol=atol, rtol=rtol)
    test_utils.assert_dallclose(ref.qr.asnumpy(), out.qr.asnumpy(), atol=atol, rtol=rtol)
    test_utils.assert_dallclose(ref.qs.asnumpy(), out.qs.asnumpy(), atol=atol, rtol=rtol)
    test_utils.assert_dallclose(ref.qg.asnumpy(), out.qg.asnumpy(), atol=atol, rtol=rtol)
    test_utils.assert_dallclose(ref.t.asnumpy(), out.t.asnumpy(), atol=atol, rtol=rtol)
