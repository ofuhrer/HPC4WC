# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.testing import serialbox as sb, test_utils


def verify_diffusion_fields(
    config: diffusion.DiffusionConfig,
    diagnostic_state: diffusion_states.DiffusionDiagnosticState,
    prognostic_state: prognostics.PrognosticState,
    diffusion_savepoint: sb.IconDiffusionExitSavepoint,
):
    ref_w = diffusion_savepoint.w().asnumpy()
    val_w = prognostic_state.w.asnumpy()
    ref_exner = diffusion_savepoint.exner().asnumpy()
    ref_theta_v = diffusion_savepoint.theta_v().asnumpy()
    val_theta_v = prognostic_state.theta_v.asnumpy()
    val_exner = prognostic_state.exner.asnumpy()
    ref_vn = diffusion_savepoint.vn().asnumpy()
    val_vn = prognostic_state.vn.asnumpy()

    validate_diagnostics = (
        config.shear_type
        >= diffusion.TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND
    )
    if validate_diagnostics:
        ref_div_ic = diffusion_savepoint.div_ic().asnumpy()
        val_div_ic = diagnostic_state.div_ic.asnumpy()
        ref_hdef_ic = diffusion_savepoint.hdef_ic().asnumpy()
        val_hdef_ic = diagnostic_state.hdef_ic.asnumpy()
        ref_dwdx = diffusion_savepoint.dwdx().asnumpy()
        val_dwdx = diagnostic_state.dwdx.asnumpy()
        ref_dwdy = diffusion_savepoint.dwdy().asnumpy()
        val_dwdy = diagnostic_state.dwdy.asnumpy()

        assert test_utils.dallclose(val_div_ic, ref_div_ic, atol=1e-16)
        assert test_utils.dallclose(val_hdef_ic, ref_hdef_ic, atol=1e-13)
        assert test_utils.dallclose(val_dwdx, ref_dwdx, atol=1e-18)
        assert test_utils.dallclose(val_dwdy, ref_dwdy, atol=1e-18)

    assert test_utils.dallclose(val_vn, ref_vn, atol=1.0e-8, rtol=1.0e-9)
    assert test_utils.dallclose(val_w, ref_w, atol=1e-14)
    assert test_utils.dallclose(val_theta_v, ref_theta_v)
    assert test_utils.dallclose(val_exner, ref_exner)


def smag_limit_numpy(func, *args):
    return 0.125 - 4.0 * func(*args)


def diff_multfac_vn_numpy(shape, k4, substeps):
    factor = min(1.0 / 128.0, k4 * substeps / 3.0)
    return np.full(shape, factor)
