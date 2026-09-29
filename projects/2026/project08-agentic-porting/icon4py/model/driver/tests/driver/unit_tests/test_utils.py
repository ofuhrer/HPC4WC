# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools

import numpy as np

from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.driver.testcases import utils
from icon4py.model.testing import test_utils
from icon4py.model.testing.fixtures import backend

from ..fixtures import *  # noqa: F403


def test_hydrostatic_adjustment_ndarray(backend):
    # TODO(jcanton,OngChia): these tests could be better
    num_cells = 10
    num_levels = 10

    xp = data_alloc.import_array_ns(backend)

    # Generate test data
    rho0 = 1.25
    exner0 = 0.935
    theta_v0 = 293.14
    wgtfac_c = xp.full((num_cells, num_levels), 1.05)
    ddqz_z_half = xp.ones((num_cells, num_levels))
    exner_ref_mc = xp.full((num_cells, num_levels), 0.89)
    d_exner_dz_ref_ic = xp.zeros((num_cells, num_levels))
    theta_ref_mc = xp.full((num_cells, num_levels), 312)
    theta_ref_ic = xp.full((num_cells, num_levels), 312)
    rho = xp.full((num_cells, num_levels), rho0)
    exner = xp.full((num_cells, num_levels), exner0)
    theta_v = xp.full((num_cells, num_levels), theta_v0)

    # Call the function
    r_rho, r_exner, r_theta_v = functools.partial(utils.hydrostatic_adjustment_ndarray)(
        wgtfac_c=wgtfac_c,
        ddqz_z_half=ddqz_z_half,
        exner_ref_mc=exner_ref_mc,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        theta_ref_mc=theta_ref_mc,
        theta_ref_ic=theta_ref_ic,
        rho=rho,
        exner=exner,
        theta_v=theta_v,
        num_levels=num_levels,
    )

    assert r_rho.shape == (num_cells, num_levels)
    assert r_exner.shape == (num_cells, num_levels)
    assert r_theta_v.shape == (num_cells, num_levels)

    assert test_utils.dallclose(
        r_rho[:, -1],
        np.full(num_cells, rho0),
    )
    assert test_utils.dallclose(
        data_alloc.as_numpy(r_rho[:, :-1]),
        np.full((num_cells, num_levels - 1), 1.0046424441749071),
    )
    assert test_utils.dallclose(
        data_alloc.as_numpy(r_exner),
        np.full((num_cells, num_levels), exner0),
    )
    assert test_utils.dallclose(
        data_alloc.as_numpy(r_theta_v),
        np.full((num_cells, num_levels), theta_v0),
    )


def test_hydrostatic_adjustment_constant_thetav_ndarray(backend):
    # TODO(jcanton,OngChia): these tests could be better
    num_cells = 10
    num_levels = 10

    xp = data_alloc.import_array_ns(backend)

    # Generate test data
    rho0 = 1.25
    exner0 = 0.935
    theta_v0 = 293.14
    wgtfac_c = xp.full((num_cells, num_levels), 1.05)
    ddqz_z_half = xp.ones((num_cells, num_levels))
    exner_ref_mc = xp.full((num_cells, num_levels), 0.89)
    d_exner_dz_ref_ic = xp.zeros((num_cells, num_levels))
    theta_ref_mc = xp.full((num_cells, num_levels), 312)
    theta_ref_ic = xp.full((num_cells, num_levels), 312)
    rho = xp.full((num_cells, num_levels), rho0)
    exner = xp.full((num_cells, num_levels), exner0)
    theta_v = xp.full((num_cells, num_levels), theta_v0)

    # Call the function
    r_rho, r_exner = utils.hydrostatic_adjustment_constant_thetav_ndarray(
        wgtfac_c=wgtfac_c,
        ddqz_z_half=ddqz_z_half,
        exner_ref_mc=exner_ref_mc,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        theta_ref_mc=theta_ref_mc,
        theta_ref_ic=theta_ref_ic,
        rho=rho,
        exner=exner,
        theta_v=theta_v,
        num_levels=num_levels,
    )

    assert r_rho.shape == (num_cells, num_levels)
    assert r_exner.shape == (num_cells, num_levels)

    assert test_utils.dallclose(
        data_alloc.as_numpy(r_rho),
        np.full((num_cells, num_levels), 1.0046424441749071),
    )
    assert test_utils.dallclose(
        data_alloc.as_numpy(r_exner),
        np.full((num_cells, num_levels), exner0),
    )
