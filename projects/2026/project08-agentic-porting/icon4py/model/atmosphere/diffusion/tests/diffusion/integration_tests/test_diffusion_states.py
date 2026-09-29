# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.diffusion import diffusion_states

from ..fixtures import *  # noqa: F403


@pytest.mark.datatest
def test_verify_geofac_n2s_field_manipulation(interpolation_savepoint, icon_grid, backend):
    geofac_n2s = interpolation_savepoint.geofac_n2s().asnumpy()
    interpolation_state = diffusion_states.DiffusionInterpolationState(
        e_bln_c_s=interpolation_savepoint.e_bln_c_s(),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=interpolation_savepoint.geofac_div(),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    geofac_c = interpolation_state.geofac_n2s_c.asnumpy()
    geofac_nbh = interpolation_state.geofac_n2s_nbh.asnumpy()
    assert np.count_nonzero(geofac_nbh) > 0
    assert np.allclose(geofac_c, geofac_n2s[:, 0])
    assert np.allclose(geofac_nbh, geofac_n2s[:, 1:])
