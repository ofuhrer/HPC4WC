# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from icon4py.model.atmosphere.diffusion import diffusion_utils
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import simple as simple_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.fixtures import backend, backend_like

from ..utils import diff_multfac_vn_numpy, smag_limit_numpy


# TODO(havogt): apply StencilTest structure, however this needs to be executed with CartesianGrid,
# i.e. must not receive connectivities.


def initial_diff_multfac_vn_numpy(shape, k4, hdiff_efdt_ratio):
    return np.full(shape, k4 * hdiff_efdt_ratio / 3.0)


def test_scale_k(backend):
    grid = simple_grid.simple_grid(allocator=backend)
    field = data_alloc.random_field(grid, dims.KDim, allocator=backend)
    scaled_field = data_alloc.zero_field(grid, dims.KDim, allocator=backend)
    factor = 2.0
    diffusion_utils.scale_k.with_backend(backend)(field, factor, scaled_field, offset_provider={})
    assert np.allclose(factor * field.asnumpy(), scaled_field.asnumpy())


def test_diff_multfac_vn_and_smag_limit_for_initial_step(backend):
    grid = simple_grid.simple_grid(allocator=backend)
    diff_multfac_vn_init = data_alloc.zero_field(grid, dims.KDim, allocator=backend)
    smag_limit_init = data_alloc.zero_field(grid, dims.KDim, allocator=backend)
    k4 = 1.0
    efdt_ratio = 24.0
    shape = diff_multfac_vn_init.asnumpy().shape

    expected_diff_multfac_vn_init = initial_diff_multfac_vn_numpy(shape, k4, efdt_ratio)
    expected_smag_limit_init = smag_limit_numpy(
        initial_diff_multfac_vn_numpy, shape, k4, efdt_ratio
    )

    diffusion_utils.setup_fields_for_initial_step.with_backend(backend)(
        k4, efdt_ratio, diff_multfac_vn_init, smag_limit_init, offset_provider={}
    )

    assert np.allclose(expected_diff_multfac_vn_init, diff_multfac_vn_init.asnumpy())
    assert np.allclose(expected_smag_limit_init, smag_limit_init.asnumpy())


def test_diff_multfac_vn_smag_limit_for_time_step_with_const_value(backend):
    grid = simple_grid.simple_grid(allocator=backend)
    diff_multfac_vn = data_alloc.zero_field(grid, dims.KDim, allocator=backend)
    smag_limit = data_alloc.zero_field(grid, dims.KDim, allocator=backend)
    k4 = 1.0
    substeps = 5.0
    efdt_ratio = 24.0
    shape = diff_multfac_vn.asnumpy().shape

    expected_diff_multfac_vn = diff_multfac_vn_numpy(shape, k4, substeps)
    expected_smag_limit = smag_limit_numpy(diff_multfac_vn_numpy, shape, k4, substeps)

    diffusion_utils._setup_runtime_diff_multfac_vn.with_backend(backend)(
        k4, efdt_ratio, out=diff_multfac_vn, offset_provider={}
    )
    diffusion_utils._setup_smag_limit.with_backend(backend)(
        diff_multfac_vn, out=smag_limit, offset_provider={}
    )

    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn.asnumpy())
    assert np.allclose(expected_smag_limit, smag_limit.asnumpy())


def test_diff_multfac_vn_smag_limit_for_loop_run_with_k4_substeps(backend):
    grid = simple_grid.simple_grid(allocator=backend)
    diff_multfac_vn = data_alloc.zero_field(grid, dims.KDim, allocator=backend)
    smag_limit = data_alloc.zero_field(grid, dims.KDim, allocator=backend)
    k4 = 0.003
    substeps = 1.0

    shape = diff_multfac_vn.asnumpy().shape
    expected_diff_multfac_vn = diff_multfac_vn_numpy(shape, k4, substeps)
    expected_smag_limit = smag_limit_numpy(diff_multfac_vn_numpy, shape, k4, substeps)
    diffusion_utils._setup_runtime_diff_multfac_vn.with_backend(backend)(
        k4, substeps, out=diff_multfac_vn, offset_provider={}
    )
    diffusion_utils._setup_smag_limit.with_backend(backend)(
        diff_multfac_vn, out=smag_limit, offset_provider={}
    )

    assert np.allclose(expected_diff_multfac_vn, diff_multfac_vn.asnumpy())
    assert np.allclose(expected_smag_limit, smag_limit.asnumpy())
