# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import TYPE_CHECKING

import numpy as np

from icon4py.model.common import dimension as dims, model_backends, model_options
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.math.smagorinsky import en_smag_fac_for_zero_nshift
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.fixtures.datatest import backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid, grid_manager
from icon4py.model.testing.reference_funcs import enhanced_smagorinski_factor_numpy


def test_init_enh_smag_fac(backend_like: model_backends.BackendLike, grid: base_grid.Grid) -> None:
    backend = model_options.customize_backend(None, backend_like)
    enh_smag_fac = data_alloc.zero_field(grid, dims.KDim, allocator=backend)
    a_vec = data_alloc.random_field(
        grid, dims.KDim, low=1.0, high=10.0, extend={dims.KDim: 1}, allocator=backend
    )
    fac = (0.67, 0.5, 1.3, 0.8)
    z = (0.1, 0.2, 0.3, 0.4)

    enhanced_smag_fac_np = enhanced_smagorinski_factor_numpy(fac, z, a_vec.asnumpy())
    en_smag_fac_for_zero_nshift.with_backend(backend)(
        a_vec,
        *fac,
        *z,
        enh_smag_fac,
        offset_provider={},
    )
    assert np.allclose(enhanced_smag_fac_np, enh_smag_fac.asnumpy())
