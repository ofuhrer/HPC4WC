# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.advection.stencils.reconstruct_cubic_coefficients_svd import (
    reconstruct_cubic_coefficients_svd,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.testing import stencil_tests


@pytest.mark.embedded_remap_error
class TestReconstructCubicCoefficientsSvd(stencil_tests.StencilTest):
    PROGRAM = reconstruct_cubic_coefficients_svd
    OUTPUTS = (
        "p_coeff_1_dsl",
        "p_coeff_2_dsl",
        "p_coeff_3_dsl",
        "p_coeff_4_dsl",
        "p_coeff_5_dsl",
        "p_coeff_6_dsl",
        "p_coeff_7_dsl",
        "p_coeff_8_dsl",
        "p_coeff_9_dsl",
        "p_coeff_10_dsl",
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_cc: np.ndarray,
        lsq_pseudoinv_1: np.ndarray,
        lsq_pseudoinv_2: np.ndarray,
        lsq_pseudoinv_3: np.ndarray,
        lsq_pseudoinv_4: np.ndarray,
        lsq_pseudoinv_5: np.ndarray,
        lsq_pseudoinv_6: np.ndarray,
        lsq_pseudoinv_7: np.ndarray,
        lsq_pseudoinv_8: np.ndarray,
        lsq_pseudoinv_9: np.ndarray,
        lsq_moments_1: np.ndarray,
        lsq_moments_2: np.ndarray,
        lsq_moments_3: np.ndarray,
        lsq_moments_4: np.ndarray,
        lsq_moments_5: np.ndarray,
        lsq_moments_6: np.ndarray,
        lsq_moments_7: np.ndarray,
        lsq_moments_8: np.ndarray,
        lsq_moments_9: np.ndarray,
        p_coeff_1_dsl: np.ndarray,
        p_coeff_2_dsl: np.ndarray,
        p_coeff_3_dsl: np.ndarray,
        p_coeff_4_dsl: np.ndarray,
        p_coeff_5_dsl: np.ndarray,
        p_coeff_6_dsl: np.ndarray,
        p_coeff_7_dsl: np.ndarray,
        p_coeff_8_dsl: np.ndarray,
        p_coeff_9_dsl: np.ndarray,
        p_coeff_10_dsl: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        p_coeff_1_dsl_cp = p_coeff_1_dsl.copy()
        p_coeff_2_dsl_cp = p_coeff_2_dsl.copy()
        p_coeff_3_dsl_cp = p_coeff_3_dsl.copy()
        p_coeff_4_dsl_cp = p_coeff_4_dsl.copy()
        p_coeff_5_dsl_cp = p_coeff_5_dsl.copy()
        p_coeff_6_dsl_cp = p_coeff_6_dsl.copy()
        p_coeff_7_dsl_cp = p_coeff_7_dsl.copy()
        p_coeff_8_dsl_cp = p_coeff_8_dsl.copy()
        p_coeff_9_dsl_cp = p_coeff_9_dsl.copy()
        p_coeff_10_dsl_cp = p_coeff_10_dsl.copy()

        c2e2c2e2c = connectivities[dims.C2E2C2E2CDim]
        lsq_moments_1 = np.expand_dims(lsq_moments_1, axis=-1)
        lsq_moments_2 = np.expand_dims(lsq_moments_2, axis=-1)
        lsq_moments_3 = np.expand_dims(lsq_moments_3, axis=-1)
        lsq_moments_4 = np.expand_dims(lsq_moments_4, axis=-1)
        lsq_moments_5 = np.expand_dims(lsq_moments_5, axis=-1)
        lsq_moments_6 = np.expand_dims(lsq_moments_6, axis=-1)
        lsq_moments_7 = np.expand_dims(lsq_moments_7, axis=-1)
        lsq_moments_8 = np.expand_dims(lsq_moments_8, axis=-1)
        lsq_moments_9 = np.expand_dims(lsq_moments_9, axis=-1)
        lsq_moments_1 = np.broadcast_to(lsq_moments_1, p_cc.shape)
        lsq_moments_2 = np.broadcast_to(lsq_moments_2, p_cc.shape)
        lsq_moments_3 = np.broadcast_to(lsq_moments_3, p_cc.shape)
        lsq_moments_4 = np.broadcast_to(lsq_moments_4, p_cc.shape)
        lsq_moments_5 = np.broadcast_to(lsq_moments_5, p_cc.shape)
        lsq_moments_6 = np.broadcast_to(lsq_moments_6, p_cc.shape)
        lsq_moments_7 = np.broadcast_to(lsq_moments_7, p_cc.shape)
        lsq_moments_8 = np.broadcast_to(lsq_moments_8, p_cc.shape)
        lsq_pseudoinv_9 = np.expand_dims(lsq_pseudoinv_9, axis=-1)
        lsq_pseudoinv_8 = np.expand_dims(lsq_pseudoinv_8, axis=-1)
        lsq_pseudoinv_7 = np.expand_dims(lsq_pseudoinv_7, axis=-1)
        lsq_pseudoinv_6 = np.expand_dims(lsq_pseudoinv_6, axis=-1)
        lsq_pseudoinv_5 = np.expand_dims(lsq_pseudoinv_5, axis=-1)
        lsq_pseudoinv_4 = np.expand_dims(lsq_pseudoinv_4, axis=-1)
        lsq_pseudoinv_3 = np.expand_dims(lsq_pseudoinv_3, axis=-1)
        lsq_pseudoinv_2 = np.expand_dims(lsq_pseudoinv_2, axis=-1)
        lsq_pseudoinv_1 = np.expand_dims(lsq_pseudoinv_1, axis=-1)

        p_coeff_10_dsl = (
            lsq_pseudoinv_9[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
            + lsq_pseudoinv_9[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
            + lsq_pseudoinv_9[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
            + lsq_pseudoinv_9[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
            + lsq_pseudoinv_9[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
            + lsq_pseudoinv_9[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
            + lsq_pseudoinv_9[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
            + lsq_pseudoinv_9[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
            + lsq_pseudoinv_9[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
        )

        p_coeff_9_dsl = (
            lsq_pseudoinv_8[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
            + lsq_pseudoinv_8[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
            + lsq_pseudoinv_8[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
            + lsq_pseudoinv_8[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
            + lsq_pseudoinv_8[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
            + lsq_pseudoinv_8[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
            + lsq_pseudoinv_8[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
            + lsq_pseudoinv_8[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
            + lsq_pseudoinv_8[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
        )

        p_coeff_8_dsl = (
            lsq_pseudoinv_7[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
            + lsq_pseudoinv_7[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
            + lsq_pseudoinv_7[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
            + lsq_pseudoinv_7[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
            + lsq_pseudoinv_7[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
            + lsq_pseudoinv_7[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
            + lsq_pseudoinv_7[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
            + lsq_pseudoinv_7[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
            + lsq_pseudoinv_7[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
        )

        p_coeff_7_dsl = (
            lsq_pseudoinv_6[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
            + lsq_pseudoinv_6[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
            + lsq_pseudoinv_6[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
            + lsq_pseudoinv_6[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
            + lsq_pseudoinv_6[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
            + lsq_pseudoinv_6[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
            + lsq_pseudoinv_6[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
            + lsq_pseudoinv_6[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
            + lsq_pseudoinv_6[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
        )

        p_coeff_6_dsl = (
            lsq_pseudoinv_5[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
            + lsq_pseudoinv_5[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
            + lsq_pseudoinv_5[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
            + lsq_pseudoinv_5[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
            + lsq_pseudoinv_5[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
            + lsq_pseudoinv_5[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
            + lsq_pseudoinv_5[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
            + lsq_pseudoinv_5[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
            + lsq_pseudoinv_5[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
        )

        p_coeff_5_dsl = (
            lsq_pseudoinv_4[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
            + lsq_pseudoinv_4[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
            + lsq_pseudoinv_4[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
            + lsq_pseudoinv_4[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
            + lsq_pseudoinv_4[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
            + lsq_pseudoinv_4[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
            + lsq_pseudoinv_4[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
            + lsq_pseudoinv_4[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
            + lsq_pseudoinv_4[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
        )

        p_coeff_4_dsl = (
            lsq_pseudoinv_3[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
            + lsq_pseudoinv_3[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
            + lsq_pseudoinv_3[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
            + lsq_pseudoinv_3[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
            + lsq_pseudoinv_3[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
            + lsq_pseudoinv_3[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
            + lsq_pseudoinv_3[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
            + lsq_pseudoinv_3[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
            + lsq_pseudoinv_3[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
        )

        p_coeff_3_dsl = (
            lsq_pseudoinv_2[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
            + lsq_pseudoinv_2[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
            + lsq_pseudoinv_2[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
            + lsq_pseudoinv_2[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
            + lsq_pseudoinv_2[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
            + lsq_pseudoinv_2[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
            + lsq_pseudoinv_2[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
            + lsq_pseudoinv_2[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
            + lsq_pseudoinv_2[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
        )

        p_coeff_2_dsl = (
            lsq_pseudoinv_1[:, 0] * (p_cc[c2e2c2e2c[:, 0]] - p_cc)
            + lsq_pseudoinv_1[:, 1] * (p_cc[c2e2c2e2c[:, 1]] - p_cc)
            + lsq_pseudoinv_1[:, 2] * (p_cc[c2e2c2e2c[:, 2]] - p_cc)
            + lsq_pseudoinv_1[:, 3] * (p_cc[c2e2c2e2c[:, 3]] - p_cc)
            + lsq_pseudoinv_1[:, 4] * (p_cc[c2e2c2e2c[:, 4]] - p_cc)
            + lsq_pseudoinv_1[:, 5] * (p_cc[c2e2c2e2c[:, 5]] - p_cc)
            + lsq_pseudoinv_1[:, 6] * (p_cc[c2e2c2e2c[:, 6]] - p_cc)
            + lsq_pseudoinv_1[:, 7] * (p_cc[c2e2c2e2c[:, 7]] - p_cc)
            + lsq_pseudoinv_1[:, 8] * (p_cc[c2e2c2e2c[:, 8]] - p_cc)
        )

        p_coeff_1_dsl = p_cc - (
            p_coeff_2_dsl * lsq_moments_1
            + p_coeff_3_dsl * lsq_moments_2
            + p_coeff_4_dsl * lsq_moments_3
            + p_coeff_5_dsl * lsq_moments_4
            + p_coeff_6_dsl * lsq_moments_5
            + p_coeff_7_dsl * lsq_moments_6
            + p_coeff_8_dsl * lsq_moments_7
            + p_coeff_9_dsl * lsq_moments_8
            + p_coeff_10_dsl * lsq_moments_9
        )

        # restriction of execution domain
        p_coeff_1_dsl[0 : kwargs["horizontal_start"], :] = p_coeff_1_dsl_cp[
            0 : kwargs["horizontal_start"], :
        ]
        p_coeff_1_dsl[kwargs["horizontal_end"] :, :] = p_coeff_1_dsl_cp[
            kwargs["horizontal_end"] :, :
        ]
        p_coeff_2_dsl[0 : kwargs["horizontal_start"], :] = p_coeff_2_dsl_cp[
            0 : kwargs["horizontal_start"], :
        ]
        p_coeff_2_dsl[kwargs["horizontal_end"] :, :] = p_coeff_2_dsl_cp[
            kwargs["horizontal_end"] :, :
        ]
        p_coeff_3_dsl[0 : kwargs["horizontal_start"], :] = p_coeff_3_dsl_cp[
            0 : kwargs["horizontal_start"], :
        ]
        p_coeff_3_dsl[kwargs["horizontal_end"] :, :] = p_coeff_3_dsl_cp[
            kwargs["horizontal_end"] :, :
        ]
        p_coeff_4_dsl[0 : kwargs["horizontal_start"], :] = p_coeff_4_dsl_cp[
            0 : kwargs["horizontal_start"], :
        ]
        p_coeff_4_dsl[kwargs["horizontal_end"] :, :] = p_coeff_4_dsl_cp[
            kwargs["horizontal_end"] :, :
        ]
        p_coeff_5_dsl[0 : kwargs["horizontal_start"], :] = p_coeff_5_dsl_cp[
            0 : kwargs["horizontal_start"], :
        ]
        p_coeff_5_dsl[kwargs["horizontal_end"] :, :] = p_coeff_5_dsl_cp[
            kwargs["horizontal_end"] :, :
        ]
        p_coeff_6_dsl[0 : kwargs["horizontal_start"], :] = p_coeff_6_dsl_cp[
            0 : kwargs["horizontal_start"], :
        ]
        p_coeff_6_dsl[kwargs["horizontal_end"] :, :] = p_coeff_6_dsl_cp[
            kwargs["horizontal_end"] :, :
        ]
        p_coeff_7_dsl[0 : kwargs["horizontal_start"], :] = p_coeff_7_dsl_cp[
            0 : kwargs["horizontal_start"], :
        ]
        p_coeff_7_dsl[kwargs["horizontal_end"] :, :] = p_coeff_7_dsl_cp[
            kwargs["horizontal_end"] :, :
        ]
        p_coeff_8_dsl[0 : kwargs["horizontal_start"], :] = p_coeff_8_dsl_cp[
            0 : kwargs["horizontal_start"], :
        ]
        p_coeff_8_dsl[kwargs["horizontal_end"] :, :] = p_coeff_8_dsl_cp[
            kwargs["horizontal_end"] :, :
        ]
        p_coeff_9_dsl[0 : kwargs["horizontal_start"], :] = p_coeff_9_dsl_cp[
            0 : kwargs["horizontal_start"], :
        ]
        p_coeff_9_dsl[kwargs["horizontal_end"] :, :] = p_coeff_9_dsl_cp[
            kwargs["horizontal_end"] :, :
        ]
        p_coeff_10_dsl[0 : kwargs["horizontal_start"], :] = p_coeff_10_dsl_cp[
            0 : kwargs["horizontal_start"], :
        ]
        p_coeff_10_dsl[kwargs["horizontal_end"] :, :] = p_coeff_10_dsl_cp[
            kwargs["horizontal_end"] :, :
        ]

        return dict(
            p_coeff_1_dsl=p_coeff_1_dsl,
            p_coeff_2_dsl=p_coeff_2_dsl,
            p_coeff_3_dsl=p_coeff_3_dsl,
            p_coeff_4_dsl=p_coeff_4_dsl,
            p_coeff_5_dsl=p_coeff_5_dsl,
            p_coeff_6_dsl=p_coeff_6_dsl,
            p_coeff_7_dsl=p_coeff_7_dsl,
            p_coeff_8_dsl=p_coeff_8_dsl,
            p_coeff_9_dsl=p_coeff_9_dsl,
            p_coeff_10_dsl=p_coeff_10_dsl,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        lsq_pseudoinv_1_field = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_2_field = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_3_field = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_4_field = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_5_field = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_6_field = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_7_field = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_8_field = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)
        lsq_pseudoinv_9_field = data_alloc.random_field(grid, dims.CellDim, dims.C2E2C2E2CDim)

        lsq_moments_1 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_2 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_3 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_4 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_5 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_6 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_7 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_8 = data_alloc.random_field(grid, dims.CellDim)
        lsq_moments_9 = data_alloc.random_field(grid, dims.CellDim)
        p_coeff_1_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_2_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_3_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_4_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_5_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_6_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_7_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_8_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_9_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_coeff_10_dsl = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)

        cell_domain = h_grid.domain(dims.CellDim)
        horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

        return dict(
            p_cc=p_cc,
            lsq_pseudoinv_1=lsq_pseudoinv_1_field,
            lsq_pseudoinv_2=lsq_pseudoinv_2_field,
            lsq_pseudoinv_3=lsq_pseudoinv_3_field,
            lsq_pseudoinv_4=lsq_pseudoinv_4_field,
            lsq_pseudoinv_5=lsq_pseudoinv_5_field,
            lsq_pseudoinv_6=lsq_pseudoinv_6_field,
            lsq_pseudoinv_7=lsq_pseudoinv_7_field,
            lsq_pseudoinv_8=lsq_pseudoinv_8_field,
            lsq_pseudoinv_9=lsq_pseudoinv_9_field,
            lsq_moments_1=lsq_moments_1,
            lsq_moments_2=lsq_moments_2,
            lsq_moments_3=lsq_moments_3,
            lsq_moments_4=lsq_moments_4,
            lsq_moments_5=lsq_moments_5,
            lsq_moments_6=lsq_moments_6,
            lsq_moments_7=lsq_moments_7,
            lsq_moments_8=lsq_moments_8,
            lsq_moments_9=lsq_moments_9,
            p_coeff_1_dsl=p_coeff_1_dsl,
            p_coeff_2_dsl=p_coeff_2_dsl,
            p_coeff_3_dsl=p_coeff_3_dsl,
            p_coeff_4_dsl=p_coeff_4_dsl,
            p_coeff_5_dsl=p_coeff_5_dsl,
            p_coeff_6_dsl=p_coeff_6_dsl,
            p_coeff_7_dsl=p_coeff_7_dsl,
            p_coeff_8_dsl=p_coeff_8_dsl,
            p_coeff_9_dsl=p_coeff_9_dsl,
            p_coeff_10_dsl=p_coeff_10_dsl,
            horizontal_start=horizontal_start,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
