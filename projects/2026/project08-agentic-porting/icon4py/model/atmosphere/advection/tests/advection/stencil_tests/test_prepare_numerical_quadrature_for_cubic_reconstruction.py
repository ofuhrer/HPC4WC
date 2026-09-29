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

from icon4py.model.atmosphere.advection.stencils.prepare_numerical_quadrature_for_cubic_reconstruction import (
    prepare_numerical_quadrature_for_cubic_reconstruction,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestPrepareNumericalQuadratureForCubicReconstruction(stencil_tests.StencilTest):
    PROGRAM = prepare_numerical_quadrature_for_cubic_reconstruction
    OUTPUTS = (
        "p_quad_vector_sum_1",
        "p_quad_vector_sum_2",
        "p_quad_vector_sum_3",
        "p_quad_vector_sum_4",
        "p_quad_vector_sum_5",
        "p_quad_vector_sum_6",
        "p_quad_vector_sum_7",
        "p_quad_vector_sum_8",
        "p_quad_vector_sum_9",
        "p_quad_vector_sum_10",
        "p_dreg_area_out",
    )

    @staticmethod
    def _compute_wgt_t_detjac(
        *,
        wgt_zeta_1,
        wgt_zeta_2,
        wgt_eta_1,
        wgt_eta_2,
        dbl_eps,
        p_coords_dreg_v_1_x,
        p_coords_dreg_v_2_x,
        p_coords_dreg_v_3_x,
        p_coords_dreg_v_4_x,
        p_coords_dreg_v_1_y,
        p_coords_dreg_v_2_y,
        p_coords_dreg_v_3_y,
        p_coords_dreg_v_4_y,
        zeta_1,
        zeta_2,
        zeta_3,
        zeta_4,
        eta_1,
        eta_2,
        eta_3,
        eta_4,
    ):
        # Compute z_wgt
        z_wgt_1 = 0.0625 * wgt_zeta_1 * wgt_eta_1
        z_wgt_2 = 0.0625 * wgt_zeta_1 * wgt_eta_2
        z_wgt_3 = 0.0625 * wgt_zeta_2 * wgt_eta_1
        z_wgt_4 = 0.0625 * wgt_zeta_2 * wgt_eta_2

        # Compute z_eta
        z_eta_1_1, z_eta_2_1, z_eta_3_1, z_eta_4_1 = (
            1.0 - eta_1,
            1.0 - eta_2,
            1.0 - eta_3,
            1.0 - eta_4,
        )
        z_eta_1_2, z_eta_2_2, z_eta_3_2, z_eta_4_2 = (
            1.0 + eta_1,
            1.0 + eta_2,
            1.0 + eta_3,
            1.0 + eta_4,
        )
        z_eta_1_3, z_eta_2_3, z_eta_3_3, z_eta_4_3 = (
            1.0 - zeta_1,
            1.0 - zeta_2,
            1.0 - zeta_3,
            1.0 - zeta_4,
        )
        z_eta_1_4, z_eta_2_4, z_eta_3_4, z_eta_4_4 = (
            1.0 + zeta_1,
            1.0 + zeta_2,
            1.0 + zeta_3,
            1.0 + zeta_4,
        )

        wgt_t_detjac_1 = dbl_eps + z_wgt_1 * (
            (
                z_eta_1_1 * (p_coords_dreg_v_2_x - p_coords_dreg_v_1_x)
                + z_eta_1_2 * (p_coords_dreg_v_3_x - p_coords_dreg_v_4_x)
            )
            * (
                z_eta_1_3 * (p_coords_dreg_v_4_y - p_coords_dreg_v_1_y)
                - z_eta_1_4 * (p_coords_dreg_v_2_y - p_coords_dreg_v_3_y)
            )
            - (
                z_eta_1_1 * (p_coords_dreg_v_2_y - p_coords_dreg_v_1_y)
                + z_eta_1_2 * (p_coords_dreg_v_3_y - p_coords_dreg_v_4_y)
            )
            * (
                z_eta_1_3 * (p_coords_dreg_v_4_x - p_coords_dreg_v_1_x)
                - z_eta_1_4 * (p_coords_dreg_v_2_x - p_coords_dreg_v_3_x)
            )
        )
        wgt_t_detjac_2 = dbl_eps + z_wgt_2 * (
            (
                z_eta_2_1 * (p_coords_dreg_v_2_x - p_coords_dreg_v_1_x)
                + z_eta_2_2 * (p_coords_dreg_v_3_x - p_coords_dreg_v_4_x)
            )
            * (
                z_eta_2_3 * (p_coords_dreg_v_4_y - p_coords_dreg_v_1_y)
                - z_eta_2_4 * (p_coords_dreg_v_2_y - p_coords_dreg_v_3_y)
            )
            - (
                z_eta_2_1 * (p_coords_dreg_v_2_y - p_coords_dreg_v_1_y)
                + z_eta_2_2 * (p_coords_dreg_v_3_y - p_coords_dreg_v_4_y)
            )
            * (
                z_eta_2_3 * (p_coords_dreg_v_4_x - p_coords_dreg_v_1_x)
                - z_eta_2_4 * (p_coords_dreg_v_2_x - p_coords_dreg_v_3_x)
            )
        )
        wgt_t_detjac_3 = dbl_eps + z_wgt_3 * (
            (
                z_eta_3_1 * (p_coords_dreg_v_2_x - p_coords_dreg_v_1_x)
                + z_eta_3_2 * (p_coords_dreg_v_3_x - p_coords_dreg_v_4_x)
            )
            * (
                z_eta_3_3 * (p_coords_dreg_v_4_y - p_coords_dreg_v_1_y)
                - z_eta_3_4 * (p_coords_dreg_v_2_y - p_coords_dreg_v_3_y)
            )
            - (
                z_eta_3_1 * (p_coords_dreg_v_2_y - p_coords_dreg_v_1_y)
                + z_eta_3_2 * (p_coords_dreg_v_3_y - p_coords_dreg_v_4_y)
            )
            * (
                z_eta_3_3 * (p_coords_dreg_v_4_x - p_coords_dreg_v_1_x)
                - z_eta_3_4 * (p_coords_dreg_v_2_x - p_coords_dreg_v_3_x)
            )
        )
        wgt_t_detjac_4 = dbl_eps + z_wgt_4 * (
            (
                z_eta_4_1 * (p_coords_dreg_v_2_x - p_coords_dreg_v_1_x)
                + z_eta_4_2 * (p_coords_dreg_v_3_x - p_coords_dreg_v_4_x)
            )
            * (
                z_eta_4_3 * (p_coords_dreg_v_4_y - p_coords_dreg_v_1_y)
                - z_eta_4_4 * (p_coords_dreg_v_2_y - p_coords_dreg_v_3_y)
            )
            - (
                z_eta_4_1 * (p_coords_dreg_v_2_y - p_coords_dreg_v_1_y)
                + z_eta_4_2 * (p_coords_dreg_v_3_y - p_coords_dreg_v_4_y)
            )
            * (
                z_eta_4_3 * (p_coords_dreg_v_4_x - p_coords_dreg_v_1_x)
                - z_eta_4_4 * (p_coords_dreg_v_2_x - p_coords_dreg_v_3_x)
            )
        )
        return wgt_t_detjac_1, wgt_t_detjac_2, wgt_t_detjac_3, wgt_t_detjac_4

    @staticmethod
    def _compute_z_gauss_points(
        *,
        p_coords_dreg_v_1_x,
        p_coords_dreg_v_2_x,
        p_coords_dreg_v_3_x,
        p_coords_dreg_v_4_x,
        p_coords_dreg_v_1_y,
        p_coords_dreg_v_2_y,
        p_coords_dreg_v_3_y,
        p_coords_dreg_v_4_y,
        shape_func_1_1,
        shape_func_2_1,
        shape_func_3_1,
        shape_func_4_1,
        shape_func_1_2,
        shape_func_2_2,
        shape_func_3_2,
        shape_func_4_2,
        shape_func_1_3,
        shape_func_2_3,
        shape_func_3_3,
        shape_func_4_3,
        shape_func_1_4,
        shape_func_2_4,
        shape_func_3_4,
        shape_func_4_4,
    ):
        z_gauss_pts_1_x = (
            shape_func_1_1 * p_coords_dreg_v_1_x
            + shape_func_2_1 * p_coords_dreg_v_2_x
            + shape_func_3_1 * p_coords_dreg_v_3_x
            + shape_func_4_1 * p_coords_dreg_v_4_x
        )
        z_gauss_pts_1_y = (
            shape_func_1_1 * p_coords_dreg_v_1_y
            + shape_func_2_1 * p_coords_dreg_v_2_y
            + shape_func_3_1 * p_coords_dreg_v_3_y
            + shape_func_4_1 * p_coords_dreg_v_4_y
        )
        z_gauss_pts_2_x = (
            shape_func_1_2 * p_coords_dreg_v_1_x
            + shape_func_2_2 * p_coords_dreg_v_2_x
            + shape_func_3_2 * p_coords_dreg_v_3_x
            + shape_func_4_2 * p_coords_dreg_v_4_x
        )
        z_gauss_pts_2_y = (
            shape_func_1_2 * p_coords_dreg_v_1_y
            + shape_func_2_2 * p_coords_dreg_v_2_y
            + shape_func_3_2 * p_coords_dreg_v_3_y
            + shape_func_4_2 * p_coords_dreg_v_4_y
        )
        z_gauss_pts_3_x = (
            shape_func_1_3 * p_coords_dreg_v_1_x
            + shape_func_2_3 * p_coords_dreg_v_2_x
            + shape_func_3_3 * p_coords_dreg_v_3_x
            + shape_func_4_3 * p_coords_dreg_v_4_x
        )
        z_gauss_pts_3_y = (
            shape_func_1_3 * p_coords_dreg_v_1_y
            + shape_func_2_3 * p_coords_dreg_v_2_y
            + shape_func_3_3 * p_coords_dreg_v_3_y
            + shape_func_4_3 * p_coords_dreg_v_4_y
        )
        z_gauss_pts_4_x = (
            shape_func_1_4 * p_coords_dreg_v_1_x
            + shape_func_2_4 * p_coords_dreg_v_2_x
            + shape_func_3_4 * p_coords_dreg_v_3_x
            + shape_func_4_4 * p_coords_dreg_v_4_x
        )
        z_gauss_pts_4_y = (
            shape_func_1_4 * p_coords_dreg_v_1_y
            + shape_func_2_4 * p_coords_dreg_v_2_y
            + shape_func_3_4 * p_coords_dreg_v_3_y
            + shape_func_4_4 * p_coords_dreg_v_4_y
        )
        return (
            z_gauss_pts_1_x,
            z_gauss_pts_1_y,
            z_gauss_pts_2_x,
            z_gauss_pts_2_y,
            z_gauss_pts_3_x,
            z_gauss_pts_3_y,
            z_gauss_pts_4_x,
            z_gauss_pts_4_y,
        )

    @staticmethod
    def _compute_vector_sums(
        *,
        wgt_t_detjac_1,
        wgt_t_detjac_2,
        wgt_t_detjac_3,
        wgt_t_detjac_4,
        z_gauss_pts_1_x,
        z_gauss_pts_1_y,
        z_gauss_pts_2_x,
        z_gauss_pts_2_y,
        z_gauss_pts_3_x,
        z_gauss_pts_3_y,
        z_gauss_pts_4_x,
        z_gauss_pts_4_y,
    ):
        p_quad_vector_sum_1 = wgt_t_detjac_1 + wgt_t_detjac_2 + wgt_t_detjac_3 + wgt_t_detjac_4
        p_quad_vector_sum_2 = (
            wgt_t_detjac_1 * z_gauss_pts_1_x
            + wgt_t_detjac_2 * z_gauss_pts_2_x
            + wgt_t_detjac_3 * z_gauss_pts_3_x
            + wgt_t_detjac_4 * z_gauss_pts_4_x
        )
        p_quad_vector_sum_3 = (
            wgt_t_detjac_1 * z_gauss_pts_1_y
            + wgt_t_detjac_2 * z_gauss_pts_2_y
            + wgt_t_detjac_3 * z_gauss_pts_3_y
            + wgt_t_detjac_4 * z_gauss_pts_4_y
        )
        p_quad_vector_sum_4 = (
            wgt_t_detjac_1 * z_gauss_pts_1_x * z_gauss_pts_1_x
            + wgt_t_detjac_2 * z_gauss_pts_2_x * z_gauss_pts_2_x
            + wgt_t_detjac_3 * z_gauss_pts_3_x * z_gauss_pts_3_x
            + wgt_t_detjac_4 * z_gauss_pts_4_x * z_gauss_pts_4_x
        )
        p_quad_vector_sum_5 = (
            wgt_t_detjac_1 * z_gauss_pts_1_y * z_gauss_pts_1_y
            + wgt_t_detjac_2 * z_gauss_pts_2_y * z_gauss_pts_2_y
            + wgt_t_detjac_3 * z_gauss_pts_3_y * z_gauss_pts_3_y
            + wgt_t_detjac_4 * z_gauss_pts_4_y * z_gauss_pts_4_y
        )
        p_quad_vector_sum_6 = (
            wgt_t_detjac_1 * z_gauss_pts_1_x * z_gauss_pts_1_y
            + wgt_t_detjac_2 * z_gauss_pts_2_x * z_gauss_pts_2_y
            + wgt_t_detjac_3 * z_gauss_pts_3_x * z_gauss_pts_3_y
            + wgt_t_detjac_4 * z_gauss_pts_4_x * z_gauss_pts_4_y
        )
        p_quad_vector_sum_7 = (
            wgt_t_detjac_1 * z_gauss_pts_1_x * z_gauss_pts_1_x * z_gauss_pts_1_x
            + wgt_t_detjac_2 * z_gauss_pts_2_x * z_gauss_pts_2_x * z_gauss_pts_2_x
            + wgt_t_detjac_3 * z_gauss_pts_3_x * z_gauss_pts_3_x * z_gauss_pts_3_x
            + wgt_t_detjac_4 * z_gauss_pts_4_x * z_gauss_pts_4_x * z_gauss_pts_4_x
        )
        p_quad_vector_sum_8 = (
            wgt_t_detjac_1 * z_gauss_pts_1_y * z_gauss_pts_1_y * z_gauss_pts_1_y
            + wgt_t_detjac_2 * z_gauss_pts_2_y * z_gauss_pts_2_y * z_gauss_pts_2_y
            + wgt_t_detjac_3 * z_gauss_pts_3_y * z_gauss_pts_3_y * z_gauss_pts_3_y
            + wgt_t_detjac_4 * z_gauss_pts_4_y * z_gauss_pts_4_y * z_gauss_pts_4_y
        )
        p_quad_vector_sum_9 = (
            wgt_t_detjac_1 * z_gauss_pts_1_x * z_gauss_pts_1_x * z_gauss_pts_1_y
            + wgt_t_detjac_2 * z_gauss_pts_2_x * z_gauss_pts_2_x * z_gauss_pts_2_y
            + wgt_t_detjac_3 * z_gauss_pts_3_x * z_gauss_pts_3_x * z_gauss_pts_3_y
            + wgt_t_detjac_4 * z_gauss_pts_4_x * z_gauss_pts_4_x * z_gauss_pts_4_y
        )
        p_quad_vector_sum_10 = (
            wgt_t_detjac_1 * z_gauss_pts_1_x * z_gauss_pts_1_y * z_gauss_pts_1_y
            + wgt_t_detjac_2 * z_gauss_pts_2_x * z_gauss_pts_2_y * z_gauss_pts_2_y
            + wgt_t_detjac_3 * z_gauss_pts_3_x * z_gauss_pts_3_y * z_gauss_pts_3_y
            + wgt_t_detjac_4 * z_gauss_pts_4_x * z_gauss_pts_4_y * z_gauss_pts_4_y
        )
        return (
            p_quad_vector_sum_1,
            p_quad_vector_sum_2,
            p_quad_vector_sum_3,
            p_quad_vector_sum_4,
            p_quad_vector_sum_5,
            p_quad_vector_sum_6,
            p_quad_vector_sum_7,
            p_quad_vector_sum_8,
            p_quad_vector_sum_9,
            p_quad_vector_sum_10,
        )

    @classmethod
    def reference(
        cls,
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_coords_dreg_v_1_x: np.ndarray,
        p_coords_dreg_v_2_x: np.ndarray,
        p_coords_dreg_v_3_x: np.ndarray,
        p_coords_dreg_v_4_x: np.ndarray,
        p_coords_dreg_v_1_y: np.ndarray,
        p_coords_dreg_v_2_y: np.ndarray,
        p_coords_dreg_v_3_y: np.ndarray,
        p_coords_dreg_v_4_y: np.ndarray,
        shape_func_1_1: float,
        shape_func_2_1: float,
        shape_func_3_1: float,
        shape_func_4_1: float,
        shape_func_1_2: float,
        shape_func_2_2: float,
        shape_func_3_2: float,
        shape_func_4_2: float,
        shape_func_1_3: float,
        shape_func_2_3: float,
        shape_func_3_3: float,
        shape_func_4_3: float,
        shape_func_1_4: float,
        shape_func_2_4: float,
        shape_func_3_4: float,
        shape_func_4_4: float,
        zeta_1: float,
        zeta_2: float,
        zeta_3: float,
        zeta_4: float,
        eta_1: float,
        eta_2: float,
        eta_3: float,
        eta_4: float,
        wgt_zeta_1: float,
        wgt_zeta_2: float,
        wgt_eta_1: float,
        wgt_eta_2: float,
        dbl_eps: float,
        eps: float,
        **kwargs: Any,
    ) -> dict:
        wgt_t_detjac_1, wgt_t_detjac_2, wgt_t_detjac_3, wgt_t_detjac_4 = cls._compute_wgt_t_detjac(
            wgt_zeta_1=wgt_zeta_1,
            wgt_zeta_2=wgt_zeta_2,
            wgt_eta_1=wgt_eta_1,
            wgt_eta_2=wgt_eta_2,
            dbl_eps=dbl_eps,
            p_coords_dreg_v_1_x=p_coords_dreg_v_1_x,
            p_coords_dreg_v_2_x=p_coords_dreg_v_2_x,
            p_coords_dreg_v_3_x=p_coords_dreg_v_3_x,
            p_coords_dreg_v_4_x=p_coords_dreg_v_4_x,
            p_coords_dreg_v_1_y=p_coords_dreg_v_1_y,
            p_coords_dreg_v_2_y=p_coords_dreg_v_2_y,
            p_coords_dreg_v_3_y=p_coords_dreg_v_3_y,
            p_coords_dreg_v_4_y=p_coords_dreg_v_4_y,
            zeta_1=zeta_1,
            zeta_2=zeta_2,
            zeta_3=zeta_3,
            zeta_4=zeta_4,
            eta_1=eta_1,
            eta_2=eta_2,
            eta_3=eta_3,
            eta_4=eta_4,
        )

        (
            z_gauss_pts_1_x,
            z_gauss_pts_1_y,
            z_gauss_pts_2_x,
            z_gauss_pts_2_y,
            z_gauss_pts_3_x,
            z_gauss_pts_3_y,
            z_gauss_pts_4_x,
            z_gauss_pts_4_y,
        ) = cls._compute_z_gauss_points(
            p_coords_dreg_v_1_x=p_coords_dreg_v_1_x,
            p_coords_dreg_v_2_x=p_coords_dreg_v_2_x,
            p_coords_dreg_v_3_x=p_coords_dreg_v_3_x,
            p_coords_dreg_v_4_x=p_coords_dreg_v_4_x,
            p_coords_dreg_v_1_y=p_coords_dreg_v_1_y,
            p_coords_dreg_v_2_y=p_coords_dreg_v_2_y,
            p_coords_dreg_v_3_y=p_coords_dreg_v_3_y,
            p_coords_dreg_v_4_y=p_coords_dreg_v_4_y,
            shape_func_1_1=shape_func_1_1,
            shape_func_2_1=shape_func_2_1,
            shape_func_3_1=shape_func_3_1,
            shape_func_4_1=shape_func_4_1,
            shape_func_1_2=shape_func_1_2,
            shape_func_2_2=shape_func_2_2,
            shape_func_3_2=shape_func_3_2,
            shape_func_4_2=shape_func_4_2,
            shape_func_1_3=shape_func_1_3,
            shape_func_2_3=shape_func_2_3,
            shape_func_3_3=shape_func_3_3,
            shape_func_4_3=shape_func_4_3,
            shape_func_1_4=shape_func_1_4,
            shape_func_2_4=shape_func_2_4,
            shape_func_3_4=shape_func_3_4,
            shape_func_4_4=shape_func_4_4,
        )

        (
            p_quad_vector_sum_1,
            p_quad_vector_sum_2,
            p_quad_vector_sum_3,
            p_quad_vector_sum_4,
            p_quad_vector_sum_5,
            p_quad_vector_sum_6,
            p_quad_vector_sum_7,
            p_quad_vector_sum_8,
            p_quad_vector_sum_9,
            p_quad_vector_sum_10,
        ) = cls._compute_vector_sums(
            wgt_t_detjac_1=wgt_t_detjac_1,
            wgt_t_detjac_2=wgt_t_detjac_2,
            wgt_t_detjac_3=wgt_t_detjac_3,
            wgt_t_detjac_4=wgt_t_detjac_4,
            z_gauss_pts_1_x=z_gauss_pts_1_x,
            z_gauss_pts_1_y=z_gauss_pts_1_y,
            z_gauss_pts_2_x=z_gauss_pts_2_x,
            z_gauss_pts_2_y=z_gauss_pts_2_y,
            z_gauss_pts_3_x=z_gauss_pts_3_x,
            z_gauss_pts_3_y=z_gauss_pts_3_y,
            z_gauss_pts_4_x=z_gauss_pts_4_x,
            z_gauss_pts_4_y=z_gauss_pts_4_y,
        )

        z_area = p_quad_vector_sum_1
        p_dreg_area_out = np.where(
            z_area >= 0.0,
            np.maximum(eps, np.absolute(z_area)),
            -np.maximum(eps, np.absolute(z_area)),
        )
        return dict(
            p_quad_vector_sum_1=p_quad_vector_sum_1,
            p_quad_vector_sum_2=p_quad_vector_sum_2,
            p_quad_vector_sum_3=p_quad_vector_sum_3,
            p_quad_vector_sum_4=p_quad_vector_sum_4,
            p_quad_vector_sum_5=p_quad_vector_sum_5,
            p_quad_vector_sum_6=p_quad_vector_sum_6,
            p_quad_vector_sum_7=p_quad_vector_sum_7,
            p_quad_vector_sum_8=p_quad_vector_sum_8,
            p_quad_vector_sum_9=p_quad_vector_sum_9,
            p_quad_vector_sum_10=p_quad_vector_sum_10,
            p_dreg_area_out=p_dreg_area_out,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_coords_dreg_v_1_x = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_2_x = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_3_x = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_4_x = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_1_y = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_2_y = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_3_y = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_coords_dreg_v_4_y = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_quad_vector_sum_1 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        p_quad_vector_sum_2 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        p_quad_vector_sum_3 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        p_quad_vector_sum_4 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        p_quad_vector_sum_5 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        p_quad_vector_sum_6 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        p_quad_vector_sum_7 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        p_quad_vector_sum_8 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        p_quad_vector_sum_9 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        p_quad_vector_sum_10 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        p_dreg_area_out = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        shape_func_1_1 = 0.001
        shape_func_2_1 = 0.001
        shape_func_3_1 = 0.001
        shape_func_4_1 = 0.001
        shape_func_1_2 = 0.001
        shape_func_2_2 = 0.001
        shape_func_3_2 = 0.001
        shape_func_4_2 = 0.001
        shape_func_1_3 = 0.001
        shape_func_2_3 = 0.001
        shape_func_3_3 = 0.001
        shape_func_4_3 = 0.001
        shape_func_1_4 = 0.001
        shape_func_2_4 = 0.001
        shape_func_3_4 = 0.001
        shape_func_4_4 = 0.001
        zeta_1 = 0.002
        zeta_2 = 0.002
        zeta_3 = 0.002
        zeta_4 = 0.002
        eta_1 = 0.5
        eta_2 = 0.5
        eta_3 = 0.5
        eta_4 = 0.5
        wgt_zeta_1 = 0.003
        wgt_zeta_2 = 0.003
        wgt_eta_1 = 0.002
        wgt_eta_2 = 0.007
        dbl_eps = np.float64(0.1)
        eps = 0.1
        return dict(
            p_coords_dreg_v_1_x=p_coords_dreg_v_1_x,
            p_coords_dreg_v_2_x=p_coords_dreg_v_2_x,
            p_coords_dreg_v_3_x=p_coords_dreg_v_3_x,
            p_coords_dreg_v_4_x=p_coords_dreg_v_4_x,
            p_coords_dreg_v_1_y=p_coords_dreg_v_1_y,
            p_coords_dreg_v_2_y=p_coords_dreg_v_2_y,
            p_coords_dreg_v_3_y=p_coords_dreg_v_3_y,
            p_coords_dreg_v_4_y=p_coords_dreg_v_4_y,
            p_quad_vector_sum_1=p_quad_vector_sum_1,
            p_quad_vector_sum_2=p_quad_vector_sum_2,
            p_quad_vector_sum_3=p_quad_vector_sum_3,
            p_quad_vector_sum_4=p_quad_vector_sum_4,
            p_quad_vector_sum_5=p_quad_vector_sum_5,
            p_quad_vector_sum_6=p_quad_vector_sum_6,
            p_quad_vector_sum_7=p_quad_vector_sum_7,
            p_quad_vector_sum_8=p_quad_vector_sum_8,
            p_quad_vector_sum_9=p_quad_vector_sum_9,
            p_quad_vector_sum_10=p_quad_vector_sum_10,
            p_dreg_area_out=p_dreg_area_out,
            shape_func_1_1=shape_func_1_1,
            shape_func_2_1=shape_func_2_1,
            shape_func_3_1=shape_func_3_1,
            shape_func_4_1=shape_func_4_1,
            shape_func_1_2=shape_func_1_2,
            shape_func_2_2=shape_func_2_2,
            shape_func_3_2=shape_func_3_2,
            shape_func_4_2=shape_func_4_2,
            shape_func_1_3=shape_func_1_3,
            shape_func_2_3=shape_func_2_3,
            shape_func_3_3=shape_func_3_3,
            shape_func_4_3=shape_func_4_3,
            shape_func_1_4=shape_func_1_4,
            shape_func_2_4=shape_func_2_4,
            shape_func_3_4=shape_func_3_4,
            shape_func_4_4=shape_func_4_4,
            zeta_1=zeta_1,
            zeta_2=zeta_2,
            zeta_3=zeta_3,
            zeta_4=zeta_4,
            eta_1=eta_1,
            eta_2=eta_2,
            eta_3=eta_3,
            eta_4=eta_4,
            wgt_zeta_1=wgt_zeta_1,
            wgt_zeta_2=wgt_zeta_2,
            wgt_eta_1=wgt_eta_1,
            wgt_eta_2=wgt_eta_2,
            dbl_eps=dbl_eps,
            eps=eps,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
