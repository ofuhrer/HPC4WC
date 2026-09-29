# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _prepare_numerical_quadrature_list_for_cubic_reconstruction(
    famask_int: fa.EdgeKField[gtx.int32],
    p_coords_dreg_v_1_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_2_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_3_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_4_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_1_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_2_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_3_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_4_y: fa.EdgeKField[ta.vpfloat],
    p_dreg_area_in: fa.EdgeKField[ta.vpfloat],
    shape_func_1_1: ta.wpfloat,
    shape_func_2_1: ta.wpfloat,
    shape_func_3_1: ta.wpfloat,
    shape_func_4_1: ta.wpfloat,
    shape_func_1_2: ta.wpfloat,
    shape_func_2_2: ta.wpfloat,
    shape_func_3_2: ta.wpfloat,
    shape_func_4_2: ta.wpfloat,
    shape_func_1_3: ta.wpfloat,
    shape_func_2_3: ta.wpfloat,
    shape_func_3_3: ta.wpfloat,
    shape_func_4_3: ta.wpfloat,
    shape_func_1_4: ta.wpfloat,
    shape_func_2_4: ta.wpfloat,
    shape_func_3_4: ta.wpfloat,
    shape_func_4_4: ta.wpfloat,
    zeta_1: ta.wpfloat,
    zeta_2: ta.wpfloat,
    zeta_3: ta.wpfloat,
    zeta_4: ta.wpfloat,
    eta_1: ta.wpfloat,
    eta_2: ta.wpfloat,
    eta_3: ta.wpfloat,
    eta_4: ta.wpfloat,
    wgt_zeta_1: ta.wpfloat,
    wgt_zeta_2: ta.wpfloat,
    wgt_eta_1: ta.wpfloat,
    wgt_eta_2: ta.wpfloat,
    dbl_eps: ta.wpfloat,
    eps: ta.wpfloat,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    z_wgt_1 = 0.0625 * wgt_zeta_1 * wgt_eta_1
    z_wgt_2 = 0.0625 * wgt_zeta_1 * wgt_eta_2
    z_wgt_3 = 0.0625 * wgt_zeta_2 * wgt_eta_1
    z_wgt_4 = 0.0625 * wgt_zeta_2 * wgt_eta_2

    z_eta_1_1 = 1.0 - eta_1
    z_eta_2_1 = 1.0 - eta_2
    z_eta_3_1 = 1.0 - eta_3
    z_eta_4_1 = 1.0 - eta_4
    z_eta_1_2 = 1.0 + eta_1
    z_eta_2_2 = 1.0 + eta_2
    z_eta_3_2 = 1.0 + eta_3
    z_eta_4_2 = 1.0 + eta_4
    z_eta_1_3 = 1.0 - zeta_1
    z_eta_2_3 = 1.0 - zeta_2
    z_eta_3_3 = 1.0 - zeta_3
    z_eta_4_3 = 1.0 - zeta_4
    z_eta_1_4 = 1.0 + zeta_1
    z_eta_2_4 = 1.0 + zeta_2
    z_eta_3_4 = 1.0 + zeta_3
    z_eta_4_4 = 1.0 + zeta_4

    famask_bool = famask_int == 1
    p_coords_dreg_v_1_x = where(famask_bool, p_coords_dreg_v_1_x, 0.0)
    p_coords_dreg_v_2_x = where(famask_bool, p_coords_dreg_v_2_x, 0.0)
    p_coords_dreg_v_3_x = where(famask_bool, p_coords_dreg_v_3_x, 0.0)
    p_coords_dreg_v_4_x = where(famask_bool, p_coords_dreg_v_4_x, 0.0)
    p_coords_dreg_v_1_y = where(famask_bool, p_coords_dreg_v_1_y, 0.0)
    p_coords_dreg_v_2_y = where(famask_bool, p_coords_dreg_v_2_y, 0.0)
    p_coords_dreg_v_3_y = where(famask_bool, p_coords_dreg_v_3_y, 0.0)
    p_coords_dreg_v_4_y = where(famask_bool, p_coords_dreg_v_4_y, 0.0)
    p_coords_dreg_v_1_x_wp = astype(p_coords_dreg_v_1_x, wpfloat)
    p_coords_dreg_v_2_x_wp = astype(p_coords_dreg_v_2_x, wpfloat)
    p_coords_dreg_v_3_x_wp = astype(p_coords_dreg_v_3_x, wpfloat)
    p_coords_dreg_v_4_x_wp = astype(p_coords_dreg_v_4_x, wpfloat)
    p_coords_dreg_v_1_y_wp = astype(p_coords_dreg_v_1_y, wpfloat)
    p_coords_dreg_v_2_y_wp = astype(p_coords_dreg_v_2_y, wpfloat)
    p_coords_dreg_v_3_y_wp = astype(p_coords_dreg_v_3_y, wpfloat)
    p_coords_dreg_v_4_y_wp = astype(p_coords_dreg_v_4_y, wpfloat)

    wgt_t_detjac_1 = where(
        famask_bool,
        dbl_eps
        + z_wgt_1
        * (
            (
                z_eta_1_1 * (p_coords_dreg_v_2_x_wp - p_coords_dreg_v_1_x_wp)
                + z_eta_1_2 * (p_coords_dreg_v_3_x_wp - p_coords_dreg_v_4_x_wp)
            )
            * (
                z_eta_1_3 * (p_coords_dreg_v_4_y_wp - p_coords_dreg_v_1_y_wp)
                - z_eta_1_4 * (p_coords_dreg_v_2_y_wp - p_coords_dreg_v_3_y_wp)
            )
            - (
                z_eta_1_1 * (p_coords_dreg_v_2_y_wp - p_coords_dreg_v_1_y_wp)
                + z_eta_1_2 * (p_coords_dreg_v_3_y_wp - p_coords_dreg_v_4_y_wp)
            )
            * (
                z_eta_1_3 * (p_coords_dreg_v_4_x_wp - p_coords_dreg_v_1_x_wp)
                - z_eta_1_4 * (p_coords_dreg_v_2_x_wp - p_coords_dreg_v_3_x_wp)
            )
        ),
        0.0,
    )
    wgt_t_detjac_2 = where(
        famask_bool,
        dbl_eps
        + z_wgt_2
        * (
            (
                z_eta_2_1 * (p_coords_dreg_v_2_x_wp - p_coords_dreg_v_1_x_wp)
                + z_eta_2_2 * (p_coords_dreg_v_3_x_wp - p_coords_dreg_v_4_x_wp)
            )
            * (
                z_eta_2_3 * (p_coords_dreg_v_4_y_wp - p_coords_dreg_v_1_y_wp)
                - z_eta_2_4 * (p_coords_dreg_v_2_y_wp - p_coords_dreg_v_3_y_wp)
            )
            - (
                z_eta_2_1 * (p_coords_dreg_v_2_y_wp - p_coords_dreg_v_1_y_wp)
                + z_eta_2_2 * (p_coords_dreg_v_3_y_wp - p_coords_dreg_v_4_y_wp)
            )
            * (
                z_eta_2_3 * (p_coords_dreg_v_4_x_wp - p_coords_dreg_v_1_x_wp)
                - z_eta_2_4 * (p_coords_dreg_v_2_x_wp - p_coords_dreg_v_3_x_wp)
            )
        ),
        0.0,
    )
    wgt_t_detjac_3 = where(
        famask_bool,
        dbl_eps
        + z_wgt_3
        * (
            (
                z_eta_3_1 * (p_coords_dreg_v_2_x_wp - p_coords_dreg_v_1_x_wp)
                + z_eta_3_2 * (p_coords_dreg_v_3_x_wp - p_coords_dreg_v_4_x_wp)
            )
            * (
                z_eta_3_3 * (p_coords_dreg_v_4_y_wp - p_coords_dreg_v_1_y_wp)
                - z_eta_3_4 * (p_coords_dreg_v_2_y_wp - p_coords_dreg_v_3_y_wp)
            )
            - (
                z_eta_3_1 * (p_coords_dreg_v_2_y_wp - p_coords_dreg_v_1_y_wp)
                + z_eta_3_2 * (p_coords_dreg_v_3_y_wp - p_coords_dreg_v_4_y_wp)
            )
            * (
                z_eta_3_3 * (p_coords_dreg_v_4_x_wp - p_coords_dreg_v_1_x_wp)
                - z_eta_3_4 * (p_coords_dreg_v_2_x_wp - p_coords_dreg_v_3_x_wp)
            )
        ),
        0.0,
    )
    wgt_t_detjac_4 = where(
        famask_bool,
        dbl_eps
        + z_wgt_4
        * (
            (
                z_eta_4_1 * (p_coords_dreg_v_2_x_wp - p_coords_dreg_v_1_x_wp)
                + z_eta_4_2 * (p_coords_dreg_v_3_x_wp - p_coords_dreg_v_4_x_wp)
            )
            * (
                z_eta_4_3 * (p_coords_dreg_v_4_y_wp - p_coords_dreg_v_1_y_wp)
                - z_eta_4_4 * (p_coords_dreg_v_2_y_wp - p_coords_dreg_v_3_y_wp)
            )
            - (
                z_eta_4_1 * (p_coords_dreg_v_2_y_wp - p_coords_dreg_v_1_y_wp)
                + z_eta_4_2 * (p_coords_dreg_v_3_y_wp - p_coords_dreg_v_4_y_wp)
            )
            * (
                z_eta_4_3 * (p_coords_dreg_v_4_x_wp - p_coords_dreg_v_1_x_wp)
                - z_eta_4_4 * (p_coords_dreg_v_2_x_wp - p_coords_dreg_v_3_x_wp)
            )
        ),
        0.0,
    )

    z_gauss_pts_1_x = (
        shape_func_1_1 * p_coords_dreg_v_1_x_wp
        + shape_func_2_1 * p_coords_dreg_v_2_x_wp
        + shape_func_3_1 * p_coords_dreg_v_3_x_wp
        + shape_func_4_1 * p_coords_dreg_v_4_x_wp
    )
    z_gauss_pts_1_y = (
        shape_func_1_1 * p_coords_dreg_v_1_y_wp
        + shape_func_2_1 * p_coords_dreg_v_2_y_wp
        + shape_func_3_1 * p_coords_dreg_v_3_y_wp
        + shape_func_4_1 * p_coords_dreg_v_4_y_wp
    )
    z_gauss_pts_2_x = (
        shape_func_1_2 * p_coords_dreg_v_1_x_wp
        + shape_func_2_2 * p_coords_dreg_v_2_x_wp
        + shape_func_3_2 * p_coords_dreg_v_3_x_wp
        + shape_func_4_2 * p_coords_dreg_v_4_x_wp
    )
    z_gauss_pts_2_y = (
        shape_func_1_2 * p_coords_dreg_v_1_y_wp
        + shape_func_2_2 * p_coords_dreg_v_2_y_wp
        + shape_func_3_2 * p_coords_dreg_v_3_y_wp
        + shape_func_4_2 * p_coords_dreg_v_4_y_wp
    )
    z_gauss_pts_3_x = (
        shape_func_1_3 * p_coords_dreg_v_1_x_wp
        + shape_func_2_3 * p_coords_dreg_v_2_x_wp
        + shape_func_3_3 * p_coords_dreg_v_3_x_wp
        + shape_func_4_3 * p_coords_dreg_v_4_x_wp
    )
    z_gauss_pts_3_y = (
        shape_func_1_3 * p_coords_dreg_v_1_y_wp
        + shape_func_2_3 * p_coords_dreg_v_2_y_wp
        + shape_func_3_3 * p_coords_dreg_v_3_y_wp
        + shape_func_4_3 * p_coords_dreg_v_4_y_wp
    )
    z_gauss_pts_4_x = (
        shape_func_1_4 * p_coords_dreg_v_1_x_wp
        + shape_func_2_4 * p_coords_dreg_v_2_x_wp
        + shape_func_3_4 * p_coords_dreg_v_3_x_wp
        + shape_func_4_4 * p_coords_dreg_v_4_x_wp
    )
    z_gauss_pts_4_y = (
        shape_func_1_4 * p_coords_dreg_v_1_y_wp
        + shape_func_2_4 * p_coords_dreg_v_2_y_wp
        + shape_func_3_4 * p_coords_dreg_v_3_y_wp
        + shape_func_4_4 * p_coords_dreg_v_4_y_wp
    )

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

    p_dreg_area = p_dreg_area_in + astype(p_quad_vector_sum_1, vpfloat)

    return (
        astype(p_quad_vector_sum_1, vpfloat),
        astype(p_quad_vector_sum_2, vpfloat),
        astype(p_quad_vector_sum_3, vpfloat),
        astype(p_quad_vector_sum_4, vpfloat),
        astype(p_quad_vector_sum_5, vpfloat),
        astype(p_quad_vector_sum_6, vpfloat),
        astype(p_quad_vector_sum_7, vpfloat),
        astype(p_quad_vector_sum_8, vpfloat),
        astype(p_quad_vector_sum_9, vpfloat),
        astype(p_quad_vector_sum_10, vpfloat),
        p_dreg_area,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def prepare_numerical_quadrature_list_for_cubic_reconstruction(
    famask_int: fa.EdgeKField[gtx.int32],
    p_coords_dreg_v_1_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_2_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_3_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_4_x: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_1_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_2_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_3_y: fa.EdgeKField[ta.vpfloat],
    p_coords_dreg_v_4_y: fa.EdgeKField[ta.vpfloat],
    p_dreg_area_in: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_1: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_2: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_3: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_4: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_5: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_6: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_7: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_8: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_9: fa.EdgeKField[ta.vpfloat],
    p_quad_vector_sum_10: fa.EdgeKField[ta.vpfloat],
    p_dreg_area: fa.EdgeKField[ta.vpfloat],
    shape_func_1_1: ta.wpfloat,
    shape_func_2_1: ta.wpfloat,
    shape_func_3_1: ta.wpfloat,
    shape_func_4_1: ta.wpfloat,
    shape_func_1_2: ta.wpfloat,
    shape_func_2_2: ta.wpfloat,
    shape_func_3_2: ta.wpfloat,
    shape_func_4_2: ta.wpfloat,
    shape_func_1_3: ta.wpfloat,
    shape_func_2_3: ta.wpfloat,
    shape_func_3_3: ta.wpfloat,
    shape_func_4_3: ta.wpfloat,
    shape_func_1_4: ta.wpfloat,
    shape_func_2_4: ta.wpfloat,
    shape_func_3_4: ta.wpfloat,
    shape_func_4_4: ta.wpfloat,
    zeta_1: ta.wpfloat,
    zeta_2: ta.wpfloat,
    zeta_3: ta.wpfloat,
    zeta_4: ta.wpfloat,
    eta_1: ta.wpfloat,
    eta_2: ta.wpfloat,
    eta_3: ta.wpfloat,
    eta_4: ta.wpfloat,
    wgt_zeta_1: ta.wpfloat,
    wgt_zeta_2: ta.wpfloat,
    wgt_eta_1: ta.wpfloat,
    wgt_eta_2: ta.wpfloat,
    dbl_eps: ta.wpfloat,
    eps: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _prepare_numerical_quadrature_list_for_cubic_reconstruction(
        famask_int=famask_int,
        p_coords_dreg_v_1_x=p_coords_dreg_v_1_x,
        p_coords_dreg_v_2_x=p_coords_dreg_v_2_x,
        p_coords_dreg_v_3_x=p_coords_dreg_v_3_x,
        p_coords_dreg_v_4_x=p_coords_dreg_v_4_x,
        p_coords_dreg_v_1_y=p_coords_dreg_v_1_y,
        p_coords_dreg_v_2_y=p_coords_dreg_v_2_y,
        p_coords_dreg_v_3_y=p_coords_dreg_v_3_y,
        p_coords_dreg_v_4_y=p_coords_dreg_v_4_y,
        p_dreg_area_in=p_dreg_area_in,
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
        out=(
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
            p_dreg_area,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
