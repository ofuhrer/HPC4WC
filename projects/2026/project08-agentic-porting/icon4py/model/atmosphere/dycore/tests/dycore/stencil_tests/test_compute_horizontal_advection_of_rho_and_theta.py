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
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_of_rho_and_theta import (
    _compute_horizontal_advection_of_rho_and_theta,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing import stencil_tests


# TODO(): copied from `test_mo_math_gradients_grad_green_gauss_cell_dsl_numpy`. delete that test?
def mo_math_gradients_grad_green_gauss_cell_dsl_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    p_ccpr1: np.ndarray,
    p_ccpr2: np.ndarray,
    geofac_grg_x: np.ndarray,
    geofac_grg_y: np.ndarray,
) -> tuple[np.ndarray, ...]:
    c2e2cO = connectivities[dims.C2E2CODim]
    geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
    p_grad_1_u = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_x * p_ccpr1[c2e2cO], 0), axis=1
    )
    geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
    p_grad_1_v = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_y * p_ccpr1[c2e2cO], 0), axis=1
    )
    p_grad_2_u = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_x * p_ccpr2[c2e2cO], 0), axis=1
    )
    p_grad_2_v = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_y * p_ccpr2[c2e2cO], 0), axis=1
    )
    return (
        p_grad_1_u,
        p_grad_1_v,
        p_grad_2_u,
        p_grad_2_v,
    )


def compute_btraj_numpy(
    *,
    p_vn: np.ndarray,
    p_vt: np.ndarray,
    pos_on_tplane_e_1: np.ndarray,
    pos_on_tplane_e_2: np.ndarray,
    primal_normal_cell_1: np.ndarray,
    dual_normal_cell_1: np.ndarray,
    primal_normal_cell_2: np.ndarray,
    dual_normal_cell_2: np.ndarray,
    p_dthalf: float,
    **kwargs: Any,
) -> tuple[np.ndarray, ...]:
    lvn_pos = np.where(p_vn > 0.0, True, False)
    pos_on_tplane_e_1 = np.expand_dims(pos_on_tplane_e_1, axis=-1)
    pos_on_tplane_e_2 = np.expand_dims(pos_on_tplane_e_2, axis=-1)
    primal_normal_cell_1 = np.expand_dims(primal_normal_cell_1, axis=-1)
    dual_normal_cell_1 = np.expand_dims(dual_normal_cell_1, axis=-1)
    primal_normal_cell_2 = np.expand_dims(primal_normal_cell_2, axis=-1)
    dual_normal_cell_2 = np.expand_dims(dual_normal_cell_2, axis=-1)

    z_ntdistv_bary_1 = np.negative(
        p_vn * p_dthalf + np.where(lvn_pos, pos_on_tplane_e_1[:, 0], pos_on_tplane_e_1[:, 1])
    )
    z_ntdistv_bary_2 = np.negative(
        p_vt * p_dthalf + np.where(lvn_pos, pos_on_tplane_e_2[:, 0], pos_on_tplane_e_2[:, 1])
    )

    p_distv_bary_1 = np.where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_1[:, 0] + z_ntdistv_bary_2 * dual_normal_cell_1[:, 0],
        z_ntdistv_bary_1 * primal_normal_cell_1[:, 1] + z_ntdistv_bary_2 * dual_normal_cell_1[:, 1],
    )

    p_distv_bary_2 = np.where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_2[:, 0] + z_ntdistv_bary_2 * dual_normal_cell_2[:, 0],
        z_ntdistv_bary_1 * primal_normal_cell_2[:, 1] + z_ntdistv_bary_2 * dual_normal_cell_2[:, 1],
    )

    return p_distv_bary_1, p_distv_bary_2


def sten_16_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    p_vn: np.ndarray,
    rho_ref_me: np.ndarray,
    theta_ref_me: np.ndarray,
    p_distv_bary_1: np.ndarray,
    p_distv_bary_2: np.ndarray,
    z_grad_rth_1: np.ndarray,
    z_grad_rth_2: np.ndarray,
    z_grad_rth_3: np.ndarray,
    z_grad_rth_4: np.ndarray,
    z_rth_pr_1: np.ndarray,
    z_rth_pr_2: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    e2c = connectivities[dims.E2CDim]
    z_rth_pr_1_e2c = z_rth_pr_1[e2c]
    z_rth_pr_2_e2c = z_rth_pr_2[e2c]
    z_grad_rth_1_e2c = z_grad_rth_1[e2c]
    z_grad_rth_2_e2c = z_grad_rth_2[e2c]
    z_grad_rth_3_e2c = z_grad_rth_3[e2c]
    z_grad_rth_4_e2c = z_grad_rth_4[e2c]

    z_rho_e = np.where(
        p_vn > 0,
        rho_ref_me
        + z_rth_pr_1_e2c[:, 0]
        + p_distv_bary_1 * z_grad_rth_1_e2c[:, 0]
        + p_distv_bary_2 * z_grad_rth_2_e2c[:, 0],
        rho_ref_me
        + z_rth_pr_1_e2c[:, 1]
        + p_distv_bary_1 * z_grad_rth_1_e2c[:, 1]
        + p_distv_bary_2 * z_grad_rth_2_e2c[:, 1],
    )

    z_theta_v_e = np.where(
        p_vn > 0,
        theta_ref_me
        + z_rth_pr_2_e2c[:, 0]
        + p_distv_bary_1 * z_grad_rth_3_e2c[:, 0]
        + p_distv_bary_2 * z_grad_rth_4_e2c[:, 0],
        theta_ref_me
        + z_rth_pr_2_e2c[:, 1]
        + p_distv_bary_1 * z_grad_rth_3_e2c[:, 1]
        + p_distv_bary_2 * z_grad_rth_4_e2c[:, 1],
    )

    return z_rho_e, z_theta_v_e


def compute_horizontal_advection_of_rho_and_theta_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    p_vn: np.ndarray,
    p_vt: np.ndarray,
    pos_on_tplane_e_1: np.ndarray,
    pos_on_tplane_e_2: np.ndarray,
    primal_normal_cell_1: np.ndarray,
    dual_normal_cell_1: np.ndarray,
    primal_normal_cell_2: np.ndarray,
    dual_normal_cell_2: np.ndarray,
    p_dthalf: float,
    rho_ref_me: np.ndarray,
    theta_ref_me: np.ndarray,
    perturbed_rho_at_cells_on_model_levels: np.ndarray,
    perturbed_theta_v_at_cells_on_model_levels: np.ndarray,
    geofac_grg_x: np.ndarray,
    geofac_grg_y: np.ndarray,
) -> tuple[np.ndarray, ...]:
    (
        z_grad_rth_1,
        z_grad_rth_2,
        z_grad_rth_3,
        z_grad_rth_4,
    ) = mo_math_gradients_grad_green_gauss_cell_dsl_numpy(
        connectivities=connectivities,
        p_ccpr1=perturbed_rho_at_cells_on_model_levels,
        p_ccpr2=perturbed_theta_v_at_cells_on_model_levels,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
    )

    p_distv_bary_1, p_distv_bary_2 = compute_btraj_numpy(
        p_vn=p_vn,
        p_vt=p_vt,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        p_dthalf=p_dthalf,
    )

    z_rho_e, z_theta_v_e = sten_16_numpy(
        connectivities=connectivities,
        p_vn=p_vn,
        rho_ref_me=rho_ref_me,
        theta_ref_me=theta_ref_me,
        p_distv_bary_1=p_distv_bary_1,
        p_distv_bary_2=p_distv_bary_2,
        z_grad_rth_1=z_grad_rth_1,
        z_grad_rth_2=z_grad_rth_2,
        z_grad_rth_3=z_grad_rth_3,
        z_grad_rth_4=z_grad_rth_4,
        z_rth_pr_1=perturbed_rho_at_cells_on_model_levels,
        z_rth_pr_2=perturbed_theta_v_at_cells_on_model_levels,
    )

    return (z_rho_e, z_theta_v_e)


@pytest.mark.embedded_remap_error
@pytest.mark.skip_value_error
class TestComputeHorizontalAvectionOfRhoAndTheta(stencil_tests.StencilTest):
    PROGRAM = _compute_horizontal_advection_of_rho_and_theta
    OUTPUTS = ("out",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_vn: np.ndarray,
        p_vt: np.ndarray,
        pos_on_tplane_e_1: np.ndarray,
        pos_on_tplane_e_2: np.ndarray,
        primal_normal_cell_1: np.ndarray,
        dual_normal_cell_1: np.ndarray,
        primal_normal_cell_2: np.ndarray,
        dual_normal_cell_2: np.ndarray,
        p_dthalf: float,
        rho_ref_me: np.ndarray,
        theta_ref_me: np.ndarray,
        perturbed_rho_at_cells_on_model_levels: np.ndarray,
        perturbed_theta_v_at_cells_on_model_levels: np.ndarray,
        geofac_grg_x: np.ndarray,
        geofac_grg_y: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_rho_e, z_theta_v_e = compute_horizontal_advection_of_rho_and_theta_numpy(
            connectivities=connectivities,
            p_vn=p_vn,
            p_vt=p_vt,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            primal_normal_cell_1=primal_normal_cell_1,
            dual_normal_cell_1=dual_normal_cell_1,
            primal_normal_cell_2=primal_normal_cell_2,
            dual_normal_cell_2=dual_normal_cell_2,
            p_dthalf=p_dthalf,
            rho_ref_me=rho_ref_me,
            theta_ref_me=theta_ref_me,
            perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
        )
        return dict(out=(z_rho_e, z_theta_v_e))

    @pytest.fixture
    def input_data(
        self, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType | gtx.Domain | tuple[gtx.Field, ...]]:
        p_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        p_vt = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        pos_on_tplane_e_1 = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        pos_on_tplane_e_2 = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        primal_normal_cell_1 = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        dual_normal_cell_1 = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        primal_normal_cell_2 = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        dual_normal_cell_2 = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, dtype=ta.wpfloat
        )
        p_dthalf = 2.0

        rho_ref_me = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        theta_ref_me = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        perturbed_rho_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        perturbed_theta_v_at_cells_on_model_levels = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat
        )
        geofac_grg_x = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=ta.wpfloat)
        geofac_grg_y = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=ta.wpfloat)
        z_rho_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        z_theta_v_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            p_vn=p_vn,
            p_vt=p_vt,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            primal_normal_cell_1=primal_normal_cell_1,
            dual_normal_cell_1=dual_normal_cell_1,
            primal_normal_cell_2=primal_normal_cell_2,
            dual_normal_cell_2=dual_normal_cell_2,
            p_dthalf=p_dthalf,
            rho_ref_me=rho_ref_me,
            theta_ref_me=theta_ref_me,
            perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            out=(z_rho_e, z_theta_v_e),
            domain=gtx.domain(
                {
                    dims.EdgeDim: (0, gtx.int32(grid.num_edges)),
                    dims.KDim: (0, gtx.int32(grid.num_levels)),
                }
            ),
        )
