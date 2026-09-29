# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_diagnostics_from_normal_wind import (
    compute_diagnostics_from_normal_wind,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests

from .test_compute_contravariant_correction import compute_contravariant_correction_numpy
from .test_compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity_numpy,
)
from .test_compute_horizontal_kinetic_energy import compute_horizontal_kinetic_energy_numpy
from .test_compute_tangential_wind import compute_tangential_wind_numpy
from .test_extrapolate_at_top import extrapolate_at_top_numpy
from .test_interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges import (
    interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_numpy,
    interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_vn_ie_numpy,
    interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_z_kin_hor_e_numpy,
)
from .test_interpolate_vt_to_interface_edges import interpolate_vt_to_interface_edges_numpy
from .test_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy,
)


def compute_diagnostics_from_normal_wind_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    tangential_wind_on_half_levels: np.ndarray,
    tangential_wind: np.ndarray,
    vn_on_half_levels: np.ndarray,
    horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
    contravariant_correction_at_edges_on_model_levels: np.ndarray,
    vn: np.ndarray,
    rbf_vec_coeff_e: np.ndarray,
    wgtfac_e: np.ndarray,
    ddxn_z_full: np.ndarray,
    ddxt_z_full: np.ndarray,
    wgtfacq_e: np.ndarray,
    horizontal_advection_of_w_at_edges_on_half_levels: np.ndarray,
    c_intp: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    w: np.ndarray,
    inv_primal_edge_length: np.ndarray,
    tangent_orientation: np.ndarray,
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: int,
    nlevp1: int,
    horizontal_start: int,
    horizontal_end: int,
    vertical_start: int,
) -> tuple[np.ndarray, ...]:
    k: np.ndarray = np.arange(nlevp1)
    k = k[np.newaxis, :]
    k_nlev = k[:, :-1]

    initial_vn_on_half_levels = vn_on_half_levels.copy()

    tangential_wind = np.where(
        k_nlev >= vertical_start,
        compute_tangential_wind_numpy(connectivities, vn, rbf_vec_coeff_e),
        tangential_wind,
    )

    horizontal_kinetic_energy_at_edges_on_model_levels = np.where(
        k_nlev >= vertical_start,
        interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_z_kin_hor_e_numpy(
            vn, tangential_wind
        ),
        horizontal_kinetic_energy_at_edges_on_model_levels,
    )

    vn_on_half_levels[:, : nlevp1 - 1] = (
        interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges_vn_ie_numpy(wgtfac_e, vn)
    )

    if not skip_compute_predictor_vertical_advection:
        tangential_wind_on_half_levels = np.where(
            k_nlev >= vertical_start,
            interpolate_vt_to_interface_edges_numpy(wgtfac_e, tangential_wind),
            tangential_wind_on_half_levels,
        )

    contravariant_correction_at_edges_on_model_levels = np.where(
        k_nlev >= nflatlev,
        compute_contravariant_correction_numpy(vn, ddxn_z_full, ddxt_z_full, tangential_wind),
        contravariant_correction_at_edges_on_model_levels,
    )

    if not skip_compute_predictor_vertical_advection:
        w_at_vertices = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
            connectivities, w, c_intp
        )
        horizontal_advection_of_w_at_edges_on_half_levels = np.where(
            k_nlev >= vertical_start,
            compute_horizontal_advection_term_for_vertical_velocity_numpy(
                connectivities=connectivities,
                vn_ie=vn_on_half_levels[:, : nlevp1 - 1],
                inv_dual_edge_length=inv_dual_edge_length,
                w=w,
                z_vt_ie=tangential_wind_on_half_levels,
                inv_primal_edge_length=inv_primal_edge_length,
                tangent_orientation=tangent_orientation,
                z_w_v=w_at_vertices,
            ),
            horizontal_advection_of_w_at_edges_on_half_levels,
        )

    vn_on_half_levels[:, -1] = extrapolate_to_surface_numpy(wgtfacq_e, vn)

    vn_on_half_levels[:horizontal_start, :] = initial_vn_on_half_levels[:horizontal_start, :]
    vn_on_half_levels[horizontal_end:, :] = initial_vn_on_half_levels[horizontal_end:, :]

    return (
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
        horizontal_advection_of_w_at_edges_on_half_levels,
    )


def extrapolate_to_surface_numpy(wgtfacq_e: np.ndarray, vn: np.ndarray) -> np.ndarray:
    vn_k_minus_1 = vn[:, -1]
    vn_k_minus_2 = vn[:, -2]
    vn_k_minus_3 = vn[:, -3]
    wgtfacq_e_k_minus_1 = wgtfacq_e[:, -1]
    wgtfacq_e_k_minus_2 = wgtfacq_e[:, -2]
    wgtfacq_e_k_minus_3 = wgtfacq_e[:, -3]
    vn_at_surface = (
        wgtfacq_e_k_minus_1 * vn_k_minus_1
        + wgtfacq_e_k_minus_2 * vn_k_minus_2
        + wgtfacq_e_k_minus_3 * vn_k_minus_3
    )
    return vn_at_surface


@pytest.mark.embedded_remap_error
@pytest.mark.continuous_benchmarking
class TestComputeDerivedHorizontalWindsAndKEAndHorizontalAdvectionofWAndContravariantCorrection(
    stencil_tests.StencilTest
):
    PROGRAM = compute_diagnostics_from_normal_wind
    OUTPUTS = (
        "tangential_wind",
        "tangential_wind_on_half_levels",
        "vn_on_half_levels",
        "horizontal_kinetic_energy_at_edges_on_model_levels",
        "contravariant_correction_at_edges_on_model_levels",
        "horizontal_advection_of_w_at_edges_on_half_levels",
    )
    STATIC_PARAMS = {
        stencil_tests.StandardStaticVariants.NONE: (),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
            "nflatlev",
        ),
        stencil_tests.StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
            "nflatlev",
        ),
    }

    @classmethod
    def reference(
        cls,
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        tangential_wind: np.ndarray,
        tangential_wind_on_half_levels: np.ndarray,
        vn_on_half_levels: np.ndarray,
        horizontal_kinetic_energy_at_edges_on_model_levels: np.ndarray,
        contravariant_correction_at_edges_on_model_levels: np.ndarray,
        horizontal_advection_of_w_at_edges_on_half_levels: np.ndarray,
        vn: np.ndarray,
        w: np.ndarray,
        rbf_vec_coeff_e: np.ndarray,
        wgtfac_e: np.ndarray,
        ddxn_z_full: np.ndarray,
        ddxt_z_full: np.ndarray,
        wgtfacq_e: np.ndarray,
        c_intp: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        tangent_orientation: np.ndarray,
        skip_compute_predictor_vertical_advection: bool,
        nflatlev: int,
        horizontal_start: int,
        horizontal_end: int,
        vertical_start: int,
        vertical_end: int,
        **kwargs: object,
    ) -> dict:
        initial_tangential_wind = tangential_wind.copy()
        initial_tangential_wind_on_half_levels = tangential_wind_on_half_levels.copy()
        initial_horizontal_kinetic_energy_at_edges_on_model_levels = (
            horizontal_kinetic_energy_at_edges_on_model_levels.copy()
        )
        initial_contravariant_correction_at_edges_on_model_levels = (
            contravariant_correction_at_edges_on_model_levels.copy()
        )
        initial_horizontal_advection_of_w_at_edges_on_half_levels = (
            horizontal_advection_of_w_at_edges_on_half_levels.copy()
        )
        (
            tangential_wind,
            tangential_wind_on_half_levels,
            vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels,
            horizontal_advection_of_w_at_edges_on_half_levels,
        ) = compute_diagnostics_from_normal_wind_numpy(
            connectivities=connectivities,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            tangential_wind=tangential_wind,
            vn_on_half_levels=vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            vn=vn,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            wgtfac_e=wgtfac_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            wgtfacq_e=wgtfacq_e,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            w=w,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            nflatlev=nflatlev,
            nlevp1=vertical_end,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
        )

        tangential_wind[:horizontal_start, :] = initial_tangential_wind[:horizontal_start, :]
        tangential_wind[horizontal_end:, :] = initial_tangential_wind[horizontal_end:, :]

        tangential_wind_on_half_levels[:horizontal_start, :] = (
            initial_tangential_wind_on_half_levels[:horizontal_start, :]
        )
        tangential_wind_on_half_levels[horizontal_end:, :] = initial_tangential_wind_on_half_levels[
            horizontal_end:, :
        ]
        horizontal_kinetic_energy_at_edges_on_model_levels[:horizontal_start, :] = (
            initial_horizontal_kinetic_energy_at_edges_on_model_levels[:horizontal_start, :]
        )
        horizontal_kinetic_energy_at_edges_on_model_levels[horizontal_end:, :] = (
            initial_horizontal_kinetic_energy_at_edges_on_model_levels[horizontal_end:, :]
        )

        contravariant_correction_at_edges_on_model_levels[:horizontal_start, :] = (
            initial_contravariant_correction_at_edges_on_model_levels[:horizontal_start, :]
        )
        contravariant_correction_at_edges_on_model_levels[horizontal_end:, :] = (
            initial_contravariant_correction_at_edges_on_model_levels[horizontal_end:, :]
        )

        horizontal_advection_of_w_at_edges_on_half_levels[:horizontal_start, :] = (
            initial_horizontal_advection_of_w_at_edges_on_half_levels[:horizontal_start, :]
        )
        horizontal_advection_of_w_at_edges_on_half_levels[horizontal_end:, :] = (
            initial_horizontal_advection_of_w_at_edges_on_half_levels[horizontal_end:, :]
        )

        return dict(
            tangential_wind=tangential_wind,
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            vn_on_half_levels=vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
        )

    @pytest.fixture(
        params=[
            {"skip_compute_predictor_vertical_advection": value} for value in [True, False]
        ],  # True for benchmarking, False for testing
        ids=lambda param: (
            f"skip_compute_predictor_vertical_advection[{param['skip_compute_predictor_vertical_advection']}]"
        ),
    )
    def input_data(
        self, grid: base.Grid, request: pytest.FixtureRequest
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        horizontal_advection_of_w_at_edges_on_half_levels = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim
        )
        tangential_wind = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        tangential_wind_on_half_levels = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        vn_on_half_levels = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}
        )
        horizontal_kinetic_energy_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        contravariant_correction_at_edges_on_model_levels = data_alloc.random_field(
            grid, dims.EdgeDim, dims.KDim
        )
        vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        rbf_vec_coeff_e = data_alloc.random_field(grid, dims.EdgeDim, dims.E2C2EDim)
        wgtfac_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddxn_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ddxt_z_full = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim)
        tangent_orientation = data_alloc.random_field(grid, dims.EdgeDim)
        wgtfacq_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        c_intp = data_alloc.random_field(grid, dims.VertexDim, dims.V2CDim)

        nlev = grid.num_levels
        nflatlev = 5  # value is set to reflect the MCH ch1 experiment. Changing this value will change the expected runtime

        skip_compute_predictor_vertical_advection = request.param[
            "skip_compute_predictor_vertical_advection"
        ]

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
        vertical_start = 0
        vertical_end = nlev + 1

        return dict(
            tangential_wind_on_half_levels=tangential_wind_on_half_levels,
            tangential_wind=tangential_wind,
            vn_on_half_levels=vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            horizontal_advection_of_w_at_edges_on_half_levels=horizontal_advection_of_w_at_edges_on_half_levels,
            vn=vn,
            w=w,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            wgtfac_e=wgtfac_e,
            ddxn_z_full=ddxn_z_full,
            ddxt_z_full=ddxt_z_full,
            wgtfacq_e=wgtfacq_e,
            c_intp=c_intp,
            inv_dual_edge_length=inv_dual_edge_length,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
            nflatlev=nflatlev,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=vertical_start,
            vertical_end=vertical_end,
        )
