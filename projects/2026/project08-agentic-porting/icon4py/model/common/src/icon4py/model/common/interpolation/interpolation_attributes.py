# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.states import model


C_LIN_E: Final[str] = "interpolation_coefficient_from_cell_to_edge"
C_BLN_AVG: Final[str] = "bilinear_cell_average_weight"
E_BLN_C_S: Final[str] = "bilinear_edge_cell_weight"
GEOFAC_DIV: Final[str] = "geometrical_factor_for_divergence"
GEOFAC_ROT: Final[str] = "geometrical_factor_for_curl"
GEOFAC_N2S: Final[str] = "geometrical_factor_for_nabla_2_scalar"
GEOFAC_GRDIV: Final[str] = "geometrical_factor_for_gradient_of_divergence"
GEOFAC_GRG_X: Final[str] = "geometrical_factor_for_green_gauss_gradient_x"
GEOFAC_GRG_Y: Final[str] = "geometrical_factor_for_green_gauss_gradient_y"
E_FLX_AVG: Final[str] = "e_flux_average"
POS_ON_TPLANE_E_X: Final[str] = "pos_on_tplane_e_x"
POS_ON_TPLANE_E_Y: Final[str] = "pos_on_tplane_e_y"
CELL_AW_VERTS: Final[str] = "cell_to_vertex_interpolation_factor_by_area_weighting"
NUDGECOEFFS_E: Final[str] = "nudging_coefficients_for_edges"
RBF_VEC_COEFF_C1: Final[str] = "rbf_interpolation_coefficient_cell_1"
RBF_VEC_COEFF_C2: Final[str] = "rbf_interpolation_coefficient_cell_2"
RBF_VEC_COEFF_E: Final[str] = "rbf_interpolation_coefficient_edge"
RBF_VEC_COEFF_V1: Final[str] = "rbf_interpolation_coefficient_vertex_1"
RBF_VEC_COEFF_V2: Final[str] = "rbf_interpolation_coefficient_vertex_2"
RBF_SCALE_CELL: Final[str] = "rbf_scale_cell"
RBF_SCALE_EDGE: Final[str] = "rbf_scale_edge"
RBF_SCALE_VERTEX: Final[str] = "rbf_scale_vertex"
LSQ_PSEUDOINV: Final[str] = "lsq_interpolation_coefficient"

attrs: dict[str, model.FieldMetaData] = {
    C_LIN_E: dict(
        standard_name=C_LIN_E,
        long_name="interpolation coefficient from cell to edges",
        units="",  # TODO(halungge): check or confirm
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="c_lin_e",
        dtype=ta.wpfloat,
    ),
    NUDGECOEFFS_E: dict(
        standard_name=NUDGECOEFFS_E,
        long_name="nudging_coefficients_for_edges",
        units="",  # TODO(yiluchen1066): : need to check unit
        dims=(dims.EdgeDim,),
        icon_var_name="nudgecoeff_e",
        dtype=ta.wpfloat,
    ),
    C_BLN_AVG: dict(
        standard_name=C_BLN_AVG,
        long_name="mass conserving bilinear cell average weight",
        units="",  # TODO(halungge): check or confirm
        dims=(dims.EdgeDim, dims.C2E2CODim),
        icon_var_name="c_bln_avg",
        dtype=ta.wpfloat,
    ),
    E_BLN_C_S: dict(
        standard_name=E_BLN_C_S,
        long_name="mass conserving bilinear edge cell weight",
        units="",  # TODO(): check or confirm
        dims=(dims.CellDim, dims.C2EDim),
        icon_var_name="e_bln_c_s",
        dtype=ta.wpfloat,
    ),
    GEOFAC_DIV: dict(
        standard_name=GEOFAC_DIV,
        long_name="geometrical factor for divergence",  # TODO(halungge): find proper description
        units="",  # TODO(halungge): check or confirm
        dims=(dims.CellDim, dims.C2EDim),
        icon_var_name="geofac_div",
        dtype=ta.wpfloat,
    ),
    GEOFAC_ROT: dict(
        standard_name=GEOFAC_ROT,
        long_name="geometrical factor for curl",
        units="",  # TODO(halungge): check or confirm
        dims=(dims.VertexDim, dims.V2EDim),
        icon_var_name="geofac_rot",
        dtype=ta.wpfloat,
    ),
    GEOFAC_N2S: dict(
        standard_name=GEOFAC_N2S,
        long_name="geometrical factor nabla-2 scalar",
        units="",  # TODO(halungge): check or confirm
        dims=(dims.CellDim, dims.C2E2CODim),
        icon_var_name="geofac_n2s",
        dtype=ta.wpfloat,
    ),
    GEOFAC_GRDIV: dict(
        standard_name=GEOFAC_GRDIV,
        long_name="geometrical factor for gradient of divergence",
        units="",  # TODO(halungge): check or confirm
        dims=(dims.EdgeDim, dims.E2C2EODim),
        icon_var_name="geofac_grdiv",
        dtype=ta.wpfloat,
    ),
    GEOFAC_GRG_X: dict(
        standard_name=GEOFAC_GRG_X,
        long_name="geometrical factor for Green Gauss gradient (first component)",
        units="",  # TODO(halungge): check or confirm
        dims=(dims.CellDim, dims.C2E2CODim),
        icon_var_name="geofac_grg",
        dtype=ta.wpfloat,
    ),
    GEOFAC_GRG_Y: dict(
        standard_name=GEOFAC_GRG_Y,
        long_name="geometrical factor for Green Gauss gradient (second component)",
        units="",  # TODO(halungge): check or confirm
        dims=(dims.CellDim, dims.C2E2CODim),
        icon_var_name="geofac_grg",
        dtype=ta.wpfloat,
    ),
    E_FLX_AVG: dict(
        standard_name=E_FLX_AVG,
        long_name="e flux average",
        units="",  # TODO(): check or confirm
        dims=(dims.EdgeDim, dims.E2C2EODim),
        icon_var_name="e_flx_avg",
        dtype=ta.wpfloat,
    ),
    POS_ON_TPLANE_E_X: dict(
        standard_name=POS_ON_TPLANE_E_X,
        long_name="position on tplane x",
        units="",  # TODO(): check or confirm
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="pos_on_tplane_e_x",
        dtype=ta.wpfloat,
    ),
    POS_ON_TPLANE_E_Y: dict(
        standard_name=POS_ON_TPLANE_E_Y,
        long_name="position on tplane y",
        units="",  # TODO(): check or confirm
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="pos_on_tplane_e_y",
        dtype=ta.wpfloat,
    ),
    CELL_AW_VERTS: dict(
        standard_name=CELL_AW_VERTS,
        long_name="coefficient for interpolation from cells to verts by area weighting",
        units="",
        dims=(dims.VertexDim, dims.V2CDim),
        icon_var_name="cells_aw_verts",
        dtype=ta.wpfloat,
    ),
    RBF_VEC_COEFF_C1: dict(
        standard_name=RBF_VEC_COEFF_C1,
        long_name="rbf interpolation coefficient from cells to edges 1",
        units="",
        dims=(dims.CellDim, dims.C2E2C2EDim),
        icon_var_name="rbf_vec_coeff_c1",
        dtype=ta.wpfloat,
    ),
    RBF_VEC_COEFF_C2: dict(
        standard_name=RBF_VEC_COEFF_C2,
        long_name="rbf interpolation coefficient from cells to edges 2",
        units="",
        dims=(dims.CellDim, dims.C2E2C2EDim),
        icon_var_name="rbf_vec_coeff_c2",
        dtype=ta.wpfloat,
    ),
    RBF_VEC_COEFF_E: dict(
        standard_name=RBF_VEC_COEFF_E,
        long_name="rbf interpolation coefficient from edges to edges",
        units="",
        dims=(dims.EdgeDim, dims.E2C2EDim),
        icon_var_name="rbf_vec_coeff_e",
        dtype=ta.wpfloat,
    ),
    RBF_VEC_COEFF_V1: dict(
        standard_name=RBF_VEC_COEFF_V1,
        long_name="rbf interpolation coefficient from vertices to edges 1",
        units="",
        dims=(dims.VertexDim, dims.V2EDim),
        icon_var_name="rbf_vec_coeff_v1",
        dtype=ta.wpfloat,
    ),
    RBF_VEC_COEFF_V2: dict(
        standard_name=RBF_VEC_COEFF_V2,
        long_name="rbf interpolation coefficient from vertices to edges 2",
        units="",
        dims=(dims.VertexDim, dims.V2EDim),
        icon_var_name="rbf_vec_coeff_v2",
        dtype=ta.wpfloat,
    ),
    RBF_SCALE_CELL: dict(
        standard_name=RBF_SCALE_CELL,
        long_name="rbf_scale_cell",
        units="",
        dims=(),
        icon_var_name="rbf_vec_scale_c",
        dtype=ta.wpfloat,
    ),
    RBF_SCALE_EDGE: dict(
        standard_name=RBF_SCALE_EDGE,
        long_name="rbf_scale_edge",
        units="",
        dims=(),
        icon_var_name="rbf_vec_scale_e",
        dtype=ta.wpfloat,
    ),
    RBF_SCALE_VERTEX: dict(
        standard_name=RBF_SCALE_VERTEX,
        long_name="rbf_scale_vertex",
        units="",
        dims=(),
        icon_var_name="rbf_vec_scale_v",
        dtype=ta.wpfloat,
    ),
    LSQ_PSEUDOINV: dict(
        standard_name=LSQ_PSEUDOINV,
        long_name="pseudo (or Moore-Penrose) inverse of least squares matrix A",
        units="",
        dims=(dims.CellDim, dims.C2E2CDim),
        icon_var_name="ptr_int_lsq%lsq_pseudoinv",
        dtype=ta.wpfloat,
    ),
}
