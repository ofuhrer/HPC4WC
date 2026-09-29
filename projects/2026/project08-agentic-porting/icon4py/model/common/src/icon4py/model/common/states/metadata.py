# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Final

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.states import model


INTERFACE_LEVEL_HEIGHT_STANDARD_NAME: Final[str] = "model_interface_height"

INTERFACE_LEVEL_STANDARD_NAME: Final[str] = "interface_model_level_number"

attrs: Final[dict[str, model.FieldMetaData]] = {
    "z_ifv": dict(
        standard_name="z_ifv",
        long_name="z_ifv",
        units="",
        dims=(dims.VertexDim, dims.KDim),
        icon_var_name="z_ifv",
        dtype=ta.wpfloat,
    ),
    "height_on_interface_levels": dict(
        standard_name="height_on_interface_levels",
        long_name="height_on_interface_levels",
        units="m",
        dims=(dims.CellDim, dims.KHalfDim),
        icon_var_name="z_ifc",
        dtype=ta.wpfloat,
    ),
    "z_ifc_sliced": dict(
        standard_name="z_ifc_sliced",
        long_name="z_ifc_sliced",
        units="m",
        dims=(dims.CellDim),
        icon_var_name="z_ifc_sliced",
        dtype=ta.wpfloat,
    ),
    "model_level_number": dict(
        standard_name="model_level_number",
        long_name="model level number",
        units="",
        dims=(dims.KDim,),
        icon_var_name="k_index",
        dtype=gtx.int32,
    ),
    INTERFACE_LEVEL_STANDARD_NAME: dict(
        standard_name=INTERFACE_LEVEL_STANDARD_NAME,
        long_name="model interface level number",
        units="",
        dims=(dims.KHalfDim,),
        icon_var_name="k_index",
        dtype=gtx.int32,
    ),
    "weighting_factor_for_quadratic_interpolation_to_cell_surface": dict(
        standard_name="weighting_factor_for_quadratic_interpolation_to_cell_surface",
        units="",
        dims=(dims.CellDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="wgtfacq_c_dsl",
        long_name="weighting factor for quadratic interpolation to cell surface",
    ),
    "weighting_factor_for_quadratic_interpolation_to_edge_center": dict(
        standard_name="weighting_factor_for_quadratic_interpolation_to_edge_center",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        dtype=ta.wpfloat,
        icon_var_name="wgtfacq_e_dsl",
        long_name="weighting factor for quadratic interpolation to edge centers",
    ),
    "cell_to_edge_interpolation_coefficient": dict(
        standard_name="cell_to_edge_interpolation_coefficient",
        units="",
        dims=(dims.EdgeDim, dims.E2CDim),
        dtype=ta.wpfloat,
        icon_var_name="c_lin_e",
        long_name="coefficients for cell to edge interpolation",
    ),
    "model_interface_height": dict(
        standard_name="model_interface_height",
        long_name="height value of half levels without topography",
        units="m",
        dims=(dims.KHalfDim,),
        dtype=ta.wpfloat,
        positive="up",
        icon_var_name="vct_a",
    ),
    "nudging_coefficient_on_edges": dict(
        standard_name="nudging_coefficient_on_edges",
        long_name="nudging coefficients on edges",
        units="",
        dtype=ta.wpfloat,
        dims=(dims.EdgeDim,),
        icon_var_name="nudgecoeff_e",
    ),
    "refin_e_ctrl": dict(
        standard_name="refin_e_ctrl",
        long_name="grid refinement control on edgeds",
        units="",
        dtype=int,
        dims=(dims.EdgeDim,),
        icon_var_name="refin_e_ctrl",
    ),
    "c_refin_ctrl": dict(
        standard_name="c_refin_ctrl",
        units="",
        dims=(dims.CellDim,),
        dtype=ta.wpfloat,
        icon_var_name="c_refin_ctrl",
        long_name="refinement control field on cells",
    ),
    "e_refin_ctrl": dict(
        standard_name="e_refin_ctrl",
        units="",
        dims=(dims.EdgeDim,),
        dtype=ta.wpfloat,
        icon_var_name="e_refin_ctrl",
        long_name="refinement contorl fields on edges",
    ),
    "cells_aw_verts_field": dict(
        standard_name="cells_aw_verts_field",
        units="",
        dims=(dims.VertexDim, dims.V2CDim),
        dtype=ta.wpfloat,
        icon_var_name="cells_aw_verts_field",
        long_name="grid savepoint field",
    ),
    "e_lev": dict(
        standard_name="e_lev",
        long_name="e_lev",
        units="",
        dims=(dims.EdgeDim,),
        icon_var_name="e_lev",
        dtype=gtx.int32,
    ),
    "e_owner_mask": dict(
        standard_name="e_owner_mask",
        units="",
        dims=(dims.EdgeDim,),
        dtype=bool,
        icon_var_name="e_owner_mask",
        long_name="grid savepoint field",
    ),
    "c_owner_mask": dict(
        standard_name="c_owner_mask",
        units="",
        dims=(dims.CellDim,),
        dtype=bool,
        icon_var_name="c_owner_mask",
        long_name="grid savepoint field",
    ),
}
