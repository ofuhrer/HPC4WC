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


# TODO(): revise names with domain scientists

Z_MC: Final[str] = "height"
DDQZ_Z_HALF: Final[str] = "functional_determinant_of_metrics_on_interface_levels"
DDQZ_Z_FULL: Final[str] = "functional_determinant_of_metrics_on_full_levels"
DDQZ_Z_FULL_E: Final[str] = "functional_determinant_of_metrics_on_full_levels_on_edges"
INV_DDQZ_Z_FULL: Final[str] = f"inverse_of_{DDQZ_Z_FULL}"
SCALING_FACTOR_FOR_3D_DIVDAMP: Final[str] = "scaling_factor_for_3d_divergence_damping"
RAYLEIGH_W: Final[str] = "rayleigh_w"
COEFF1_DWDZ: Final[str] = "coeff1_dwdz"
COEFF2_DWDZ: Final[str] = "coeff2_dwdz"
EXNER_REF_MC: Final[str] = "exner_ref_mc"
THETA_REF_MC: Final[str] = "theta_ref_mc"
THETA_REF_IC: Final[str] = "theta_ref_ic"
THETA_REF_ME: Final[str] = "theta_ref_me"
RHO_REF_MC: Final[str] = "rho_ref_mc"
RHO_REF_ME: Final[str] = "rho_ref_me"
D_EXNER_DZ_REF_IC: Final[str] = "d_exner_dz_ref_ic"
D2DEXDZ2_FAC1_MC: Final[str] = "d2dexdz2_fac1_mc"
D2DEXDZ2_FAC2_MC: Final[str] = "d2dexdz2_fac2_mc"
DDXT_Z_HALF_E: Final[str] = "ddxt_z_half_e"
DDXN_Z_HALF_E: Final[str] = "ddxn_z_half_e"
DDXN_Z_FULL: Final[str] = "ddxn_z_full"
DDXT_Z_FULL: Final[str] = "ddxt_z_full"
EXNER_W_IMPLICIT_WEIGHT_PARAMETER: Final[str] = (
    "implicitness_weight_for_exner_and_w_in_vertical_dycore_solver"
)
EXNER_W_EXPLICIT_WEIGHT_PARAMETER: Final[str] = (
    "explicitness_weight_for_exner_and_w_in_vertical_dycore_solver"
)
EXNER_EXFAC: Final[str] = "exner_exfac"
WGTFAC_C: Final[str] = "wgtfac_c"
WGTFAC_E: Final[str] = "wgtfac_e"
FLAT_IDX_MAX: Final[str] = "flat_idx_max"
NFLAT_GRADP: Final[str] = "nflat_gradp"
PG_EXDIST_DSL: Final[str] = "distance_for_pressure_gradient_extrapolation"
MASK_PROG_HALO_C: Final[str] = "mask_prog_halo_c"
HORIZONTAL_MASK_FOR_3D_DIVDAMP: Final[str] = "horizontal_mask_for_3d_divdamp"
ZDIFF_GRADP: Final[str] = "zdiff_gradp"
VERTOFFSET_GRADP: Final[str] = "vertoffset_gradp"
COEFF_GRADEKIN: Final[str] = "coeff_gradekin"
WGTFACQ_C: Final[str] = "weighting_factor_for_quadratic_interpolation_to_cell_surface"
WGTFACQ_E: Final[str] = "weighting_factor_for_quadratic_interpolation_to_edge_center"
MAXSLP: Final[str] = "maxslp"
MAXHGTD: Final[str] = "maxhgtd"
MAXSLP_AVG: Final[str] = "maxslp_avg"
MAXHGTD_AVG: Final[str] = "maxhgtd_avg"
MAX_NBHGT: Final[str] = "max_nbhgt"
ZD_DIFFCOEF: Final[str] = "zd_diffcoef"
ZD_INTCOEF: Final[str] = "zd_intcoef"
ZD_VERTOFFSET: Final[str] = "zd_vertoffset"
CELL_HEIGHT_ON_HALF_LEVEL: Final[str] = "vertical_coordinates_on_half_levels"
DEEPATMO_DIVH: Final[str] = "deepatmo_divh"
DEEPATMO_DIVZL: Final[str] = "deepatmo_divzL"
DEEPATMO_DIVZU: Final[str] = "deepatmo_divzU"


attrs: dict[str, model.FieldMetaData] = {
    NFLAT_GRADP: dict(
        standard_name=NFLAT_GRADP,
        long_name="number of flat edges for gradp calculation",
        units="",
        icon_var_name="nflat_gradp",
        dtype=gtx.int32,
    ),
    Z_MC: dict(
        standard_name=Z_MC,
        long_name="height",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="z_mc",
        dtype=ta.wpfloat,
    ),
    DDQZ_Z_HALF: dict(
        standard_name=DDQZ_Z_HALF,
        long_name="functional_determinant_of_metrics_on_interface_levels",
        units="",
        dims=(dims.CellDim, dims.KHalfDim),
        icon_var_name="ddqz_z_half",
        dtype=ta.wpfloat,
    ),
    DDQZ_Z_FULL: dict(
        standard_name=DDQZ_Z_FULL,
        long_name="functional determinant of the metrics (is positive), full levels",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="ddqz_z_full",
        dtype=ta.wpfloat,
    ),
    DDQZ_Z_FULL_E: dict(
        standard_name=DDQZ_Z_FULL,
        long_name="functional determinant at full level on edges",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="ddqz_z_full_e",
        dtype=ta.wpfloat,
    ),
    INV_DDQZ_Z_FULL: dict(
        standard_name=INV_DDQZ_Z_FULL,
        long_name="inv_ddqz_z_full",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="inv_ddqz_z_full",
        dtype=ta.wpfloat,
    ),
    SCALING_FACTOR_FOR_3D_DIVDAMP: dict(
        standard_name=SCALING_FACTOR_FOR_3D_DIVDAMP,
        long_name="Scaling factor for 3D divergence damping",
        units="",
        dims=(dims.KDim,),
        icon_var_name="scalfac_dd3d",
        dtype=ta.wpfloat,
    ),
    RAYLEIGH_W: dict(
        standard_name=RAYLEIGH_W,
        long_name="rayleigh_w",
        units="",
        dims=(dims.KHalfDim,),
        icon_var_name="rayleigh_w",
        dtype=ta.wpfloat,
    ),
    COEFF1_DWDZ: dict(
        standard_name=COEFF1_DWDZ,
        long_name="coeff1_dwdz",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="coeff1_dwdz",
        dtype=ta.wpfloat,
    ),
    COEFF2_DWDZ: dict(
        standard_name=COEFF2_DWDZ,
        long_name="coeff2_dwdz",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="coeff2_dwdz",
        dtype=ta.wpfloat,
    ),
    EXNER_REF_MC: dict(
        standard_name=EXNER_REF_MC,
        long_name="exner_ref_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="exner_ref_mc",
        dtype=ta.wpfloat,
    ),
    THETA_REF_MC: dict(
        standard_name=THETA_REF_MC,
        long_name="theta_ref_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="theta_ref_mc",
        dtype=ta.wpfloat,
    ),
    RHO_REF_MC: dict(
        standard_name=RHO_REF_MC,
        long_name="rho_ref_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="rho_ref_mc",
        dtype=ta.wpfloat,
    ),
    THETA_REF_IC: dict(
        standard_name=THETA_REF_IC,
        long_name="theta_ref_ic",
        units="",
        dims=(dims.CellDim, dims.KHalfDim),
        icon_var_name="theta_ref_ic",
        dtype=ta.wpfloat,
    ),
    D_EXNER_DZ_REF_IC: dict(
        standard_name=D_EXNER_DZ_REF_IC,
        long_name="d_exner_dz_ref_ic",
        units="",
        dims=(dims.CellDim, dims.KHalfDim),
        icon_var_name="d_exner_dz_ref_ic",
        dtype=ta.wpfloat,
    ),
    THETA_REF_ME: dict(
        standard_name=THETA_REF_ME,
        long_name="theta_ref_me",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="theta_ref_me",
        dtype=ta.wpfloat,
    ),
    RHO_REF_ME: dict(
        standard_name=RHO_REF_ME,
        long_name="rho_ref_me",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="rho_ref_me",
        dtype=ta.wpfloat,
    ),
    D2DEXDZ2_FAC1_MC: dict(
        standard_name=D2DEXDZ2_FAC1_MC,
        long_name="d2dexdz2_fac1_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="d2dexdz2_fac1_mc",
        dtype=ta.wpfloat,
    ),
    D2DEXDZ2_FAC2_MC: dict(
        standard_name=D2DEXDZ2_FAC2_MC,
        long_name="d2dexdz2_fac2_mc",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="d2dexdz2_fac2_mc",
        dtype=ta.wpfloat,
    ),
    DDXT_Z_HALF_E: dict(
        standard_name=DDXT_Z_HALF_E,
        long_name="ddxt_z_half_e",
        units="",
        dims=(dims.EdgeDim, dims.KHalfDim),
        icon_var_name="ddxt_z_half_e",
        dtype=ta.wpfloat,
    ),
    DDXN_Z_HALF_E: dict(
        standard_name=DDXN_Z_HALF_E,
        long_name="ddxn_z_half_e",
        units="",
        dims=(dims.EdgeDim, dims.KHalfDim),
        icon_var_name="ddxn_z_half_e",
        dtype=ta.wpfloat,
    ),
    DDXN_Z_FULL: dict(
        standard_name=DDXN_Z_FULL,
        long_name="normal_direction_of_slope",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="ddxn_z_full",
        dtype=ta.wpfloat,
    ),
    DDXT_Z_FULL: dict(
        standard_name="tangential_direction_of_slope",
        long_name="slope of the terrain (tangential direction)",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="ddxt_z_full",
        dtype=ta.wpfloat,
    ),
    EXNER_W_IMPLICIT_WEIGHT_PARAMETER: dict(
        standard_name="exner_w_implicit_weight_parameter",
        long_name="implicitness_weight_for_exner_and_w_in_vertical_dycore_solver",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="vwind_impl_wgt",
        dtype=ta.wpfloat,
    ),
    EXNER_W_EXPLICIT_WEIGHT_PARAMETER: dict(
        standard_name="exner_w_explicit_weight_parameter",
        long_name="explicitness_weight_for_exner_and_w_in_vertical_dycore_solver",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="vwind_expl_wgt",
        dtype=ta.wpfloat,
    ),
    EXNER_EXFAC: dict(
        standard_name=EXNER_EXFAC,
        long_name="exner_exfac",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="exner_exfac",
        dtype=ta.wpfloat,
    ),
    WGTFAC_C: dict(
        standard_name=WGTFAC_C,
        long_name="wgtfac_c",
        units="",
        dims=(dims.CellDim, dims.KHalfDim),
        icon_var_name="wgtfac_c",
        dtype=ta.wpfloat,
    ),
    WGTFAC_E: dict(
        standard_name=WGTFAC_E,
        long_name="wgtfac_e",
        units="",
        dims=(dims.EdgeDim, dims.KHalfDim),
        icon_var_name="wgtfac_e",
        dtype=ta.wpfloat,
    ),
    FLAT_IDX_MAX: dict(
        standard_name=FLAT_IDX_MAX,
        long_name="flat_idx_max",
        units="",
        dims=(dims.EdgeDim,),
        icon_var_name="flat_idx_max",
        dtype=ta.wpfloat,
    ),
    PG_EXDIST_DSL: dict(
        standard_name=PG_EXDIST_DSL,
        long_name="extrapolation distance for pressure gradient downward extrapolation",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="pg_exdist_dsl",
        dtype=ta.wpfloat,
    ),
    MASK_PROG_HALO_C: dict(
        standard_name=MASK_PROG_HALO_C,
        long_name="mask_prog_halo_c",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="mask_prog_halo_c",
        dtype=bool,
    ),
    HORIZONTAL_MASK_FOR_3D_DIVDAMP: dict(
        standard_name=HORIZONTAL_MASK_FOR_3D_DIVDAMP,
        long_name="horizontal mask for 3D divergence damping",
        units="",
        dims=(dims.EdgeDim,),
        icon_var_name="hmask_dd3d",
        dtype=ta.wpfloat,
    ),
    ZDIFF_GRADP: dict(
        standard_name=ZDIFF_GRADP,
        long_name="zdiff_gradp",
        units="",
        dims=(dims.EdgeDim, dims.E2CDim, dims.KDim),
        icon_var_name="zdiff_gradp",
        dtype=ta.wpfloat,
    ),
    VERTOFFSET_GRADP: dict(
        standard_name=VERTOFFSET_GRADP,
        long_name="vertoffset_gradp",
        units="",
        dims=(dims.EdgeDim, dims.E2CDim, dims.KDim),
        icon_var_name="vertoffset_gradp",
        dtype=gtx.int32,
    ),
    COEFF_GRADEKIN: dict(
        standard_name=COEFF_GRADEKIN,
        long_name="coeff_gradekin",
        units="",
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="coeff_gradekin",
        dtype=ta.wpfloat,
    ),
    WGTFACQ_C: dict(
        standard_name=WGTFACQ_C,
        long_name="weighting_factor_for_quadratic_interpolation_to_cell_surface",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="weighting_factor_for_quadratic_interpolation_to_cell_surface",
        dtype=ta.wpfloat,
    ),
    WGTFACQ_E: dict(
        standard_name=WGTFACQ_E,
        long_name="weighting_factor_for_quadratic_interpolation_to_edge_center",
        units="",
        dims=(dims.EdgeDim, dims.KDim),
        icon_var_name="weighting_factor_for_quadratic_interpolation_to_edge_center",
        dtype=ta.wpfloat,
    ),
    MAXSLP: dict(
        standard_name=MAXSLP,
        long_name="maxslp",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="z_maxslp",
        dtype=ta.wpfloat,
    ),
    MAXHGTD: dict(
        standard_name=MAXHGTD,
        long_name="maxhgtd",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="z_maxhgtd",
        dtype=ta.wpfloat,
    ),
    MAXSLP_AVG: dict(
        standard_name=MAXSLP_AVG,
        long_name="maxslp_avg",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="z_maxslp_avg",
        dtype=ta.wpfloat,
    ),
    MAXHGTD_AVG: dict(
        standard_name=MAXHGTD_AVG,
        long_name="maxhgtd_avg",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="z_maxhgtd_avg",
        dtype=ta.wpfloat,
    ),
    MAX_NBHGT: dict(
        standard_name=MAX_NBHGT,
        long_name="max_nbhgt",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="max_nbhgt",
        dtype=ta.wpfloat,
    ),
    ZD_DIFFCOEF: dict(
        standard_name=ZD_DIFFCOEF,
        long_name="zd_diffcoef",
        units="",
        dims=(dims.CellDim, dims.KDim),
        icon_var_name="zd_diffcoef",
        dtype=ta.wpfloat,
    ),
    ZD_INTCOEF: dict(
        standard_name=ZD_INTCOEF,
        long_name="zd_intcoef",
        units="",
        dims=(dims.CellDim, dims.C2E2CDim, dims.KDim),
        icon_var_name="zd_intcoef",
        dtype=ta.wpfloat,
    ),
    ZD_VERTOFFSET: dict(
        standard_name=ZD_VERTOFFSET,
        long_name="zd_vertoffset",
        units="",
        dims=(dims.CellDim, dims.C2E2CDim, dims.KDim),
        icon_var_name="zd_vertoffset",
        dtype=gtx.int32,
    ),
    CELL_HEIGHT_ON_HALF_LEVEL: dict(
        standard_name=CELL_HEIGHT_ON_HALF_LEVEL,
        long_name="vertical_coordinates_on_half_levels",
        units="m",
        dims=(dims.CellDim, dims.KHalfDim),
        icon_var_name="z_ifc",
        dtype=ta.wpfloat,
    ),
    DEEPATMO_DIVH: dict(
        standard_name=DEEPATMO_DIVH,
        long_name="",
        units="",
        dims=(dims.KDim,),
        icon_var_name="deepatmo_divh_mc",
        dtype=ta.wpfloat,
    ),
    DEEPATMO_DIVZL: dict(
        standard_name=DEEPATMO_DIVZL,
        long_name="",
        units="",
        dims=(dims.KDim,),
        icon_var_name="deepatmo_divzL_mc",
        dtype=ta.wpfloat,
    ),
    DEEPATMO_DIVZU: dict(
        standard_name=DEEPATMO_DIVZU,
        long_name="",
        units="",
        dims=(dims.KDim,),
        icon_var_name="deepatmo_divzU_mc",
        dtype=ta.wpfloat,
    ),
}
