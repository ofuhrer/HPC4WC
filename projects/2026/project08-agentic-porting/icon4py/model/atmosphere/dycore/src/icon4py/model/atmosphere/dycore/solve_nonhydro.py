# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Any, Final

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next import common as gtx_common

import icon4py.model.atmosphere.dycore.solve_nonhydro_stencils as nhsolve_stencils
import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.dycore import dycore_states, dycore_utils
from icon4py.model.atmosphere.dycore.stencils import (
    compute_cell_diagnostics_for_dycore,
    compute_edge_diagnostics_for_dycore_and_update_vn,
    vertically_implicit_dycore_solver,
)
from icon4py.model.atmosphere.dycore.stencils.compute_dwdz_for_divergence_damping import (
    compute_dwdz_for_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.compute_exner_from_rhotheta import (
    compute_exner_from_rhotheta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_velocity_quantities import (
    compute_averaged_vn_and_fluxes,
    compute_horizontal_velocity_quantities_and_fluxes,
)
from icon4py.model.atmosphere.dycore.stencils.compute_hydrostatic_correction_term import (
    compute_hydrostatic_correction_term,
)
from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.atmosphere.dycore.stencils.update_mass_flux_weighted import (
    update_mass_flux_weighted,
)
from icon4py.model.atmosphere.dycore.stencils.update_theta_and_exner_in_halo import (
    update_theta_and_exner_in_halo,
)
from icon4py.model.atmosphere.dycore.velocity_advection import VelocityAdvection
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    base as grid_def,
    horizontal as h_grid,
    icon as icon_grid,
    vertical as v_grid,
)
from icon4py.model.common.math import smagorinsky
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)


@dataclasses.dataclass
class IntermediateFields:
    """
    Encapsulate internal fields of SolveNonHydro that contain shared state over predictor and corrector step.

    Encapsulates internal fields used in SolveNonHydro. Fields (and the class!)
    follow the naming convention of ICON to prepend local fields of a module with z_. Contrary to
    other such z_ fields inside SolveNonHydro the fields in this dataclass
    contain state that is built up over the predictor and corrector part in a timestep.
    """

    horizontal_pressure_gradient: fa.EdgeKField[ta.anyfloat]
    """
    Declared as z_gradh_exner in ICON.
    """
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat]
    """
    Declared as z_rho_e in ICON.
    """
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat]
    """
    Declared as z_theta_v_e in ICON.
    """
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.anyfloat]
    """
    Declared as z_kin_hor_e in ICON.
    """
    tangential_wind_on_half_levels: fa.EdgeKField[ta.anyfloat]
    """
    Declared as z_vt_ie in ICON. Tangential wind at edge on k-half levels. NOTE THAT IT ONLY HAS nlev LEVELS because it is only used for computing horizontal advection of w and thus level nlevp1 is not needed because w[nlevp1-1] is diagnostic.
    """
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[ta.anyfloat]
    """
    Declared as z_graddiv_vn in ICON.
    """
    dwdz_at_cells_on_model_levels: fa.CellKField[ta.anyfloat]
    """
    Declared as z_dwdz_dd in ICON.
    """

    @classmethod
    def allocate(
        cls, grid: grid_def.Grid, allocator: gtx_typing.Allocator | None
    ) -> IntermediateFields:
        return IntermediateFields(
            horizontal_pressure_gradient=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, allocator=allocator
            ),
            rho_at_edges_on_model_levels=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, allocator=allocator
            ),
            theta_v_at_edges_on_model_levels=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, allocator=allocator
            ),
            horizontal_gradient_of_normal_wind_divergence=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, allocator=allocator
            ),
            dwdz_at_cells_on_model_levels=data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, allocator=allocator
            ),
            horizontal_kinetic_energy_at_edges_on_model_levels=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, allocator=allocator
            ),
            tangential_wind_on_half_levels=data_alloc.zero_field(
                grid, dims.EdgeDim, dims.KDim, allocator=allocator
            ),
        )


@dataclasses.dataclass(kw_only=True)
class NonHydrostaticConfig:
    """
    Contains necessary parameter to configure a nonhydro run.

    Encapsulates namelist parameters and derived parameters.
    TODO: (magdalena) values should be read from a configuration file.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    itime_scheme: typing.Annotated[
        dycore_states.TimeSteppingScheme,
        common_conf_opt.ConfigOption(
            description="Options for predictor-corrector time-stepping scheme.",
            icon_equivalent=common_conf_opt.IconOption(
                name="itime_scheme", path=("nonhydrostatic_nml",)
            ),
        ),
    ] = dycore_states.TimeSteppingScheme.MOST_EFFICIENT

    iadv_rhotheta: typing.Annotated[
        dycore_states.RhoThetaAdvectionType,
        common_conf_opt.ConfigOption(
            description="Advection method for rho and theta.",
            icon_equivalent=common_conf_opt.IconOption(
                name="iadv_rhotheta", path=("nonhydrostatic_nml",)
            ),
        ),
    ] = dycore_states.RhoThetaAdvectionType.MIURA

    igradp_method: typing.Annotated[
        dycore_states.HorizontalPressureDiscretizationType,
        common_conf_opt.ConfigOption(
            description=("Discretization of horizontal pressure gradient."),
            icon_equivalent=common_conf_opt.IconOption(
                name="igradp_method", path=("nonhydrostatic_nml",)
            ),
        ),
    ] = dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO

    rayleigh_type: typing.Annotated[
        constants.RayleighType,
        common_conf_opt.ConfigOption(
            description="Type of Rayleigh damping.",
            icon_equivalent=common_conf_opt.IconOption(
                name="rayleigh_type",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = constants.RayleighType.KLEMP

    divdamp_order: typing.Annotated[
        dycore_states.DivergenceDampingOrder,
        common_conf_opt.ConfigOption(
            description="Order of divergence damping.",
            icon_equivalent=common_conf_opt.IconOption(
                name="divdamp_order",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = dycore_states.DivergenceDampingOrder.COMBINED  # the ICON default is 4,

    divdamp_type: typing.Annotated[
        dycore_states.DivergenceDampingType,
        common_conf_opt.ConfigOption(
            description="Type of divergence damping.",
            icon_equivalent=common_conf_opt.IconOption(
                name="divdamp_type",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = dycore_states.DivergenceDampingType.THREE_DIMENSIONAL

    l_vert_nested: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="Whether to use vertical nesting (variable number of vertical levels).",
            icon_equivalent=common_conf_opt.IconOption(
                name="lvert_nest",
                path=("run_nml",),
            ),
        ),
    ] = False

    deepatmos_mode: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="Deep atmosphere mode.",
            icon_equivalent=common_conf_opt.IconOption(
                name="ldeepatmo",
                path=("dynamics_nml",),
            ),
        ),
    ] = False

    iau_init: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="Start from DWD analysis with incremental analysis update.",
            icon_equivalent=common_conf_opt.IconOption(
                name="init_mode",
                path=("initicon_nml",),
                converter=lambda init_mode: bool(init_mode == 5),
            ),
        ),
    ] = False

    extra_diffu: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description=(
                "Apply additional momentum diffusion at grid points close to the stability "
                "limit for vertical advection."
            ),
            icon_equivalent=common_conf_opt.IconOption(
                name="lextra_diffu",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = True

    rhotheta_offctr: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description=(
                "Off-centering of density and potential temperature at interface level."
                "Specifying a negative value here reduces the amount of vertical "
                "wind off-centering needed for stability of sound waves."
            ),
            icon_equivalent=common_conf_opt.IconOption(
                name="rhotheta_offctr",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = -0.1

    veladv_offctr: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Off-centering of velocity advection in corrector step.",
            icon_equivalent=common_conf_opt.IconOption(
                name="veladv_offctr",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = 0.25

    max_nudging_coefficient: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Maximum relaxation coefficient for lateral boundary nudging",
            icon_equivalent=common_conf_opt.IconOption(
                name="nudge_max_coeff", path=("interpol_nml",), read_from_icon=False
            ),
        ),
    ] = constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO * 0.02

    # TODO(muellch): The four divdamp factors and heights should be in one or two dataclasses.
    fourth_order_divdamp_factor: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for divergence damping at height 'fourth_order_divdamp_z' and below.",
            icon_equivalent=common_conf_opt.IconOption(
                name="divdamp_fac",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = 0.0025

    fourth_order_divdamp_factor2: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for divergence damping at height 'fourth_order_divdamp_z2'.",
            icon_equivalent=common_conf_opt.IconOption(
                name="divdamp_fac2",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = 0.004

    fourth_order_divdamp_factor3: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for divergence damping at height 'fourth_order_divdamp_z3'.",
            icon_equivalent=common_conf_opt.IconOption(
                name="divdamp_fac3",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = 0.004

    fourth_order_divdamp_factor4: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for divergence damping at height 'fourth_order_divdamp_z4 and higher'.",
            icon_equivalent=common_conf_opt.IconOption(
                name="divdamp_fac4",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = 0.004

    fourth_order_divdamp_z: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description=(
                "Height up to which divdamp_fac is used, and where the linear profile "
                "up to height 'fourth_order_divdamp_z2' starts"
            ),
            icon_equivalent=common_conf_opt.IconOption(
                name="divdamp_z",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = 32500.0

    fourth_order_divdamp_z2: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description=(
                "Height with scaling factor 'fourth_order_divdamp_factor2' where the linear profile starting at "
                "'fourth_order_divdamp_z' ends, and where the quadratic profile up to 'fourth_order_divdamp_z4' starts."
            ),
            icon_equivalent=common_conf_opt.IconOption(
                name="divdamp_z2",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = 40000.0

    fourth_order_divdamp_z3: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description=(
                "Height with scaling factor 'fourth_order_divdamp_factor3'. Needed to determine the quadratic function "
                " between 'fourth_order_divdamp_z2' and 'fourth_order_divdamp_z4'."
            ),
            icon_equivalent=common_conf_opt.IconOption(
                name="divdamp_z3",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = 60000.0

    fourth_order_divdamp_z4: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Height from which scaling factor 'fourth_order_divdamp_factor4' is used.",
            icon_equivalent=common_conf_opt.IconOption(
                name="divdamp_z4",
                path=("nonhydrostatic_nml",),
            ),
        ),
    ] = 80000.0

    def __post_init__(self) -> None:
        self._validate()

    @classmethod
    def from_fortran_dict(cls, atmo_dict: dict[str, Any], **overrides: Any) -> NonHydrostaticConfig:
        return common_conf_opt.construct_config_from_icon(cls, atmo_dict, **overrides)

    def _validate(self) -> None:
        """Apply consistency checks and validation on configuration parameters."""

        if self.l_vert_nested:
            raise NotImplementedError("Vertical nesting support not implemented")

        if self.deepatmos_mode:
            raise NotImplementedError("Deep atmosphere mode not implemented")

        if self.igradp_method != dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
            raise NotImplementedError("igradp_method can only be 3")

        if self.itime_scheme != dycore_states.TimeSteppingScheme.MOST_EFFICIENT:
            raise NotImplementedError("itime_scheme can only be 4")

        if self.iadv_rhotheta != dycore_states.RhoThetaAdvectionType.MIURA:
            raise NotImplementedError("iadv_rhotheta can only be 2 (Miura scheme)")

        if self.divdamp_type == dycore_states.DivergenceDampingType.TWO_DIMENSIONAL:
            raise NotImplementedError(
                "`DivergenceDampingType.TWO_DIMENSIONAL` (2) is not yet implemented"
            )

        if self.rayleigh_type != constants.RayleighType.KLEMP:
            raise NotImplementedError(
                "Only Klemp type of the Rayleigh damping (nudging vertical wind towards zero) is implemented."
            )

        if not self.extra_diffu:
            raise NotImplementedError(
                "extra_diffu=False is not supported; only True is implemented"
            )


class NonHydrostaticParams:
    """Calculates derived quantities depending on the NonHydrostaticConfig."""

    def __init__(self, config: NonHydrostaticConfig):
        #: Weighting coefficients for velocity advection if tendency averaging is used
        #: The off-centering specified here turned out to be beneficial to numerical
        #: stability in extreme situations
        self.advection_explicit_weight_parameter: Final[float] = 0.5 - config.veladv_offctr
        """
        Declared as wgt_nnow_vel in ICON.
        """
        self.advection_implicit_weight_parameter: Final[float] = 0.5 + config.veladv_offctr
        """
        Declared as wgt_nnew_vel in ICON.
        """

        #: Weighting coefficients for rho and theta at interface levels in the corrector step
        #: This empirically determined weighting minimizes the vertical wind off-centering
        #: needed for numerical stability of vertical sound wave propagation
        self.rhotheta_implicit_weight_parameter: Final[float] = 0.5 + config.rhotheta_offctr
        """
        Declared as wgt_nnew_rth in ICON.
        """
        self.rhotheta_explicit_weight_parameter: Final[float] = (
            1.0 - self.rhotheta_implicit_weight_parameter
        )
        """
        Declared as wgt_nnow_rth in ICON.
        """


class SolveNonhydro:
    def __init__(
        self,
        *,
        grid: icon_grid.IconGrid,
        config: NonHydrostaticConfig,
        params: NonHydrostaticParams,
        metric_state_nonhydro: dycore_states.MetricStateNonHydro,
        interpolation_state: dycore_states.InterpolationState,
        vertical_params: v_grid.VerticalGrid,
        edge_geometry: grid_states.EdgeParams,
        cell_geometry: grid_states.CellParams,
        owner_mask: fa.CellField[bool],
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
        exchange: decomposition.ExchangeRuntime,
    ):
        self._exchange = exchange

        self._grid = grid
        self._config = config
        self._params = params
        self._metric_state_nonhydro = metric_state_nonhydro
        self._interpolation_state = interpolation_state
        self._vertical_params = vertical_params
        self._edge_geometry = edge_geometry
        self._cell_params = cell_geometry
        self._determine_local_domains()

        self._compute_exner_from_rhotheta_in_lateral_boundary = setup_program(
            backend=backend,
            program=compute_exner_from_rhotheta,
            constant_args={
                "rd_o_cvd": constants.RD_O_CVD,
                "rd_o_p0ref": constants.RD_O_P0REF,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary,
                "horizontal_end": self._end_cell_lateral_boundary_level_4,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
        )

        self._update_theta_and_exner_in_halo = setup_program(
            backend=backend,
            program=update_theta_and_exner_in_halo,
            constant_args={
                "mask_prog_halo_c": self._metric_state_nonhydro.mask_prog_halo_c,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_halo,
                "horizontal_end": self._end_cell_end,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
        )

        self._compute_hydrostatic_correction_term = setup_program(
            backend=backend,
            program=compute_hydrostatic_correction_term,
            constant_args={
                "ikoffset": self._metric_state_nonhydro.vertoffset_gradp,
                "zdiff_gradp": self._metric_state_nonhydro.zdiff_gradp,
                "inv_ddqz_z_full": self._metric_state_nonhydro.inv_ddqz_z_full,
                "inv_dual_edge_length": self._edge_geometry.inverse_dual_edge_lengths,
                "grav_o_cpd": constants.GRAV_O_CPD,
            },
            horizontal_sizes={
                "horizontal_start": self._start_edge_nudging_level_2,
                "horizontal_end": self._end_edge_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(self._grid.num_levels - 1),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        self._compute_rho_theta_pgrad_and_update_vn = setup_program(
            backend=backend,
            program=compute_edge_diagnostics_for_dycore_and_update_vn.compute_rho_theta_pgrad_and_update_vn,
            constant_args={
                "reference_rho_at_edges_on_model_levels": self._metric_state_nonhydro.reference_rho_at_edges_on_model_levels,
                "reference_theta_at_edges_on_model_levels": self._metric_state_nonhydro.reference_theta_at_edges_on_model_levels,
                "geofac_grg_x": self._interpolation_state.geofac_grg_x,
                "geofac_grg_y": self._interpolation_state.geofac_grg_y,
                "pos_on_tplane_e_x": self._interpolation_state.pos_on_tplane_e_1,
                "pos_on_tplane_e_y": self._interpolation_state.pos_on_tplane_e_2,
                "primal_normal_cell_x": self._edge_geometry.primal_normal_cell[0],
                "dual_normal_cell_x": self._edge_geometry.dual_normal_cell[0],
                "primal_normal_cell_y": self._edge_geometry.primal_normal_cell[1],
                "dual_normal_cell_y": self._edge_geometry.dual_normal_cell[1],
                "ddxn_z_full": self._metric_state_nonhydro.ddxn_z_full,
                "c_lin_e": self._interpolation_state.c_lin_e,
                "ikoffset": self._metric_state_nonhydro.vertoffset_gradp,
                "zdiff_gradp": self._metric_state_nonhydro.zdiff_gradp,
                "pg_exdist": self._metric_state_nonhydro.pg_exdist,
                "inv_dual_edge_length": self._edge_geometry.inverse_dual_edge_lengths,
                "limited_area": self._grid.limited_area,
            },
            variants={"is_iau_active": [False, True] if self._config.iau_init else [False]},
            horizontal_sizes={
                "start_edge_lateral_boundary": self._start_edge_lateral_boundary,
                "start_edge_lateral_boundary_level_7": self._start_edge_lateral_boundary_level_7,
                "start_edge_nudging_level_2": self._start_edge_nudging_level_2,
                "end_edge_nudging": self._end_edge_nudging,
                "end_edge_local": self._end_edge_local,
                "end_edge_halo": self._end_edge_halo,
                "horizontal_start": gtx.int32(0),
                "horizontal_end": self._end_edge_halo_level_2,
            },
            vertical_sizes={
                "nflatlev": self._vertical_params.nflatlev,
                "nflat_gradp": self._metric_state_nonhydro.nflat_gradp,
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        assert self._cell_params.mean_cell_area is not None

        self._apply_divergence_damping_and_update_vn = setup_program(
            backend=backend,
            program=compute_edge_diagnostics_for_dycore_and_update_vn.apply_divergence_damping_and_update_vn,
            constant_args={
                "horizontal_mask_for_3d_divdamp": self._metric_state_nonhydro.horizontal_mask_for_3d_divdamp,
                "scaling_factor_for_3d_divdamp": self._metric_state_nonhydro.scaling_factor_for_3d_divdamp,
                "inv_dual_edge_length": self._edge_geometry.inverse_dual_edge_lengths,
                "nudgecoeff_e": self._interpolation_state.nudgecoeff_e,
                "geofac_grdiv": self._interpolation_state.geofac_grdiv,
                "advection_explicit_weight_parameter": self._params.advection_explicit_weight_parameter,
                "advection_implicit_weight_parameter": self._params.advection_implicit_weight_parameter,
                "limited_area": self._grid.limited_area,
                "divdamp_order": gtx.int32(self._config.divdamp_order),
                "mean_cell_area": self._cell_params.mean_cell_area,
                "max_nudging_coefficient": self._config.max_nudging_coefficient,
                "dbl_eps": constants.DBL_EPS,
            },
            variants={
                "apply_2nd_order_divergence_damping": [False, True],
                "apply_4th_order_divergence_damping": [False, True],
                "is_iau_active": [False, True] if self._config.iau_init else [False],
            },
            horizontal_sizes={
                "horizontal_start": gtx.int32(self._start_edge_nudging_level_2),
                "horizontal_end": self._end_edge_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        self._compute_horizontal_velocity_quantities_and_fluxes = setup_program(
            backend=backend,
            program=compute_horizontal_velocity_quantities_and_fluxes,
            constant_args={
                "ddqz_z_full_e": self._metric_state_nonhydro.ddqz_z_full_e,
                "ddxn_z_full": self._metric_state_nonhydro.ddxn_z_full,
                "ddxt_z_full": self._metric_state_nonhydro.ddxt_z_full,
                "wgtfac_e": self._metric_state_nonhydro.wgtfac_e,
                "wgtfacq_e": self._metric_state_nonhydro.wgtfacq_e,
                "e_flx_avg": self._interpolation_state.e_flx_avg,
                "geofac_grdiv": self._interpolation_state.geofac_grdiv,
                "rbf_vec_coeff_e": self._interpolation_state.rbf_vec_coeff_e,
            },
            horizontal_sizes={
                "horizontal_start": gtx.int32(self._start_edge_lateral_boundary_level_5),
                "horizontal_end": self._end_edge_halo_level_2,
            },
            vertical_sizes={
                "nflatlev": self._vertical_params.nflatlev,
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )

        self._compute_averaged_vn_and_fluxes = setup_program(
            backend=backend,
            program=compute_averaged_vn_and_fluxes,
            constant_args={
                "e_flx_avg": self._interpolation_state.e_flx_avg,
                "ddqz_z_full_e": self._metric_state_nonhydro.ddqz_z_full_e,
            },
            variants={
                "at_first_substep": [False, True],
                "prepare_advection": [False, True],
            },
            horizontal_sizes={
                "horizontal_start": gtx.int32(self._start_edge_lateral_boundary_level_5),
                "horizontal_end": self._end_edge_halo_level_2,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        self._vertically_implicit_solver_at_predictor_step = setup_program(
            backend=backend,
            program=vertically_implicit_dycore_solver.vertically_implicit_solver_at_predictor_step,
            constant_args={
                "geofac_div": self._interpolation_state.geofac_div,
                "exner_w_explicit_weight_parameter": self._metric_state_nonhydro.exner_w_explicit_weight_parameter,
                "inv_ddqz_z_full": self._metric_state_nonhydro.inv_ddqz_z_full,
                "exner_w_implicit_weight_parameter": self._metric_state_nonhydro.exner_w_implicit_weight_parameter,
                "ddqz_z_half": self._metric_state_nonhydro.ddqz_z_half,
                "reference_exner_at_cells_on_model_levels": self._metric_state_nonhydro.reference_exner_at_cells_on_model_levels,
                "e_bln_c_s": self._interpolation_state.e_bln_c_s,
                "wgtfac_c": self._metric_state_nonhydro.wgtfac_c,
                "wgtfacq_c": self._metric_state_nonhydro.wgtfacq_c,
                "rayleigh_type": self._config.rayleigh_type,
                "divdamp_type": self._config.divdamp_type,
            },
            variants={
                "at_first_substep": [False, True],
                "is_iau_active": [False, True] if self._config.iau_init else [False],
            },
            horizontal_sizes={
                "start_cell_index_nudging": self._start_cell_nudging,
                "end_cell_index_local": self._end_cell_local,
                "start_cell_index_lateral_lvl3": self._start_cell_lateral_boundary_level_3,
                "end_cell_index_halo_lvl1": self._end_cell_halo,
            },
            vertical_sizes={
                "end_index_of_damping_layer": self._vertical_params.end_index_of_damping_layer,
                "kstart_moist": self._vertical_params.kstart_moist,
                "flat_level_index_plus1": gtx.int32(self._vertical_params.nflatlev + 1),
                "vertical_start_index_model_top": gtx.int32(0),
                "vertical_end_index_model_surface": gtx.int32(self._grid.num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )

        self._vertically_implicit_solver_at_corrector_step = setup_program(
            backend=backend,
            program=vertically_implicit_dycore_solver.vertically_implicit_solver_at_corrector_step,
            constant_args={
                "exner_w_explicit_weight_parameter": self._metric_state_nonhydro.exner_w_explicit_weight_parameter,
                "inv_ddqz_z_full": self._metric_state_nonhydro.inv_ddqz_z_full,
                "exner_w_implicit_weight_parameter": self._metric_state_nonhydro.exner_w_implicit_weight_parameter,
                "ddqz_z_half": self._metric_state_nonhydro.ddqz_z_half,
                "reference_exner_at_cells_on_model_levels": self._metric_state_nonhydro.reference_exner_at_cells_on_model_levels,
                "advection_explicit_weight_parameter": self._params.advection_explicit_weight_parameter,
                "advection_implicit_weight_parameter": self._params.advection_implicit_weight_parameter,
                "rayleigh_type": self._config.rayleigh_type,
            },
            variants={
                "at_first_substep": [False, True],
                "at_last_substep": [False, True],
                "lprep_adv": [False, True],
                "is_iau_active": [False, True] if self._config.iau_init else [False],
            },
            horizontal_sizes={
                "start_cell_index_nudging": self._start_cell_nudging,
                "end_cell_index_local": self._end_cell_local,
            },
            vertical_sizes={
                "end_index_of_damping_layer": self._vertical_params.end_index_of_damping_layer,
                "kstart_moist": self._vertical_params.kstart_moist,
                "vertical_start_index_model_top": gtx.int32(0),
                "vertical_end_index_model_surface": gtx.int32(self._grid.num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )

        self._compute_dwdz_for_divergence_damping = setup_program(
            backend=backend,
            program=compute_dwdz_for_divergence_damping,
            constant_args={
                "inv_ddqz_z_full": self._metric_state_nonhydro.inv_ddqz_z_full,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary,
                "horizontal_end": self._end_cell_lateral_boundary_level_4,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        self._init_cell_kdim_field_with_zero_wp = setup_program(
            backend=backend,
            program=init_cell_kdim_field_with_zero_wp,
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary,
                "horizontal_end": self._end_cell_lateral_boundary_level_4,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels + 1),
            },
        )
        self._update_mass_flux_weighted = setup_program(
            backend=backend,
            program=update_mass_flux_weighted,
            constant_args={
                "vwind_expl_wgt": self._metric_state_nonhydro.exner_w_explicit_weight_parameter,
                "vwind_impl_wgt": self._metric_state_nonhydro.exner_w_implicit_weight_parameter,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary,
                "horizontal_end": self._end_cell_lateral_boundary_level_4,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
        )
        self._compute_rayleigh_damping_factor = setup_program(
            backend=backend,
            program=dycore_utils.compute_rayleigh_damping_factor,
            constant_args={
                "rayleigh_w": self._metric_state_nonhydro.rayleigh_w,
            },
        )

        self._compute_perturbed_quantities_and_interpolation = setup_program(
            backend=backend,
            program=compute_cell_diagnostics_for_dycore.compute_perturbed_quantities_and_interpolation,
            constant_args={
                "reference_rho_at_cells_on_model_levels": self._metric_state_nonhydro.reference_rho_at_cells_on_model_levels,
                "reference_theta_at_cells_on_model_levels": self._metric_state_nonhydro.reference_theta_at_cells_on_model_levels,
                "reference_theta_at_cells_on_half_levels": self._metric_state_nonhydro.reference_theta_at_cells_on_half_levels,
                "wgtfacq_c": self._metric_state_nonhydro.wgtfacq_c,
                "wgtfac_c": self._metric_state_nonhydro.wgtfac_c,
                "exner_w_explicit_weight_parameter": self._metric_state_nonhydro.exner_w_explicit_weight_parameter,
                "ddz_of_reference_exner_at_cells_on_half_levels": self._metric_state_nonhydro.ddz_of_reference_exner_at_cells_on_half_levels,
                "ddqz_z_half": self._metric_state_nonhydro.ddqz_z_half,
                "time_extrapolation_parameter_for_exner": self._metric_state_nonhydro.time_extrapolation_parameter_for_exner,
                "reference_exner_at_cells_on_model_levels": self._metric_state_nonhydro.reference_exner_at_cells_on_model_levels,
                "inv_ddqz_z_full": self._metric_state_nonhydro.inv_ddqz_z_full,
                "d2dexdz2_fac1_mc": self._metric_state_nonhydro.d2dexdz2_fac1_mc,
                "d2dexdz2_fac2_mc": self._metric_state_nonhydro.d2dexdz2_fac2_mc,
                "igradp_method": self._config.igradp_method,
            },
            horizontal_sizes={
                "start_cell_lateral_boundary_level_3": self._start_cell_lateral_boundary_level_3,
                "start_cell_halo_level_2": self._start_cell_halo_level_2,
                "end_cell_halo": self._end_cell_halo,
                "end_cell_halo_level_2": self._end_cell_halo_level_2,
                "start_cell_lateral_boundary": self._start_cell_lateral_boundary,
            },
            vertical_sizes={
                "nflatlev": self._vertical_params.nflatlev,
                "nflat_gradp": self._metric_state_nonhydro.nflat_gradp,
                "model_top": gtx.int32(0),
                "surface_level": gtx.int32(self._grid.num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )

        self._compute_interpolation_and_nonhydro_buoy = setup_program(
            backend=backend,
            program=compute_cell_diagnostics_for_dycore.compute_interpolation_and_nonhydro_buoy,
            constant_args={
                "reference_theta_at_cells_on_model_levels": self._metric_state_nonhydro.reference_theta_at_cells_on_model_levels,
                "ddz_of_reference_exner_at_cells_on_half_levels": self._metric_state_nonhydro.ddz_of_reference_exner_at_cells_on_half_levels,
                "ddqz_z_half": self._metric_state_nonhydro.ddqz_z_half,
                "wgtfac_c": self._metric_state_nonhydro.wgtfac_c,
                "exner_w_explicit_weight_parameter": self._metric_state_nonhydro.exner_w_explicit_weight_parameter,
                "rhotheta_explicit_weight_parameter": self._params.rhotheta_explicit_weight_parameter,
                "rhotheta_implicit_weight_parameter": self._params.rhotheta_implicit_weight_parameter,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary_level_3,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._stencils_61_62 = setup_program(
            backend=backend,
            program=nhsolve_stencils.stencils_61_62,
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary,
                "horizontal_end": self._end_cell_lateral_boundary_level_4,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels + 1),
            },
        )
        self._en_smag_fac_for_zero_nshift = setup_program(
            backend=backend,
            program=smagorinsky.en_smag_fac_for_zero_nshift,
            constant_args={
                "vect_a": self._vertical_params.interface_physical_height,
                "hdiff_smag_fac": self._config.fourth_order_divdamp_factor,
                "hdiff_smag_fac2": self._config.fourth_order_divdamp_factor2,
                "hdiff_smag_fac3": self._config.fourth_order_divdamp_factor3,
                "hdiff_smag_fac4": self._config.fourth_order_divdamp_factor4,
                "hdiff_smag_z": self._config.fourth_order_divdamp_z,
                "hdiff_smag_z2": self._config.fourth_order_divdamp_z2,
                "hdiff_smag_z3": self._config.fourth_order_divdamp_z3,
                "hdiff_smag_z4": self._config.fourth_order_divdamp_z4,
            },
            offset_provider={},
        )
        self._init_test_fields = setup_program(
            backend=backend,
            program=nhsolve_stencils.init_test_fields,
            horizontal_sizes={
                "edges_start": self._start_edge_lateral_boundary,
                "edges_end": self._end_edge_local,
                "cells_start": self._start_cell_lateral_boundary,
                "cells_end": self._end_cell_end,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": self._grid.num_levels,
            },
        )

        self.velocity_advection = VelocityAdvection(
            grid=grid,
            metric_state=metric_state_nonhydro,
            interpolation_state=interpolation_state,
            vertical_params=vertical_params,
            edge_params=edge_geometry,
            owner_mask=owner_mask,
            backend=backend,
        )
        self._allocate_local_fields(model_backends.get_allocator(backend))

        self._en_smag_fac_for_zero_nshift(
            enh_smag_fac=self.interpolated_fourth_order_divdamp_factor,
        )

        self.p_test_run = False

        self._dtime_previous_substep: float = 0.0
        """
        Dynamic substep length of previous substep in order to track if rayleigh damping coefficients need to be
        recomputed or not. The substep length should only change in case of high CFL condition.
        """

    def _allocate_local_fields(self, allocator: gtx_typing.Allocator | None) -> None:
        self.temporal_extrapolation_of_perturbed_exner = data_alloc.zero_field(
            self._grid,
            dims.CellDim,
            dims.KDim,
            dtype=ta.vpfloat,
            extend={dims.KDim: 1},
            allocator=allocator,
        )
        """
        Declared as z_exner_ex_pr in ICON.
        """
        self.exner_at_cells_on_half_levels = data_alloc.zero_field(
            self._grid,
            dims.CellDim,
            dims.KDim,
            dtype=ta.vpfloat,
            extend={dims.KDim: 1},
            allocator=allocator,
        )
        """
        Declared as z_exner_ic in ICON.
        """
        self.ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
            data_alloc.zero_field(
                self._grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, allocator=allocator
            )
        )
        """
        Declared as z_dexner_dz_c_1 in ICON.
        """
        self.perturbed_theta_v_at_cells_on_half_levels = data_alloc.zero_field(
            self._grid,
            dims.CellDim,
            dims.KDim,
            dtype=ta.vpfloat,
            extend={dims.KDim: 1},
            allocator=allocator,
        )

        """
        Declared as z_theta_v_pr_ic in ICON.
        """
        self.nonhydro_buoy_at_cells_on_half_levels = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, allocator=allocator
        )
        """
        Declared as z_th_ddz_exner_c in ICON. theta' dpi0/dz + theta (1 - eta_impl) dpi'/dz.
        It represents the vertical pressure gradient and buoyancy acceleration.
        Note that it only has nlev because it is only used in computation of the explicit
        term for updating w, and w at model top/bottom is diagnosed.
        """
        self.perturbed_rho_at_cells_on_model_levels = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, allocator=allocator
        )
        """
        Declared as z_rth_pr_1 in ICON.
        """
        self.perturbed_theta_v_at_cells_on_model_levels = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, allocator=allocator
        )
        """
        Declared as z_rth_pr_2 in ICON.
        """
        self.d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
            data_alloc.zero_field(
                self._grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat, allocator=allocator
            )
        )
        """
        Declared as z_dexner_dz_c_2 in ICON.
        """
        self.z_vn_avg = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        self.theta_v_flux_at_edges_on_model_levels = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        """
        Declared as z_theta_v_fl_e in ICON.
        """
        self.z_rho_v = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        self.z_theta_v_v = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        self.k_field = data_alloc.index_field(
            self._grid, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        )
        self._contravariant_correction_at_edges_on_model_levels = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat, allocator=allocator
        )
        """
        Declared as z_w_concorr_me in ICON. vn dz/dn + vt dz/dt, z is topography height
        """
        self.hydrostatic_correction_on_lowest_level = gtx.constructors.zeros(
            domain={
                dims.EdgeDim: (0, self._grid.num_edges),
                dims.KDim: (self._grid.num_levels - 1, self._grid.num_levels),
            },
            allocator=allocator,
            dtype=ta.vpfloat,
        )
        # using GT4Py internal API to create a 1D field view from the (num_edges, 1)-sized field
        self.hydrostatic_correction_on_lowest_level_1d_view = gtx_common._field(
            self.hydrostatic_correction_on_lowest_level.ndarray[:, 0],
            domain={dims.EdgeDim: (0, self._grid.num_edges)},
        )
        """
        Declared as z_hydro_corr in ICON. Used for computation of horizontal pressure gradient over steep slope.
        """
        self.rayleigh_damping_factor = data_alloc.zero_field(
            self._grid, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        """
        Declared as z_raylfac in ICON.
        """
        self.interpolated_fourth_order_divdamp_factor = data_alloc.zero_field(
            self._grid, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        )
        """
        Declared as enh_divdamp_fac in ICON.
        """
        self.intermediate_fields = IntermediateFields.allocate(grid=self._grid, allocator=allocator)

    def _determine_local_domains(self) -> None:
        vertex_domain = h_grid.domain(dims.VertexDim)
        cell_domain = h_grid.domain(dims.CellDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        edge_halo_level_2 = edge_domain(h_grid.Zone.HALO_LEVEL_2)

        self._start_cell_lateral_boundary = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY)
        )
        self._start_cell_lateral_boundary_level_3 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
        )
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._start_cell_local = self._grid.start_index(cell_domain(h_grid.Zone.LOCAL))
        self._start_cell_halo = self._grid.start_index(cell_domain(h_grid.Zone.HALO))
        self._start_cell_halo_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.HALO_LEVEL_2)
        )

        self._end_cell_lateral_boundary_level_4 = self._grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
        )
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))
        self._end_cell_halo_level_2 = self._grid.end_index(cell_domain(h_grid.Zone.HALO_LEVEL_2))
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        self._start_edge_lateral_boundary = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY)
        )
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._start_edge_lateral_boundary_level_7 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
        )
        self._start_edge_nudging_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )

        self._start_edge_halo_level_2 = self._grid.start_index(edge_halo_level_2)

        self._end_edge_nudging = self._grid.end_index(edge_domain(h_grid.Zone.NUDGING))
        self._end_edge_local = self._grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._end_edge_halo_level_2 = self._grid.end_index(edge_halo_level_2)
        self._end_edge_end = self._grid.end_index(edge_domain(h_grid.Zone.END))

        self._start_vertex_lateral_boundary_level_2 = self._grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_vertex_halo = self._grid.end_index(vertex_domain(h_grid.Zone.HALO))

    def _get_rayleigh_damping_factor(self, dtime: float) -> fa.KField[float]:
        if dtime != self._dtime_previous_substep:
            #  Precompute Rayleigh damping factor if substep magnitude changes
            self._compute_rayleigh_damping_factor(
                rayleigh_damping_factor=self.rayleigh_damping_factor,
                dtime=dtime,
            )
            self._dtime_previous_substep = dtime
        return self.rayleigh_damping_factor

    def time_step(
        self,
        *,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        prep_adv: dycore_states.PrepAdvection,
        second_order_divdamp_factor: float,
        dtime: float,
        ndyn_substeps_var: int,
        at_initial_timestep: bool,
        lprep_adv: bool,
        at_first_substep: bool,
        at_last_substep: bool,
        is_iau_active: bool = False,
        iau_wgt_dyn: float = 0.0,
    ) -> None:
        """
        Update prognostic variables (prognostic_states.next) after the dynamical process over one substep.
        Args:
            diagnostic_state_nh: diagnostic variables used for solving the governing equations. It includes local variables and the physics tendency term that comes from physics
            prognostic_states: prognostic variables
            prep_adv: variables for tracer advection
            second_order_divdamp_factor: Originally declared as divdamp_fac_o2 in ICON. Second order (nabla2) divergence damping coefficient.
            dtime: time step
            ndyn_substeps_var: number of dynamical substeps
            at_initial_timestep: initial time step of the model run
            lprep_adv: Preparation for tracer advection
            at_first_substep: first substep
            at_last_substep: last substep
            is_iau_active: Incremental analysis update active during dycore step
            iau_wgt_dyn: weight scalar for the incremental analysis update
        """
        log.info(
            f"running timestep: dtime = {dtime}, initial_timestep = {at_initial_timestep}, first_substep = {at_first_substep}, last_substep = {at_last_substep}, prep_adv = {lprep_adv}"
        )

        if self.p_test_run:
            self._init_test_fields(
                self.intermediate_fields.rho_at_edges_on_model_levels,
                self.intermediate_fields.theta_v_at_edges_on_model_levels,
                self.intermediate_fields.dwdz_at_cells_on_model_levels,
                self.intermediate_fields.horizontal_gradient_of_normal_wind_divergence,
            )

        self.run_predictor_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_states=prognostic_states,
            z_fields=self.intermediate_fields,
            dtime=dtime,
            at_initial_timestep=at_initial_timestep,
            at_first_substep=at_first_substep,
            is_iau_active=is_iau_active,
            iau_wgt_dyn=iau_wgt_dyn,
        )

        self.run_corrector_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_states=prognostic_states,
            z_fields=self.intermediate_fields,
            prep_adv=prep_adv,
            second_order_divdamp_factor=second_order_divdamp_factor,
            dtime=dtime,
            ndyn_substeps_var=ndyn_substeps_var,
            lprep_adv=lprep_adv,
            at_first_substep=at_first_substep,
            at_last_substep=at_last_substep,
            is_iau_active=is_iau_active,
            iau_wgt_dyn=iau_wgt_dyn,
        )
        if self._grid.limited_area:
            self._compute_exner_from_rhotheta_in_lateral_boundary(
                rho=prognostic_states.next.rho,
                theta_v=prognostic_states.next.theta_v,
                exner=prognostic_states.next.exner,
            )
        self._update_theta_and_exner_in_halo(
            rho_now=prognostic_states.current.rho,
            rho_new=prognostic_states.next.rho,
            theta_v_now=prognostic_states.current.theta_v,
            theta_v_new=prognostic_states.next.theta_v,
            exner_now=prognostic_states.current.exner,
            exner_new=prognostic_states.next.exner,
        )

    def run_predictor_step(
        self,
        *,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        z_fields: IntermediateFields,
        dtime: float,
        at_initial_timestep: bool,
        at_first_substep: bool,
        is_iau_active: bool,
        iau_wgt_dyn: float,
    ) -> None:
        """
        Runs the predictor step of the non-hydrostatic solver.
        """

        log.info(
            f"running predictor step: dtime = {dtime}, initial_timestep = {at_initial_timestep} at_first_substep = {at_first_substep}"
        )

        if at_first_substep:
            # Recompute only vn tendency
            skip_compute_predictor_vertical_advection: bool = (
                self._config.itime_scheme == dycore_states.TimeSteppingScheme.MOST_EFFICIENT
                and not (at_initial_timestep and at_first_substep)
            )

            assert self._cell_params.area is not None

            self.velocity_advection.run_predictor_step(
                skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
                diagnostic_state=diagnostic_state_nh,
                prognostic_state=prognostic_states.current,
                contravariant_correction_at_edges_on_model_levels=self._contravariant_correction_at_edges_on_model_levels,
                horizontal_kinetic_energy_at_edges_on_model_levels=z_fields.horizontal_kinetic_energy_at_edges_on_model_levels,
                tangential_wind_on_half_levels=z_fields.tangential_wind_on_half_levels,
                dtime=dtime,
                cell_areas=self._cell_params.area,
            )

        self._compute_perturbed_quantities_and_interpolation(
            temporal_extrapolation_of_perturbed_exner=self.temporal_extrapolation_of_perturbed_exner,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=self.ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=self.d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            perturbed_exner_at_cells_on_model_levels=diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels,
            exner_at_cells_on_half_levels=self.exner_at_cells_on_half_levels,
            perturbed_rho_at_cells_on_model_levels=self.perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels=self.perturbed_theta_v_at_cells_on_model_levels,
            rho_at_cells_on_half_levels=diagnostic_state_nh.rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels=self.perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=diagnostic_state_nh.theta_v_at_cells_on_half_levels,
            current_rho=prognostic_states.current.rho,
            current_theta_v=prognostic_states.current.theta_v,
            nonhydro_buoy_at_cells_on_half_levels=self.nonhydro_buoy_at_cells_on_half_levels,
            current_exner=prognostic_states.current.exner,
        )

        log.debug("predictor: start stencil compute_rho_theta_pgrad_and_update_vn")
        self._compute_hydrostatic_correction_term(
            theta_v=prognostic_states.current.theta_v,
            theta_v_ic=diagnostic_state_nh.theta_v_at_cells_on_half_levels,
            z_hydro_corr=self.hydrostatic_correction_on_lowest_level,
        )

        self._compute_rho_theta_pgrad_and_update_vn(
            rho_at_edges_on_model_levels=z_fields.rho_at_edges_on_model_levels,
            theta_v_at_edges_on_model_levels=z_fields.theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient=z_fields.horizontal_pressure_gradient,
            next_vn=prognostic_states.next.vn,
            current_vn=prognostic_states.current.vn,
            tangential_wind=diagnostic_state_nh.tangential_wind,
            perturbed_rho_at_cells_on_model_levels=self.perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels=self.perturbed_theta_v_at_cells_on_model_levels,
            temporal_extrapolation_of_perturbed_exner=self.temporal_extrapolation_of_perturbed_exner,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=self.ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=self.d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            hydrostatic_correction_on_lowest_level=self.hydrostatic_correction_on_lowest_level_1d_view,
            predictor_normal_wind_advective_tendency=diagnostic_state_nh.normal_wind_advective_tendency.predictor,
            normal_wind_tendency_due_to_slow_physics_process=diagnostic_state_nh.normal_wind_tendency_due_to_slow_physics_process,
            normal_wind_iau_increment=diagnostic_state_nh.normal_wind_iau_increment,
            grf_tend_vn=diagnostic_state_nh.grf_tend_vn,
            is_iau_active=is_iau_active,
            iau_wgt_dyn=iau_wgt_dyn,
            dtime=dtime,
        )

        log.debug("exchanging prognostic field 'vn' and local field 'rho_at_edges_on_model_levels'")
        self._exchange.exchange(
            dims.EdgeDim,
            prognostic_states.next.vn,
            z_fields.rho_at_edges_on_model_levels,
            stream=decomposition.DEFAULT_STREAM,
        )

        self._compute_horizontal_velocity_quantities_and_fluxes(
            spatially_averaged_vn=self.z_vn_avg,
            horizontal_gradient_of_normal_wind_divergence=z_fields.horizontal_gradient_of_normal_wind_divergence,
            tangential_wind=diagnostic_state_nh.tangential_wind,
            mass_flux_at_edges_on_model_levels=diagnostic_state_nh.mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=self.theta_v_flux_at_edges_on_model_levels,
            tangential_wind_on_half_levels=z_fields.tangential_wind_on_half_levels,
            vn_on_half_levels=diagnostic_state_nh.vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels=z_fields.horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels=self._contravariant_correction_at_edges_on_model_levels,
            vn=prognostic_states.next.vn,
            rho_at_edges_on_model_levels=z_fields.rho_at_edges_on_model_levels,
            theta_v_at_edges_on_model_levels=z_fields.theta_v_at_edges_on_model_levels,
        )

        self._vertically_implicit_solver_at_predictor_step(
            contravariant_correction_at_cells_on_half_levels=diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels,
            next_w=prognostic_states.next.w,
            next_rho=prognostic_states.next.rho,
            next_exner=prognostic_states.next.exner,
            next_theta_v=prognostic_states.next.theta_v,
            dwdz_at_cells_on_model_levels=z_fields.dwdz_at_cells_on_model_levels,
            exner_dynamical_increment=diagnostic_state_nh.exner_dynamical_increment,
            mass_flux_at_edges_on_model_levels=diagnostic_state_nh.mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=self.theta_v_flux_at_edges_on_model_levels,
            predictor_vertical_wind_advective_tendency=diagnostic_state_nh.vertical_wind_advective_tendency.predictor,
            nonhydro_buoy_at_cells_on_half_levels=self.nonhydro_buoy_at_cells_on_half_levels,
            rho_at_cells_on_half_levels=diagnostic_state_nh.rho_at_cells_on_half_levels,
            contravariant_correction_at_edges_on_model_levels=self._contravariant_correction_at_edges_on_model_levels,
            current_exner=prognostic_states.current.exner,
            current_rho=prognostic_states.current.rho,
            current_theta_v=prognostic_states.current.theta_v,
            current_w=prognostic_states.current.w,
            theta_v_at_cells_on_half_levels=diagnostic_state_nh.theta_v_at_cells_on_half_levels,
            perturbed_exner_at_cells_on_model_levels=diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels,
            exner_tendency_due_to_slow_physics=diagnostic_state_nh.exner_tendency_due_to_slow_physics,
            rho_iau_increment=diagnostic_state_nh.rho_iau_increment,
            exner_iau_increment=diagnostic_state_nh.exner_iau_increment,
            rayleigh_damping_factor=self._get_rayleigh_damping_factor(dtime),
            dtime=dtime,
            at_first_substep=at_first_substep,
            is_iau_active=is_iau_active,
            iau_wgt_dyn=iau_wgt_dyn,
        )

        if self._grid.limited_area:
            self._stencils_61_62(
                rho_now=prognostic_states.current.rho,
                grf_tend_rho=diagnostic_state_nh.grf_tend_rho,
                theta_v_now=prognostic_states.current.theta_v,
                grf_tend_thv=diagnostic_state_nh.grf_tend_thv,
                w_now=prognostic_states.current.w,
                grf_tend_w=diagnostic_state_nh.grf_tend_w,
                rho_new=prognostic_states.next.rho,
                exner_new=prognostic_states.next.exner,
                w_new=prognostic_states.next.w,
                dtime=dtime,
            )

        if self._grid.limited_area and self._config.divdamp_type >= 3:
            self._compute_dwdz_for_divergence_damping(
                w=prognostic_states.next.w,
                w_concorr_c=diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels,
                z_dwdz_dd=z_fields.dwdz_at_cells_on_model_levels,
            )

        if self._config.divdamp_type >= 3:
            log.debug(
                "exchanging prognostic field 'w' and local field 'dwdz_at_cells_on_model_levels'"
            )
            self._exchange.exchange(
                dims.CellDim,
                prognostic_states.next.w,
                z_fields.dwdz_at_cells_on_model_levels,
                stream=decomposition.DEFAULT_STREAM,
            )
        else:
            log.debug("exchanging prognostic field 'w'")
            self._exchange.exchange(
                dims.CellDim,
                prognostic_states.next.w,
                stream=decomposition.DEFAULT_STREAM,
            )

    def run_corrector_step(
        self,
        *,
        diagnostic_state_nh: dycore_states.DiagnosticStateNonHydro,
        prognostic_states: common_utils.TimeStepPair[prognostics.PrognosticState],
        z_fields: IntermediateFields,
        second_order_divdamp_factor: float,
        prep_adv: dycore_states.PrepAdvection,
        dtime: float,
        ndyn_substeps_var: int,
        lprep_adv: bool,
        at_first_substep: bool,
        at_last_substep: bool,
        is_iau_active: bool,
        iau_wgt_dyn: float,
    ) -> None:
        log.info(
            f"running corrector step: dtime = {dtime}, prep_adv = {lprep_adv},  "
            f"second_order_divdamp_factor = {second_order_divdamp_factor}, at_first_substep = {at_first_substep}, at_last_substep = {at_last_substep}  "
        )

        # Inverse value of ndyn_substeps for tracer advection precomputations
        r_nsubsteps = 1.0 / ndyn_substeps_var

        # scaling factor for second-order divergence damping: second_order_divdamp_factor_from_sfc_to_divdamp_z*delta_x**2
        # delta_x**2 is approximated by the mean cell area
        # Coefficient for reduced fourth-order divergence d
        assert self._cell_params.area is not None
        assert self._cell_params.mean_cell_area is not None
        second_order_divdamp_scaling_coeff = (
            second_order_divdamp_factor * self._cell_params.mean_cell_area
        )

        log.debug("corrector run velocity advection")
        self.velocity_advection.run_corrector_step(
            diagnostic_state=diagnostic_state_nh,
            prognostic_state=prognostic_states.next,
            horizontal_kinetic_energy_at_edges_on_model_levels=z_fields.horizontal_kinetic_energy_at_edges_on_model_levels,
            tangential_wind_on_half_levels=z_fields.tangential_wind_on_half_levels,
            dtime=dtime,
            cell_areas=self._cell_params.area,
        )

        self._compute_interpolation_and_nonhydro_buoy(
            rho_at_cells_on_half_levels=diagnostic_state_nh.rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels=self.perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels=diagnostic_state_nh.theta_v_at_cells_on_half_levels,
            nonhydro_buoy_at_cells_on_half_levels=self.nonhydro_buoy_at_cells_on_half_levels,
            w=prognostic_states.next.w,
            contravariant_correction_at_cells_on_half_levels=diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels,
            current_rho=prognostic_states.current.rho,
            next_rho=prognostic_states.next.rho,
            current_theta_v=prognostic_states.current.theta_v,
            next_theta_v=prognostic_states.next.theta_v,
            perturbed_exner_at_cells_on_model_levels=diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels,
            dtime=dtime,
        )

        log.debug("corrector: start stencil apply_divergence_damping_and_update_vn")
        apply_2nd_order_divergence_damping = (
            self._config.divdamp_order == dycore_states.DivergenceDampingOrder.SECOND_ORDER
            or (
                self._config.divdamp_order == dycore_states.DivergenceDampingOrder.COMBINED
                and second_order_divdamp_scaling_coeff > 1.0e-6
            )
        )
        apply_4th_order_divergence_damping = (
            self._config.divdamp_order == dycore_states.DivergenceDampingOrder.FOURTH_ORDER
            or (
                self._config.divdamp_order == dycore_states.DivergenceDampingOrder.COMBINED
                and second_order_divdamp_factor <= (4.0 * self._config.fourth_order_divdamp_factor)
            )
        )

        self._apply_divergence_damping_and_update_vn(
            horizontal_gradient_of_normal_wind_divergence=z_fields.horizontal_gradient_of_normal_wind_divergence,
            next_vn=prognostic_states.next.vn,
            current_vn=prognostic_states.current.vn,
            dwdz_at_cells_on_model_levels=z_fields.dwdz_at_cells_on_model_levels,
            predictor_normal_wind_advective_tendency=diagnostic_state_nh.normal_wind_advective_tendency.predictor,
            corrector_normal_wind_advective_tendency=diagnostic_state_nh.normal_wind_advective_tendency.corrector,
            normal_wind_tendency_due_to_slow_physics_process=diagnostic_state_nh.normal_wind_tendency_due_to_slow_physics_process,
            normal_wind_iau_increment=diagnostic_state_nh.normal_wind_iau_increment,
            theta_v_at_edges_on_model_levels=z_fields.theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient=z_fields.horizontal_pressure_gradient,
            interpolated_fourth_order_divdamp_factor=self.interpolated_fourth_order_divdamp_factor,
            second_order_divdamp_factor=second_order_divdamp_factor,
            second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
            dtime=dtime,
            apply_2nd_order_divergence_damping=apply_2nd_order_divergence_damping,
            apply_4th_order_divergence_damping=apply_4th_order_divergence_damping,
            is_iau_active=is_iau_active,
            iau_wgt_dyn=iau_wgt_dyn,
        )

        log.debug("exchanging prognostic field 'vn'")
        self._exchange.exchange(
            dims.EdgeDim,
            prognostic_states.next.vn,
            stream=decomposition.DEFAULT_STREAM,
        )

        self._compute_averaged_vn_and_fluxes(
            spatially_averaged_vn=self.z_vn_avg,
            mass_flux_at_edges_on_model_levels=diagnostic_state_nh.mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=self.theta_v_flux_at_edges_on_model_levels,
            substep_and_spatially_averaged_vn=prep_adv.vn_traj,
            substep_averaged_mass_flux=prep_adv.mass_flx_me,
            vn=prognostic_states.next.vn,
            rho_at_edges_on_model_levels=z_fields.rho_at_edges_on_model_levels,
            theta_v_at_edges_on_model_levels=z_fields.theta_v_at_edges_on_model_levels,
            prepare_advection=lprep_adv,
            at_first_substep=at_first_substep,
            r_nsubsteps=r_nsubsteps,
        )

        self._vertically_implicit_solver_at_corrector_step(
            next_w=prognostic_states.next.w,
            next_rho=prognostic_states.next.rho,
            next_exner=prognostic_states.next.exner,
            next_theta_v=prognostic_states.next.theta_v,
            dynamical_vertical_mass_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels=prep_adv.dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            exner_dynamical_increment=diagnostic_state_nh.exner_dynamical_increment,
            geofac_div=self._interpolation_state.geofac_div,
            mass_flux_at_edges_on_model_levels=diagnostic_state_nh.mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels=self.theta_v_flux_at_edges_on_model_levels,
            predictor_vertical_wind_advective_tendency=diagnostic_state_nh.vertical_wind_advective_tendency.predictor,
            corrector_vertical_wind_advective_tendency=diagnostic_state_nh.vertical_wind_advective_tendency.corrector,
            nonhydro_buoy_at_cells_on_half_levels=self.nonhydro_buoy_at_cells_on_half_levels,
            rho_at_cells_on_half_levels=diagnostic_state_nh.rho_at_cells_on_half_levels,
            contravariant_correction_at_cells_on_half_levels=diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels,
            current_exner=prognostic_states.current.exner,
            current_rho=prognostic_states.current.rho,
            current_theta_v=prognostic_states.current.theta_v,
            current_w=prognostic_states.current.w,
            theta_v_at_cells_on_half_levels=diagnostic_state_nh.theta_v_at_cells_on_half_levels,
            perturbed_exner_at_cells_on_model_levels=diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels,
            exner_tendency_due_to_slow_physics=diagnostic_state_nh.exner_tendency_due_to_slow_physics,
            rho_iau_increment=diagnostic_state_nh.rho_iau_increment,
            exner_iau_increment=diagnostic_state_nh.exner_iau_increment,
            is_iau_active=is_iau_active,
            iau_wgt_dyn=iau_wgt_dyn,
            rayleigh_damping_factor=self._get_rayleigh_damping_factor(dtime),
            lprep_adv=lprep_adv,
            r_nsubsteps=r_nsubsteps,
            ndyn_substeps_var=float(ndyn_substeps_var),
            dtime=dtime,
            at_first_substep=at_first_substep,
            at_last_substep=at_last_substep,
        )

        # prepare flux field for tracer advection on lateral boundary, if exists
        if self._grid.limited_area:
            if lprep_adv:
                if at_first_substep:
                    log.debug(
                        "corrector step sets prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels to zero"
                    )
                    self._init_cell_kdim_field_with_zero_wp(
                        field_with_zero_wp=prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels,
                    )
                self._update_mass_flux_weighted(
                    rho_ic=diagnostic_state_nh.rho_at_cells_on_half_levels,
                    w_now=prognostic_states.current.w,
                    w_new=prognostic_states.next.w,
                    w_concorr_c=diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels,
                    mass_flx_ic=prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels,
                    r_nsubsteps=r_nsubsteps,
                )

        log.debug("exchange prognostic fields 'rho' , 'exner', 'w'")
        self._exchange.exchange(
            dims.CellDim,
            prognostic_states.next.rho,
            prognostic_states.next.exner,
            prognostic_states.next.w,
            stream=decomposition.DEFAULT_STREAM,
        )
