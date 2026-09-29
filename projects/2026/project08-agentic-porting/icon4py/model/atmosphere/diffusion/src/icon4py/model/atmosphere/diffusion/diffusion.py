# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import functools
import logging
import math
import sys
import typing
from typing import Any, Final

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

import icon4py.model.common.grid.states as grid_states
import icon4py.model.common.states.prognostic_state as prognostics
from icon4py.model.atmosphere.diffusion import diffusion_states, diffusion_utils
from icon4py.model.atmosphere.diffusion.diffusion_utils import (
    copy_field,
    init_diffusion_local_fields_for_regular_timestep,
    scale_k,
    setup_fields_for_initial_step,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_theta_and_exner import (
    apply_diffusion_to_theta_and_exner,
)
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_vn import apply_diffusion_to_vn
from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence import (
    apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostic_quantities_for_turbulence import (
    calculate_diagnostic_quantities_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools import (
    calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_and_smag_coefficients_for_vn import (
    calculate_nabla2_and_smag_coefficients_for_vn,
)
from icon4py.model.common import constants, dimension as dims, model_backends
from icon4py.model.common.config import options as common_conf_opt
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid, vertical as v_grid
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.model_options import setup_program
from icon4py.model.common.utils import data_allocation as data_alloc


"""
Diffusion module ported from ICON mo_nh_diffusion.f90.

Supports only diffusion_type (=hdiff_order) 5 from the diffusion namelist.
"""

log = logging.getLogger(__name__)


class DiffusionType(int, enum.Enum):
    """
    Order of nabla operator for diffusion.

    Note: Called `hdiff_order` in `mo_diffusion_nml.f90`.
    Note: We currently only support type 5.
    """

    NO_DIFFUSION = -1  #: no diffusion
    LINEAR_2ND_ORDER = 2  #: 2nd order linear diffusion on all vertical levels
    SMAGORINSKY_NO_BACKGROUND = 3  #: Smagorinsky diffusion without background diffusion
    LINEAR_4TH_ORDER = 4  #: 4th order linear diffusion on all vertical levels
    SMAGORINSKY_4TH_ORDER = 5  #: Smagorinsky diffusion with fourth-order background diffusion


class SmagorinskyStencilType(int, enum.Enum):
    """
    Type of the reconstruction stencil for the Smagorinsky diffusion of normal wind (vn).

    Note: Called `itype_vn_diffu` in `mo_diffusion_nml.f90`.
    Note: We currently only support type 1 in combination with lsmag_3d=False.
    """

    DIAMOND_VERTICES = (
        1  #: Smagorinsky diffusion of vn with diamond stencil on vertices (only for vn)
    )
    CELLS_AND_VERTICES = 2  #: Smagorinsky diffusion of vn with stencil on neighboring vertices (E2V) and cell centers (E2C)


class TemperatureDiscretizationType(int, enum.Enum):
    """
    Type of the discretization of the Smagorinsky diffusion of temperature.

    Note: Called `itype_t_diffu` in `mo_diffusion_nml.f90`.
    Note: We currently only support type 2.
    """

    HOMOGENEOUS = 1  #: K Lap(T)
    HETEROGENEOUS = 2  #: Div (K Grad(T))


class TurbulenceShearForcingType(int, enum.Enum):
    """
    Type of shear forcing used in turbulence.

    Note: called `itype_sher` in `mo_turbdiff_nml.f90`
    """

    VERTICAL_OF_HORIZONTAL_WIND = 0  #: only vertical shear of horizontal wind
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND = (
        1  #: as `VERTICAL_ONLY` plus horizontal shear correction
    )
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND = (
        2  #: as `VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND` plus shear form vertical velocity
    )
    VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND_LTHESH = 3  #: same as `VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND` but scaling of coarse-grid horizontal shear production term with 1/sqrt(Ri) (if LTKESH = TRUE)


class ForcingType(int, enum.Enum):
    """
    Type of physics forcing applied to the model.

    Note: called `iforcing` in `mo_run_nml.f90`
    """

    NO_FORCING = 0  #: no physics forcing (diagnostic / idealized runs)
    AES = 2  #: Atmospheric Earth System / ECHAM forcing (iaes)
    NWP = 3  #: Numerical Weather Prediction forcing (inwp)


@dataclasses.dataclass(kw_only=True)
class DiffusionConfig:
    """
    Contains necessary parameter to configure a diffusion run.

    Encapsulates namelist parameters and derived parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    # TODO(halungge): to be read from config
    # TODO(halungge):  handle dependencies on other namelists (see below...)

    diffusion_type: typing.Annotated[
        DiffusionType,
        common_conf_opt.ConfigOption(
            description="Order of Nabla operator for diffusion.",
            icon_equivalent=common_conf_opt.IconOption("hdiff_order", ("diffusion_nml",)),
        ),
    ] = DiffusionType.SMAGORINSKY_4TH_ORDER

    apply_to_vertical_wind: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, apply diffusion to the vertical wind field.",
            icon_equivalent=common_conf_opt.IconOption("lhdiff_w", ("diffusion_nml",)),
        ),
    ] = True

    apply_to_horizontal_wind: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If true, apply diffusion on the horizontal wind field.",
            icon_equivalent=common_conf_opt.IconOption("lhdiff_vn", ("diffusion_nml",)),
        ),
    ] = True

    apply_to_temperature: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, apply horizontal diffusion to temperature field.",
            icon_equivalent=common_conf_opt.IconOption("lhdiff_temp", ("diffusion_nml",)),
        ),
    ] = True

    apply_smag_diff_to_vertical_wind: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, apply Smagorinsky diffusion to vertical wind field.",
            icon_equivalent=common_conf_opt.IconOption(
                "lhdiff_smag_w", ("diffusion_nml",), list_to_value=True
            ),
        ),
    ] = False

    compute_3d_smag_coeff: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, compute 3D Smagorinsky diffusion coefficient.",
            icon_equivalent=common_conf_opt.IconOption(
                "lsmag_3d", ("diffusion_nml",), list_to_value=True
            ),
        ),
    ] = False

    type_vn_diffu: typing.Annotated[
        SmagorinskyStencilType,
        common_conf_opt.ConfigOption(
            description="Reconstruction method used for Smagorinsky diffusion.",
            icon_equivalent=common_conf_opt.IconOption("itype_vn_diffu", ("diffusion_nml",)),
        ),
    ] = SmagorinskyStencilType.DIAMOND_VERTICES

    type_t_diffu: typing.Annotated[
        TemperatureDiscretizationType,
        common_conf_opt.ConfigOption(
            description="Options for discretizing the Smagorinsky temperature diffusion.",
            icon_equivalent=common_conf_opt.IconOption("itype_t_diffu", ("diffusion_nml",)),
        ),
    ] = TemperatureDiscretizationType.HETEROGENEOUS

    hdiff_efdt_ratio: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Ratio of e-folding time to (2*)time step.",
            icon_equivalent=common_conf_opt.IconOption("hdiff_efdt_ratio", ("diffusion_nml",)),
        ),
    ] = 36.0

    hdiff_w_efdt_ratio: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Ratio of e-folding time to time step for w diffusion (NH only).",
            icon_equivalent=common_conf_opt.IconOption("hdiff_w_efdt_ratio", ("diffusion_nml",)),
        ),
    ] = 15.0

    # TODO(muellch): The four smagorinsky factors and heights should be in one or two dataclasses.
    smagorinski_scaling_factor: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Smagorinsky factor for z <= smagorinski_scaling_height (constant base value).",
            icon_equivalent=common_conf_opt.IconOption("hdiff_smag_fac", ("diffusion_nml",)),
        ),
    ] = 0.015

    smagorinski_scaling_factor2: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description=(
                "Smagorinsky factor at z = smagorinski_scaling_height2: end of the linear segment and"
                "start of the quadratic segment. The linear slope is (factor2-factor1)/(height2-height1)."
            ),
            icon_equivalent=common_conf_opt.IconOption("hdiff_smag_fac2", ("diffusion_nml",)),
        ),
    ] = 2e-6 * (1600.0 + 25000.0 + math.sqrt(1600.0 * (1600 + 50000.0)))

    smagorinski_scaling_factor3: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description=(
                "Smagorinsky factor at z = smagorinski_scaling_height3: interior control point of the"
                "quadratic segment (height2 <= height3 <= height4), used to fit the quadratic coefficients."
            ),
            icon_equivalent=common_conf_opt.IconOption("hdiff_smag_fac3", ("diffusion_nml",)),
        ),
    ] = 0.0

    smagorinski_scaling_factor4: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description=(
                "Smagorinsky factor for z >= smagorinski_scaling_height4 (constant asymptotic value)."
                "Also the third control point that defines the quadratic segment together with factor2 and factor3."
            ),
            icon_equivalent=common_conf_opt.IconOption("hdiff_smag_fac4", ("diffusion_nml",)),
        ),
    ] = 1.0

    smagorinski_scaling_height: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description=(
                "Lower boundary of the linear segment: factor is constant at smagorinski_scaling_factor "
                "below this height."
            ),
            icon_equivalent=common_conf_opt.IconOption("hdiff_smag_z", ("diffusion_nml",)),
        ),
    ] = 32500.0

    smagorinski_scaling_height2: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Transition height between linear and quadratic segments.",
            icon_equivalent=common_conf_opt.IconOption("hdiff_smag_z2", ("diffusion_nml",)),
        ),
    ] = 1600.0 + 50000.0 + math.sqrt(1600.0 * (1600 + 50000.0))

    smagorinski_scaling_height3: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Interior control point height within the quadratic segment (height2 <= height3 <= height4).",
            icon_equivalent=common_conf_opt.IconOption("hdiff_smag_z3", ("diffusion_nml",)),
        ),
    ] = 50000.0

    smagorinski_scaling_height4: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description=(
                "Upper boundary of the quadratic segment: factor is constant at "
                "smagorinski_scaling_factor4 above this height."
            ),
            icon_equivalent=common_conf_opt.IconOption("hdiff_smag_z4", ("diffusion_nml",)),
        ),
    ] = 90000.0

    apply_zdiffusion_t: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description="If True, apply truly horizontal temperature diffusion over steep slopes.",
            icon_equivalent=common_conf_opt.IconOption("l_zdiffu_t", ("nonhydrostatic_nml",)),
        ),
    ] = True

    ndyn_substeps: typing.Annotated[
        int,
        common_conf_opt.ConfigOption(
            description="Number of dynamics substeps per fast-physics step.",
            icon_equivalent=common_conf_opt.IconOption("ndyn_substeps", ("nonhydrostatic_nml",)),
        ),
    ] = 5

    temperature_boundary_diffusion_denominator: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Denominator for temperature boundary diffusion.",
            icon_equivalent=common_conf_opt.IconOption("denom_diffu_t", ("gridref_nml",)),
        ),
    ] = 135.0

    velocity_boundary_diffusion_denominator: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Denominator for velocity boundary diffusion.",
            icon_equivalent=common_conf_opt.IconOption("denom_diffu_v", ("gridref_nml",)),
        ),
    ] = 200.0

    max_nudging_coefficient: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Maximum relaxation coefficient for lateral boundary nudging",
            icon_equivalent=common_conf_opt.IconOption(
                name="nudge_max_coeff", path=("interpol_nml",), read_from_icon=False
            ),
        ),
    ] = constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO * 0.02

    shear_type: typing.Annotated[
        TurbulenceShearForcingType,
        common_conf_opt.ConfigOption(
            description="Type of shear forcing used in turbulence.",
            icon_equivalent=common_conf_opt.IconOption("itype_sher", ("turbdiff_nml",)),
        ),
    ] = TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND

    iforcing: typing.Annotated[
        ForcingType,
        common_conf_opt.ConfigOption(
            description="Type of physics forcing.",
            icon_equivalent=common_conf_opt.IconOption("iforcing", ("run_nml",)),
        ),
    ] = ForcingType.NO_FORCING

    a_hshr: typing.Annotated[
        float,
        common_conf_opt.ConfigOption(
            description="Scaling factor for horizontal shear production term.",
            icon_equivalent=common_conf_opt.IconOption("a_hshr", ("turbdiff_nml",)),
        ),
    ] = 1.0

    loutshs: typing.Annotated[
        bool,
        common_conf_opt.ConfigOption(
            description=(
                "Output flag for horizontal shear."
                "Called 'loutshs' in mo_turbdiff_nml.f90, "
                "not a namelist parameter: its default is FALSE and "
                "only set to true in fortran `IF (.NOT. ldynamics)`."
            )
        ),
    ] = False

    def __post_init__(self) -> None:

        self._validate()

    @classmethod
    def from_fortran_dict(cls, atmo_dict: dict[str, Any], **overrides: Any) -> DiffusionConfig:
        return common_conf_opt.construct_config_from_icon(cls, atmo_dict, **overrides)

    def _validate(self) -> None:
        """Apply consistency checks and validation on configuration parameters."""
        if self.diffusion_type != DiffusionType.SMAGORINSKY_4TH_ORDER:
            raise NotImplementedError(
                "Only diffusion type 5 = `Smagorinsky diffusion with fourth-order background "
                "diffusion` is implemented"
            )

        if self.type_vn_diffu != SmagorinskyStencilType.DIAMOND_VERTICES:
            raise NotImplementedError(
                "Only type_vn_diffu 1 = `Smagorinsky diffusion with diamond stencil on vertices` is implemented"
            )

        if self.type_t_diffu != TemperatureDiscretizationType.HETEROGENEOUS:
            raise NotImplementedError(
                "Only type_t_diffu 2 = `Smagorinsky diffusion with heterogeneous discretization` is implemented"
            )

        if self.apply_smag_diff_to_vertical_wind:
            raise NotImplementedError("Smagorinsky diffusion for vertical wind is not implemented")

        if self.compute_3d_smag_coeff:
            raise NotImplementedError("3D Smagorinsky diffusion computation is not implemented")

        if self.shear_type not in (
            TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND,
            TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND,
            TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND,
        ):
            raise NotImplementedError(
                f"Turbulence Shear only {TurbulenceShearForcingType.VERTICAL_OF_HORIZONTAL_WIND} "
                f"and {TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND} "
                f"and {TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_VERTICAL_WIND} "
                f"implemented"
            )

    @functools.cached_property
    def substep_as_float(self) -> float:
        return float(self.ndyn_substeps)


@dataclasses.dataclass(frozen=True)
class DiffusionParams:
    """Calculates derived quantities depending on the diffusion config."""

    config: dataclasses.InitVar[DiffusionConfig]
    K2: Final[float] = dataclasses.field(init=False)
    K4: Final[float] = dataclasses.field(init=False)
    K6: Final[float] = dataclasses.field(init=False)
    K4W: Final[float] = dataclasses.field(init=False)
    smagorinski_factor: Final[tuple[float, float, float, float]] = dataclasses.field(init=False)
    smagorinski_height: Final[tuple[float, float, float, float]] = dataclasses.field(init=False)

    def __post_init__(self, config: DiffusionConfig) -> None:
        object.__setattr__(
            self,
            "K2",
            (1.0 / (config.hdiff_efdt_ratio * 8.0) if config.hdiff_efdt_ratio > 0.0 else 0.0),
        )
        object.__setattr__(self, "K4", self.K2 / 8.0)
        object.__setattr__(self, "K6", self.K2 / 64.0)
        object.__setattr__(
            self,
            "K4W",
            (1.0 / (config.hdiff_w_efdt_ratio * 36.0) if config.hdiff_w_efdt_ratio > 0 else 0.0),
        )

        object.__setattr__(
            self,
            "smagorinski_factor",
            (
                config.smagorinski_scaling_factor,
                config.smagorinski_scaling_factor2,
                config.smagorinski_scaling_factor3,
                config.smagorinski_scaling_factor4,
            ),
        )
        object.__setattr__(
            self,
            "smagorinski_height",
            (
                config.smagorinski_scaling_height,
                config.smagorinski_scaling_height2,
                config.smagorinski_scaling_height3,
                config.smagorinski_scaling_height4,
            ),
        )


class Diffusion:
    """Class that configures diffusion and does one diffusion step."""

    def __init__(
        self,
        *,
        grid: icon_grid.IconGrid,
        config: DiffusionConfig,
        params: DiffusionParams,
        vertical_grid: v_grid.VerticalGrid,
        metric_state: diffusion_states.DiffusionMetricState,
        interpolation_state: diffusion_states.DiffusionInterpolationState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        backend: gtx_typing.Backend
        | model_backends.DeviceType
        | model_backends.BackendDescriptor
        | None,
        exchange: decomposition.ExchangeRuntime,
    ) -> None:
        self._allocator = model_backends.get_allocator(backend)
        self._exchange = exchange
        self.config = config
        self._params = params
        self._grid = grid
        self._vertical_grid = vertical_grid
        self._metric_state = metric_state
        self._interpolation_state = interpolation_state
        self._edge_params = edge_params
        self._cell_params = cell_params

        assert self._cell_params.area is not None

        self.halo_exchange_wait = decomposition.create_halo_exchange_wait(
            self._exchange,
        )  # wait on a communication handle
        self.rd_o_cvd: float = constants.GAS_CONSTANT_DRY_AIR / (
            constants.CPD - constants.GAS_CONSTANT_DRY_AIR
        )
        #: threshold temperature deviation from neighboring grid points that activates extra diffusion against runaway cooling
        self.thresh_tdiff: float = -5.0
        self._horizontal_start_index_w_diffusion: gtx.int32 = gtx.int32(0)

        self.nudgezone_diff: float = 0.04 / (
            config.max_nudging_coefficient + sys.float_info.epsilon
        )
        self.bdy_diff: float = 0.015 / (config.max_nudging_coefficient + sys.float_info.epsilon)
        self.fac_bdydiff_v: float = (
            math.sqrt(config.substep_as_float) / config.velocity_boundary_diffusion_denominator
        )

        self.smag_offset: float = 0.25 * params.K4 * config.substep_as_float
        self.diff_multfac_w: float = min(1.0 / 48.0, params.K4W * config.substep_as_float)
        self._determine_horizontal_domains()

        self.mo_intp_rbf_rbf_vec_interpol_vertex = setup_program(
            backend=backend,
            program=mo_intp_rbf_rbf_vec_interpol_vertex,
            constant_args={
                "ptr_coeff_1": self._interpolation_state.rbf_coeff_1,
                "ptr_coeff_2": self._interpolation_state.rbf_coeff_2,
            },
            horizontal_sizes={
                "horizontal_start": self._vertex_start_lateral_boundary_level_2,
                "horizontal_end": self._vertex_end_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        self.calculate_nabla2_and_smag_coefficients_for_vn = setup_program(
            backend=backend,
            program=calculate_nabla2_and_smag_coefficients_for_vn,
            constant_args={
                "tangent_orientation": self._edge_params.tangent_orientation,
                "inv_primal_edge_length": self._edge_params.inverse_primal_edge_lengths,
                "inv_vert_vert_length": self._edge_params.inverse_vertex_vertex_lengths,
                "primal_normal_vert_x": self._edge_params.primal_normal_vert[0],
                "primal_normal_vert_y": self._edge_params.primal_normal_vert[1],
                "dual_normal_vert_x": self._edge_params.dual_normal_vert[0],
                "dual_normal_vert_y": self._edge_params.dual_normal_vert[1],
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_lateral_boundary_level_5,
                "horizontal_end": self._edge_end_halo_level_2,
            },
            vertical_sizes={"vertical_start": 0, "vertical_end": self._grid.num_levels},
            offset_provider=self._grid.connectivities,
        )

        self.calculate_diagnostic_quantities_for_turbulence = setup_program(
            backend=backend,
            program=calculate_diagnostic_quantities_for_turbulence,
            constant_args={
                "e_bln_c_s": self._interpolation_state.e_bln_c_s,
                "geofac_div": self._interpolation_state.geofac_div,
                "wgtfac_c": self._metric_state.wgtfac_c,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={"vertical_start": 1, "vertical_end": self._grid.num_levels},
            offset_provider=self._grid.connectivities,
        )
        self.apply_diffusion_to_vn = setup_program(
            backend=backend,
            program=apply_diffusion_to_vn,
            constant_args={
                "primal_normal_vert_v1": self._edge_params.primal_normal_vert[0],
                "primal_normal_vert_v2": self._edge_params.primal_normal_vert[1],
                "inv_vert_vert_length": self._edge_params.inverse_vertex_vertex_lengths,
                "inv_primal_edge_length": self._edge_params.inverse_primal_edge_lengths,
                "area_edge": self._edge_params.edge_areas,
                "nudgecoeff_e": self._interpolation_state.nudgecoeff_e,
                "nudgezone_diff": self.nudgezone_diff,
                "fac_bdydiff_v": self.fac_bdydiff_v,
                "limited_area": self._grid.limited_area,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_lateral_boundary_level_5,
                "horizontal_end": self._edge_end_local,
                "start_2nd_nudge_line_idx_e": self._edge_start_nudging_level_2,
            },
            vertical_sizes={"vertical_start": 0, "vertical_end": self._grid.num_levels},
            offset_provider=self._grid.connectivities,
        )
        self.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence = setup_program(
            backend=backend,
            program=apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence,
            constant_args={
                "geofac_n2s": self._interpolation_state.geofac_n2s,
                "geofac_grg_x": self._interpolation_state.geofac_grg_x,
                "geofac_grg_y": self._interpolation_state.geofac_grg_y,
                "area": self._cell_params.area,
                "diff_multfac_w": self.diff_multfac_w,
                "type_shear": self.config.shear_type,
            },
            horizontal_sizes={
                "horizontal_start": self._horizontal_start_index_w_diffusion,
                "horizontal_end": self._cell_end_halo,
                "halo_idx": self._cell_end_local,
                "interior_idx": self._cell_start_interior,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
                "nrdmax": gtx.int32(
                    self._vertical_grid.end_index_of_damping_layer + 1
                ),  # +1 since Fortran includes boundaries
            },
            offset_provider=self._grid.connectivities,
        )
        self.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools = setup_program(
            backend=backend,
            program=calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools,
            constant_args={
                "theta_ref_mc": self._metric_state.theta_ref_mc,
                "thresh_tdiff": self.thresh_tdiff,
                "smallest_vpfloat": constants.DBL_EPS,
            },
            horizontal_sizes={
                "horizontal_start": self._edge_start_nudging,
                "horizontal_end": self._edge_end_halo,
            },
            vertical_sizes={
                "vertical_start": self._grid.num_levels - 2,
                "vertical_end": self._grid.num_levels,
            },
            offset_provider=self._grid.connectivities,
        )
        self.apply_diffusion_to_theta_and_exner = setup_program(
            backend=backend,
            program=apply_diffusion_to_theta_and_exner,
            constant_args={
                "geofac_div": self._interpolation_state.geofac_div,
                "zd_vertoffset": self._metric_state.zd_vertoffset,
                "zd_diffcoef": self._metric_state.zd_diffcoef,
                "vcoef": self._metric_state.zd_intcoef,
                "geofac_n2s_c": self._interpolation_state.geofac_n2s_c,
                "geofac_n2s_nbh": self._interpolation_state.geofac_n2s_nbh,
                "inv_dual_edge_length": self._edge_params.inverse_dual_edge_lengths,
                "area": self._cell_params.area,
                "apply_zdiffusion_t": self.config.apply_zdiffusion_t,
                "rd_o_cvd": self.rd_o_cvd,
            },
            horizontal_sizes={
                "horizontal_start": self._cell_start_nudging,
                "horizontal_end": self._cell_end_local,
            },
            vertical_sizes={
                "vertical_start": 0,
                "vertical_end": self._grid.num_levels,
            },
            offset_provider=self._grid.connectivities,
        )
        self.copy_field = setup_program(backend=backend, program=copy_field)
        self.scale_k = setup_program(backend=backend, program=scale_k)
        self.setup_fields_for_initial_step = setup_program(
            backend=backend, program=setup_fields_for_initial_step
        )

        self.init_diffusion_local_fields_for_regular_timestep = setup_program(
            backend=backend,
            program=init_diffusion_local_fields_for_regular_timestep,
            offset_provider={},
        )

        self._allocate_local_fields(model_backends.get_allocator(backend))

        self.init_diffusion_local_fields_for_regular_timestep(
            params.K4,
            config.substep_as_float,
            *params.smagorinski_factor,
            *params.smagorinski_height,
            self._vertical_grid.interface_physical_height,
            self.diff_multfac_vn,
            self.smag_limit,
            self.enh_smag_fac,
            offset_provider={},
        )
        setup_program(
            backend=backend,
            program=diffusion_utils.init_nabla2_factor_in_upper_damping_zone,
            constant_args={
                "physical_heights": self._vertical_grid.interface_physical_height,
                "nshift": 0,
            },
            vertical_sizes={
                "vertical_start": 1,
                "vertical_end": gtx.int32(self._vertical_grid.end_index_of_damping_layer + 1),
                "end_index_of_damping_layer": self._vertical_grid.end_index_of_damping_layer,
                "heights_1": self._vertical_grid.interface_physical_height.ndarray[1].item(),
                "heights_nrd_shift": self._vertical_grid.interface_physical_height.ndarray[
                    self._vertical_grid.end_index_of_damping_layer + 1
                ].item(),
            },
        )(diff_multfac_n2w=self.diff_multfac_n2w)

    def _allocate_local_fields(self, allocator: gtx_typing.Allocator | None) -> None:
        self.diff_multfac_vn = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.diff_multfac_n2w = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.smag_limit = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.enh_smag_fac = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        self.u_vert = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, allocator=allocator
        )
        self.v_vert = data_alloc.zero_field(
            self._grid, dims.VertexDim, dims.KDim, allocator=allocator
        )
        self.kh_smag_e = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self.kh_smag_ec = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self.z_nabla2_e = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self.diff_multfac_smag = data_alloc.zero_field(self._grid, dims.KDim, allocator=allocator)
        # TODO(halungge): this is KHalfDim
        self.vertical_index = data_alloc.index_field(
            self._grid, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        )
        self.horizontal_cell_index = data_alloc.index_field(
            self._grid, dims.CellDim, allocator=allocator
        )
        self.horizontal_edge_index = data_alloc.index_field(
            self._grid, dims.EdgeDim, allocator=allocator
        )
        self.w_tmp = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        )
        self.theta_v_tmp = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )

    def _determine_horizontal_domains(self) -> None:
        cell_domain = h_grid.domain(dims.CellDim)
        edge_domain = h_grid.domain(dims.EdgeDim)
        vertex_domain = h_grid.domain(dims.VertexDim)

        def _get_start_index_for_w_diffusion() -> gtx.int32:
            return (
                self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
                if self._grid.limited_area
                else self._grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
            )

        self._cell_start_interior = self._grid.start_index(cell_domain(h_grid.Zone.INTERIOR))
        self._cell_start_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._cell_end_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._cell_end_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))

        self._edge_start_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._edge_start_nudging = self._grid.start_index(edge_domain(h_grid.Zone.NUDGING))
        self._edge_start_nudging_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
        )
        self._edge_end_local = self._grid.end_index(edge_domain(h_grid.Zone.LOCAL))
        self._edge_end_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))
        self._edge_end_halo_level_2 = self._grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))

        self._vertex_start_lateral_boundary_level_2 = self._grid.start_index(
            vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._vertex_end_local = self._grid.end_index(vertex_domain(h_grid.Zone.LOCAL))

        self._horizontal_start_index_w_diffusion = _get_start_index_for_w_diffusion()

    def run(
        self,
        diagnostic_state: diffusion_states.DiffusionDiagnosticState,
        prognostic_state: prognostics.PrognosticState,
        dtime: float,
        initial_run: bool = False,
    ) -> None:
        """
        Do one diffusion step.

        In ICON at the start of the simulation diffusion is run with linit=True:

        'For real-data runs, perform an extra diffusion call before the first time
        step because no other filtering of the interpolated velocity field is done'

        The initial run uses special values for diff_multfac_vn, smag_limit and smag_offset.
        """
        if initial_run:
            diff_multfac_vn = data_alloc.zero_field(
                self._grid, dims.KDim, allocator=self._allocator
            )
            smag_limit = data_alloc.zero_field(self._grid, dims.KDim, allocator=self._allocator)
            self.setup_fields_for_initial_step(
                self._params.K4,
                self.config.hdiff_efdt_ratio,
                diff_multfac_vn,
                smag_limit,
            )
            smag_offset = 0.0
        else:
            diff_multfac_vn = self.diff_multfac_vn
            smag_limit = self.smag_limit
            smag_offset = self.smag_offset

        self.scale_k(self.enh_smag_fac, dtime, self.diff_multfac_smag)

        log.debug("rbf interpolation 1: start")
        self.mo_intp_rbf_rbf_vec_interpol_vertex(
            p_e_in=prognostic_state.vn,
            p_u_out=self.u_vert,
            p_v_out=self.v_vert,
        )
        log.debug("rbf interpolation 1: end")

        # 2.  HALO EXCHANGE -- CALL sync_patch_array_mult u_vert and v_vert
        # TODO(phimuell, muellch): Is asynchronous mode okay here.
        log.debug("communication rbf extrapolation of vn - start")
        self._exchange(
            self.u_vert,
            self.v_vert,
            dim=dims.VertexDim,
            full_exchange=True,
            stream=decomposition.DEFAULT_STREAM,
        )
        log.debug("communication rbf extrapolation of vn - end")

        log.debug("running stencil 01(calculate_nabla2_and_smag_coefficients_for_vn): start")
        self.calculate_nabla2_and_smag_coefficients_for_vn(
            diff_multfac_smag=self.diff_multfac_smag,
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            vn=prognostic_state.vn,
            smag_limit=smag_limit,
            kh_smag_e=self.kh_smag_e,
            kh_smag_ec=self.kh_smag_ec,
            z_nabla2_e=self.z_nabla2_e,
            smag_offset=smag_offset,
        )
        log.debug("running stencil 01 (calculate_nabla2_and_smag_coefficients_for_vn): end")

        if (
            self.config.shear_type
            >= TurbulenceShearForcingType.VERTICAL_HORIZONTAL_OF_HORIZONTAL_WIND
            or self.config.loutshs
            or self.config.a_hshr > 0.0
        ):
            log.debug(
                "running stencils 02 03 (calculate_diagnostic_quantities_for_turbulence): start"
            )
            self.calculate_diagnostic_quantities_for_turbulence(
                kh_smag_ec=self.kh_smag_ec,
                vn=prognostic_state.vn,
                diff_multfac_smag=self.diff_multfac_smag,
                div_ic=diagnostic_state.div_ic,
                hdef_ic=diagnostic_state.hdef_ic,
            )
            log.debug(
                "running stencils 02 03 (calculate_diagnostic_quantities_for_turbulence): end"
            )

        log.debug("2nd rbf interpolation: start")
        self.mo_intp_rbf_rbf_vec_interpol_vertex(
            p_e_in=self.z_nabla2_e, p_u_out=self.u_vert, p_v_out=self.v_vert
        )
        log.debug("2nd rbf interpolation: end")

        # 6.  HALO EXCHANGE -- CALL sync_patch_array_mult (Vertex Fields)
        # TODO(phimuell, muellch): Is asynchronous mode okay here.
        log.debug("communication rbf extrapolation of z_nable2_e - start")
        self._exchange(
            self.u_vert,
            self.v_vert,
            dim=dims.VertexDim,
            full_exchange=True,
            stream=decomposition.DEFAULT_STREAM,
        )
        log.debug("communication rbf extrapolation of z_nable2_e - end")

        log.debug("running stencils 04 05 06 (apply_diffusion_to_vn): start")
        self.apply_diffusion_to_vn(
            u_vert=self.u_vert,
            v_vert=self.v_vert,
            z_nabla2_e=self.z_nabla2_e,
            kh_smag_e=self.kh_smag_e,
            diff_multfac_vn=diff_multfac_vn,
            vn=prognostic_state.vn,
        )
        log.debug("running stencils 04 05 06 (apply_diffusion_to_vn): end")

        log.debug("communication of prognostic.vn : start")
        handle_edge_comm = self._exchange(
            prognostic_state.vn,
            dim=dims.EdgeDim,
            full_exchange=False,
            stream=decomposition.DEFAULT_STREAM,
        )

        log.debug(
            "running stencils 07 08 09 10 (apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence): start"
        )
        # TODO(halungge): get rid of this copying. So far passing an empty buffer instead did not verify?
        self.copy_field(prognostic_state.w, self.w_tmp)

        self.apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
            w_old=self.w_tmp,
            w=prognostic_state.w,
            dwdx=diagnostic_state.dwdx,
            dwdy=diagnostic_state.dwdy,
            diff_multfac_w=self.diff_multfac_w,
            diff_multfac_n2w=self.diff_multfac_n2w,
        )
        log.debug(
            "running stencils 07 08 09 10 (apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence): end"
        )

        self.halo_exchange_wait(
            handle_edge_comm,
            stream=decomposition.DEFAULT_STREAM,
        )  # need to do this here, since we currently only use 1 communication object.
        log.debug("communication of prognostic.vn - end")

        if self.config.apply_to_temperature:
            log.debug(
                "running fused stencils 11 12 (calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools): start"
            )
            self.calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools(
                theta_v=prognostic_state.theta_v,
                kh_smag_e=self.kh_smag_e,
            )
            log.debug(
                "running stencils 11 12 (calculate_enhanced_diffusion_coefficients_for_grid_point_cold_pools): end"
            )
            log.debug("running stencil 13 to 16 (apply_diffusion_to_theta_and_exner): start")
            self.copy_field(
                prognostic_state.theta_v, self.theta_v_tmp
            )  # TODO(): write in a way that we can avoid the copy
            self.apply_diffusion_to_theta_and_exner(
                kh_smag_e=self.kh_smag_e,
                theta_v_in=self.theta_v_tmp,
                theta_v=prognostic_state.theta_v,
                exner=prognostic_state.exner,
            )
            # The halo exchange can be skipped in the case of NWP or AES physics because the column-wise physics
            # computations, which happen right after diffusion, do not require the halo lines to be correct and there
            # is another halo exchange after the physics are applied.
            log.debug("running stencil 13 to 16 apply_diffusion_to_theta_and_exner: end")
            if initial_run or self.config.iforcing not in (ForcingType.NWP, ForcingType.AES):
                log.debug("communication of prognostic cell fields: theta and exner - start")
                self._exchange.exchange(
                    dims.CellDim,
                    prognostic_state.theta_v,
                    prognostic_state.exner,
                    stream=decomposition.DEFAULT_STREAM,
                )
                log.debug("communication of prognostic cell fields: theta and exner - done")

        # The halo exchange can be skipped in the case of NWP or AES physics because the column-wise physics
        # computations, which happen right after diffusion, do not require the halo lines to be correct and there
        # is another halo exchange after the physics are applied.
        if initial_run or self.config.iforcing not in (ForcingType.NWP, ForcingType.AES):
            log.debug("communication of prognostic cell field: w - start")
            self._exchange.exchange(
                dims.CellDim,
                prognostic_state.w,
                stream=decomposition.DEFAULT_STREAM,
            )
            log.debug("communication of prognostic cell field: w - done")
