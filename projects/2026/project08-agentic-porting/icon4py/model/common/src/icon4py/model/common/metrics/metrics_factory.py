# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import logging
import math
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

import icon4py.model.common.metrics.compute_weight_factors as weight_factors
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    grid_refinement as refinement,
    horizontal as h_grid,
    icon,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.interpolation.stencils import cell_2_edge_interpolation
from icon4py.model.common.math import vertical_operations as vertical_ops
from icon4py.model.common.metrics import (
    compute_advection_metrics,
    compute_coeff_gradekin,
    compute_diffusion_metrics,
    compute_zdiff_gradp,
    metric_fields as mf,
    metrics_attributes as attrs,
    reference_atmosphere,
    reference_atmosphere as ra,
)
from icon4py.model.common.states import factory, model
from icon4py.model.common.utils import data_allocation as data_alloc, fortran_config


cell_domain = h_grid.domain(dims.CellDim)
edge_domain = h_grid.domain(dims.EdgeDim)
vertex_domain = h_grid.domain(dims.VertexDim)
vertical_domain = v_grid.domain(dims.KDim)
vertical_half_domain = v_grid.domain(dims.KHalfDim)
log = logging.getLogger(__name__)


@dataclasses.dataclass
class MetricsConfig:
    exner_expol: float = 1.0 / 3.0
    """
    Temporal extrapolation (fraction of dt) of Exner function for computation
    of horizontal pressure gradient. This damps horizontally propagating sound
    waves.
    """

    vwind_offctr: float = 0.15
    """
    Oﬀ-centering in vertical wind solver. Higher values may be needed for R2B5
    or coarser grids when the model top is above 50 km. Negative values are not
    allowed.
    """

    thslp_zdiffu: float = 0.025
    """
    Slope threshold above which truly horizontal temperature diﬀusion is
    activated.
    """

    thhgtd_zdiffu: float = 200.0
    """
    Threshold of height diﬀerence between neighboring grid points above which
    truly horizontal temperature diﬀusion is activated (alternative criterion
    to thslp_zdiﬀu).
    """

    # TODO (jcanton): move to vertical_grid.config
    rayleigh_type: constants.RayleighType = constants.RayleighType.KLEMP
    """
    Type of Rayleigh damping to be applied in the upper part of the model
    domain. Only KLEMP type is supported.
    """

    # TODO (jcanton): move to vertical_grid.config
    rayleigh_coeff: float = 0.05
    """
    Rayleigh damping coeﬃcient 1/tau_0 (Klemp, Dudhia, Hassiotis: MWR136,
    pp.3987-4004)
    """

    divdamp_trans_start: float = 12500.0
    """
    Lower bound of transition zone between 2D and 3D divergence damping.
    """

    divdamp_trans_end: float = 17500.0
    """
    Upper bound of transition zone between 2D and 3D divergence damping.
    """

    divdamp_type: int = 3
    """Type of divergence damping."""

    igradp_method: int = 3
    """Method for computing the horizontal pressure gradient."""

    def __post_init__(self):
        if self.rayleigh_type != constants.RayleighType.KLEMP:
            raise NotImplementedError(
                f"Only rayleigh_type = KLEMP is implemented, got {self.rayleigh_type}."
            )

    @classmethod
    def from_fortran_dict(cls, atmo_dict: dict[str, Any], **overrides: Any) -> MetricsConfig:
        nonhydrostatic_nml = atmo_dict["nonhydrostatic_nml"]
        return cls(
            exner_expol=nonhydrostatic_nml["exner_expol"],
            vwind_offctr=nonhydrostatic_nml["vwind_offctr"],
            thslp_zdiffu=nonhydrostatic_nml["thslp_zdiffu"],
            thhgtd_zdiffu=nonhydrostatic_nml["thhgtd_zdiffu"],
            rayleigh_type=constants.RayleighType(nonhydrostatic_nml["rayleigh_type"]),
            rayleigh_coeff=fortran_config.list_to_value(nonhydrostatic_nml["rayleigh_coeff"]),
            divdamp_trans_start=nonhydrostatic_nml["divdamp_trans_start"],
            divdamp_trans_end=nonhydrostatic_nml["divdamp_trans_end"],
            divdamp_type=nonhydrostatic_nml["divdamp_type"],
            igradp_method=nonhydrostatic_nml["igradp_method"],
            **overrides,
        )


class MetricsFieldsFactory(factory.FieldSource, factory.GridProvider):
    def __init__(
        self,
        *,
        grid: icon.IconGrid,
        vertical_grid: v_grid.VerticalGrid,
        decomposition_info: decomposition.DecompositionInfo,
        geometry_source: geometry.GridGeometry,
        topography: fa.CellField[ta.wpfloat],
        interpolation_source: interpolation_factory.InterpolationFieldsFactory,
        backend: gtx_typing.Backend | None,
        metadata: dict[str, model.FieldMetaData],
        config: MetricsConfig,
        exchange: decomposition.ExchangeRuntime = decomposition.single_node_exchange,
        global_reductions: decomposition.Reductions = decomposition.single_node_reductions,
    ):
        self._backend = backend
        self._xp = data_alloc.import_array_ns(backend)
        self._allocator = model_backends.get_allocator(backend)
        self._grid = grid
        self._vertical_grid = vertical_grid
        self._decomposition_info = decomposition_info
        self._attrs = metadata
        self._providers: dict[str, factory.FieldProvider] = {}
        self._geometry = geometry_source
        self._exchange = exchange
        self._interpolation_source = interpolation_source
        self._global_reductions = global_reductions
        log.info(
            f"initialized metrics factory for backend = '{self._backend_name()}' and grid = '{self._grid}'"
        )
        log.debug(f"using array_ns {self._xp} ")
        self._config = config
        self._vct_a_1 = self._vertical_grid.interface_physical_height.ndarray[0].item()
        self._damping_height = vertical_grid.config.rayleigh_damping_height

        k_index = data_alloc.index_field(
            self._grid, dims.KDim, extend={dims.KDim: 1}, allocator=self._allocator
        )
        e_lev = data_alloc.index_field(self._grid, dims.EdgeDim, allocator=self._allocator)
        e_owner_mask = gtx.as_field(
            (dims.EdgeDim,),
            self._decomposition_info.owner_mask(dims.EdgeDim),
            allocator=self._allocator,
        )
        c_owner_mask = gtx.as_field(
            (dims.CellDim,),
            self._decomposition_info.owner_mask(dims.CellDim),
            allocator=self._allocator,
        )
        c_refin_ctrl = self._grid.refinement_control[dims.CellDim]

        e_refin_ctrl = self._grid.refinement_control[dims.EdgeDim]
        self.register_provider(
            factory.PrecomputedFieldProvider(
                fields={
                    "topography": topography,
                    "vct_a": self._vertical_grid.interface_physical_height,
                    "height_u": self._vertical_grid.interface_physical_height[
                        : self._grid.num_levels
                    ],
                    "height_l": self._vertical_grid.interface_physical_height[
                        1 : self._grid.num_levels + 1
                    ],
                    "c_refin_ctrl": c_refin_ctrl,
                    "e_refin_ctrl": e_refin_ctrl,
                    "e_owner_mask": e_owner_mask,
                    "c_owner_mask": c_owner_mask,
                    "k_lev": k_index,
                    "e_lev": e_lev,
                }
            )
        )
        self._register_computed_fields()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} on (grid={self._grid!r}) providing fields f{self.metadata.keys()}"

    @property
    def _sources(self) -> factory.FieldSource:
        return factory.CompositeSource(me=self, others=(self._geometry, self._interpolation_source))

    def _register_computed_fields(self) -> None:  # noqa: PLR0915 [too-many-statements]
        vertical_coordinates_on_half_levels = factory.NumpyDataProvider(
            func=functools.partial(
                v_grid.compute_vertical_coordinate,
                exchange=self._exchange,
            ),
            fields=(attrs.CELL_HEIGHT_ON_HALF_LEVEL,),
            domain=(dims.CellDim, dims.KHalfDim),
            deps={
                "vct_a": "vct_a",
                "topography": "topography",
                "cell_areas": geometry_attrs.CELL_AREA,
                "geofac_n2s": interpolation_attributes.GEOFAC_N2S,
            },
            connectivities={"c2e2co": dims.C2E2CODim},
            params={
                "nflatlev": self._vertical_grid.nflatlev,
                "model_top_height": self._vertical_grid.config.model_top_height,
                "SLEVE_decay_scale_1": self.vertical_grid.config.SLEVE_decay_scale_1,
                "SLEVE_decay_exponent": self._vertical_grid.config.SLEVE_decay_exponent,
                "SLEVE_decay_scale_2": self._vertical_grid.config.SLEVE_decay_scale_2,
                "SLEVE_minimum_layer_thickness_1": self._vertical_grid.config._SLEVE_minimum_layer_thickness_1,
                "SLEVE_minimum_relative_layer_thickness_1": self._vertical_grid.config._SLEVE_minimum_relative_layer_thickness_1,
                "SLEVE_minimum_layer_thickness_2": self._vertical_grid.config._SLEVE_minimum_layer_thickness_2,
                "SLEVE_minimum_relative_layer_thickness_2": self._vertical_grid.config._SLEVE_minimum_relative_layer_thickness_2,
                "lowest_layer_thickness": self._vertical_grid.config.lowest_layer_thickness,
            },
        )
        self.register_provider(vertical_coordinates_on_half_levels)

        height = factory.ProgramFieldProvider(
            func=vertical_ops.average_two_vertical_levels_downwards_on_cells.with_backend(
                self._backend
            ),
            domain={
                dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.END)),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"average": attrs.Z_MC},
            deps={"input_field": attrs.CELL_HEIGHT_ON_HALF_LEVEL},
            do_exchange=False,
        )
        self.register_provider(height)

        compute_ddqz_z_half = factory.ProgramFieldProvider(
            func=mf.compute_ddqz_z_half.with_backend(self._backend),
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KHalfDim: (
                    vertical_half_domain(v_grid.Zone.TOP),
                    vertical_half_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"ddqz_z_half": attrs.DDQZ_Z_HALF},
            deps={
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "z_mc": attrs.Z_MC,
            },
            params={"nlev": self._grid.num_levels},
            do_exchange=False,
        )
        self.register_provider(compute_ddqz_z_half)

        ddqz_z_full_and_inverse = factory.ProgramFieldProvider(
            func=mf.compute_ddqz_z_full_and_inverse.with_backend(self._backend),
            deps={"z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL},
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"ddqz_z_full": attrs.DDQZ_Z_FULL, "inv_ddqz_z_full": attrs.INV_DDQZ_Z_FULL},
            do_exchange=False,
        )
        self.register_provider(ddqz_z_full_and_inverse)

        ddqz_full_on_edges = factory.ProgramFieldProvider(
            func=cell_2_edge_interpolation.cell_2_edge_interpolation.with_backend(self._backend),
            deps={"in_field": attrs.DDQZ_Z_FULL, "coeff": interpolation_attributes.C_LIN_E},
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LOCAL),
                    edge_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"out_field": attrs.DDQZ_Z_FULL_E},
            do_exchange=True,
        )
        self.register_provider(ddqz_full_on_edges)

        compute_scaling_factor_for_3d_divdamp = factory.ProgramFieldProvider(
            func=mf.compute_scaling_factor_for_3d_divdamp.with_backend(self._backend),
            domain={
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                )
            },
            fields={"scaling_factor_for_3d_divdamp": attrs.SCALING_FACTOR_FOR_3D_DIVDAMP},
            deps={"vct_a": "vct_a"},
            params={
                "divdamp_trans_start": self._config.divdamp_trans_start,
                "divdamp_trans_end": self._config.divdamp_trans_end,
                "divdamp_type": self._config.divdamp_type,
            },
            do_exchange=False,
        )
        self.register_provider(compute_scaling_factor_for_3d_divdamp)

        compute_rayleigh_w = factory.ProgramFieldProvider(
            func=mf.compute_rayleigh_w.with_backend(self._backend),
            deps={"vct_a": "vct_a"},
            domain={
                dims.KHalfDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    v_grid.Domain(dims.KHalfDim, v_grid.Zone.DAMPING, 1),
                )
            },
            fields={"rayleigh_w": attrs.RAYLEIGH_W},
            params={
                "damping_height": self._damping_height,
                "rayleigh_type": self._config.rayleigh_type,
                "rayleigh_coeff": self._config.rayleigh_coeff,
                "vct_a_1": self._vct_a_1,
                "pi_const": math.pi,
            },
            do_exchange=False,
        )
        self.register_provider(compute_rayleigh_w)

        compute_coeff_dwdz = factory.ProgramFieldProvider(
            func=mf.compute_coeff_dwdz.with_backend(self._backend),
            deps={
                "ddqz_z_full": attrs.DDQZ_Z_FULL,
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    v_grid.Domain(dims.KDim, v_grid.Zone.TOP, 1),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"coeff1_dwdz": attrs.COEFF1_DWDZ, "coeff2_dwdz": attrs.COEFF2_DWDZ},
            do_exchange=False,
        )
        self.register_provider(compute_coeff_dwdz)

        compute_theta_exner_rho_ref_mc = factory.ProgramFieldProvider(
            func=ra.compute_reference_atmosphere_cell_fields.with_backend(self._backend),
            deps={
                "z_height": attrs.Z_MC,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={
                "theta_ref_mc": attrs.THETA_REF_MC,
                "exner_ref_mc": attrs.EXNER_REF_MC,
                "rho_ref_mc": attrs.RHO_REF_MC,
            },
            params={
                "p0ref": constants.REFERENCE_PRESSURE,
                "p0sl_bg": constants.SEA_LEVEL_PRESSURE,
                "grav": constants.GRAV,
                "cpd": constants.CPD,
                "rd": constants.RD,
                "h_scal_bg": constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
                "t0sl_bg": constants.SEA_LEVEL_TEMPERATURE,
                "del_t_bg": constants.DELTA_TEMPERATURE,
            },
            do_exchange=False,
        )
        self.register_provider(compute_theta_exner_rho_ref_mc)

        compute_theta_rho_ref_me = factory.ProgramFieldProvider(
            func=ra.compute_reference_atmosphere_edge_fields.with_backend(self._backend),
            deps={"z_mc": attrs.Z_MC, "c_lin_e": interpolation_attributes.C_LIN_E},
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    edge_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={
                "rho_ref_me": attrs.RHO_REF_ME,
                "theta_ref_me": attrs.THETA_REF_ME,
            },
            params={
                "p0ref": constants.REFERENCE_PRESSURE,
                "p0sl_bg": constants.SEA_LEVEL_PRESSURE,
                "grav": constants.GRAV,
                "cpd": constants.CPD,
                "rd": constants.RD,
                "h_scal_bg": constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
                "t0sl_bg": constants.SEA_LEVEL_TEMPERATURE,
                "del_t_bg": constants.DELTA_TEMPERATURE,
            },
            do_exchange=True,
        )
        self.register_provider(compute_theta_rho_ref_me)

        compute_theta_d_exner_dz_ref_ic = factory.ProgramFieldProvider(
            func=ra.compute_theta_d_exner_dz_ref_ic.with_backend(self._backend),
            deps={
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KHalfDim: (
                    vertical_half_domain(v_grid.Zone.TOP),
                    vertical_half_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={
                "theta_ref_ic": attrs.THETA_REF_IC,
                "d_exner_dz_ref_ic": attrs.D_EXNER_DZ_REF_IC,
            },
            params={
                "t0sl_bg": constants.SEA_LEVEL_TEMPERATURE,
                "del_t_bg": constants.DELTA_TEMPERATURE,
                "h_scal_bg": constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
                "grav": constants.GRAV,
                "rd": constants.RD,
                "cpd": constants.CPD,
                "p0sl_bg": constants.SEA_LEVEL_PRESSURE,
                "rd_o_cpd": constants.RD_O_CPD,
                "p0ref": constants.REFERENCE_PRESSURE,
            },
            do_exchange=False,
        )
        self.register_provider(compute_theta_d_exner_dz_ref_ic)

        compute_d2dexdz2_fac_mc = factory.ProgramFieldProvider(
            func=reference_atmosphere.compute_d2dexdz2_fac_mc.with_backend(self._backend),
            deps={
                "theta_ref_mc": attrs.THETA_REF_MC,
                "inv_ddqz_z_full": attrs.INV_DDQZ_Z_FULL,
                "exner_ref_mc": attrs.EXNER_REF_MC,
                "z_mc": attrs.Z_MC,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={
                attrs.D2DEXDZ2_FAC1_MC: attrs.D2DEXDZ2_FAC1_MC,
                attrs.D2DEXDZ2_FAC2_MC: attrs.D2DEXDZ2_FAC2_MC,
            },
            params={
                "cpd": constants.CPD,
                "grav": constants.GRAV,
                "del_t_bg": constants.DEL_T_BG,
                "h_scal_bg": constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
            },
            do_exchange=False,
        )
        self.register_provider(compute_d2dexdz2_fac_mc)

        compute_ddxt_z_half_e = factory.ProgramFieldProvider(
            func=mf.compute_ddxt_z_half_e.with_backend(self._backend),
            deps={
                "cell_in": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "c_int": interpolation_attributes.CELL_AW_VERTS,
                "inv_primal_edge_length": f"inverse_of_{geometry_attrs.EDGE_LENGTH}",
                "tangent_orientation": geometry_attrs.TANGENT_ORIENTATION,
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3),
                    edge_domain(h_grid.Zone.INTERIOR),
                ),
                dims.KHalfDim: (
                    vertical_half_domain(v_grid.Zone.TOP),
                    vertical_half_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.DDXT_Z_HALF_E: attrs.DDXT_Z_HALF_E},
            do_exchange=True,
        )
        self.register_provider(compute_ddxt_z_half_e)

        compute_ddxn_z_half_e = factory.ProgramFieldProvider(
            func=mf.compute_ddxn_z_half_e.with_backend(self._backend),
            deps={
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "inv_dual_edge_length": f"inverse_of_{geometry_attrs.DUAL_EDGE_LENGTH}",
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    edge_domain(h_grid.Zone.INTERIOR),
                ),
                dims.KHalfDim: (
                    vertical_half_domain(v_grid.Zone.TOP),
                    vertical_half_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.DDXN_Z_HALF_E: attrs.DDXN_Z_HALF_E},
            do_exchange=True,
        )
        self.register_provider(compute_ddxn_z_half_e)

        # ddxn_z_full is dependent only on attrs.DDXN_Z_HALF_E, which has halo exchange. That's why halo_exchange is set to True
        compute_ddxn_z_full = factory.ProgramFieldProvider(
            func=vertical_ops.average_two_vertical_levels_downwards_on_edges.with_backend(
                self._backend
            ),
            deps={
                "input_field": attrs.DDXN_Z_HALF_E,
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LOCAL),
                    edge_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"average": attrs.DDXN_Z_FULL},
            do_exchange=False,
        )
        self.register_provider(compute_ddxn_z_full)

        compute_ddxt_z_full = factory.ProgramFieldProvider(
            func=vertical_ops.average_two_vertical_levels_downwards_on_edges.with_backend(
                self._backend
            ),
            deps={
                "input_field": attrs.DDXT_Z_HALF_E,
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    edge_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"average": attrs.DDXT_Z_FULL},
            do_exchange=True,
        )
        self.register_provider(compute_ddxt_z_full)

        compute_exner_w_implicit_weight_parameter_np = factory.NumpyDataProvider(
            func=mf.compute_exner_w_implicit_weight_parameter,
            domain=(dims.CellDim,),
            connectivities={"c2e": dims.C2EDim},
            fields=(attrs.EXNER_W_IMPLICIT_WEIGHT_PARAMETER,),
            deps={
                "vct_a": "vct_a",
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "z_ddxn_z_half_e": attrs.DDXN_Z_HALF_E,
                "z_ddxt_z_half_e": attrs.DDXT_Z_HALF_E,
                "dual_edge_length": geometry_attrs.DUAL_EDGE_LENGTH,
            },
            params={
                "vwind_offctr": self._config.vwind_offctr,
                "nlev": self._grid.num_levels,
                "horizontal_start_cell": self._grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
            },
        )
        self.register_provider(compute_exner_w_implicit_weight_parameter_np)

        compute_exner_w_explicit_weight_parameter = factory.ProgramFieldProvider(
            func=mf.compute_exner_w_explicit_weight_parameter.with_backend(self._backend),
            deps={
                "exner_w_implicit_weight_parameter": attrs.EXNER_W_IMPLICIT_WEIGHT_PARAMETER,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
            },
            fields={"exner_w_explicit_weight_parameter": attrs.EXNER_W_EXPLICIT_WEIGHT_PARAMETER},
            do_exchange=False,
        )
        self.register_provider(compute_exner_w_explicit_weight_parameter)

        compute_exner_exfac = factory.ProgramFieldProvider(
            func=mf.compute_exner_exfac.with_backend(self._backend),
            deps={
                "maxslp": attrs.MAXSLP,
                "maxhgtd": attrs.MAXHGTD,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.EXNER_EXFAC: attrs.EXNER_EXFAC},
            params={
                "exner_expol": self._config.exner_expol,
                "lateral_boundary_level_2": self._grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
            },
            do_exchange=False,
        )
        self.register_provider(compute_exner_exfac)

        wgtfac_c_provider = factory.ProgramFieldProvider(
            func=weight_factors.compute_wgtfac_c.with_backend(self._backend),
            deps={
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
            },
            domain={
                dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.END)),
                dims.KHalfDim: (
                    vertical_half_domain(v_grid.Zone.TOP),
                    vertical_half_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.WGTFAC_C: attrs.WGTFAC_C},
            params={"nlev": self._grid.num_levels},
            do_exchange=False,
        )
        self.register_provider(wgtfac_c_provider)

        compute_wgtfac_e = factory.ProgramFieldProvider(
            func=mf.compute_wgtfac_e.with_backend(self._backend),
            deps={
                "wgtfac_c": attrs.WGTFAC_C,
                "c_lin_e": interpolation_attributes.C_LIN_E,
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LOCAL),
                    edge_domain(h_grid.Zone.LOCAL),
                ),
                dims.KHalfDim: (
                    vertical_half_domain(v_grid.Zone.TOP),
                    vertical_half_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"wgtfac_e": attrs.WGTFAC_E},
            do_exchange=True,
        )
        self.register_provider(compute_wgtfac_e)

        max_flat_index_provider = factory.NumpyDataProvider(
            func=functools.partial(
                mf.compute_flat_max_idx,
                exchange=self._exchange,
            ),
            deps={
                "z_mc": attrs.Z_MC,
                "c_lin_e": interpolation_attributes.C_LIN_E,
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "k_lev": "k_lev",
            },
            connectivities={"e2c": dims.E2CDim},
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    edge_domain(h_grid.Zone.END),
                ),
            },
            fields={"flat_idx_max": attrs.FLAT_IDX_MAX},
        )
        self.register_provider(max_flat_index_provider)

        nflat_gradp_provider = factory.NumpyDataProvider(
            func=functools.partial(
                mf.compute_nflat_gradp,
                min_reduction=self._global_reductions.min,
            ),
            domain=(),
            deps={
                "flat_idx_max": attrs.FLAT_IDX_MAX,
                "e_owner_mask": "e_owner_mask",
            },
            params={
                "lateral_boundary_level": self._grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
                "nlev": self._grid.num_levels,
            },
            fields=(attrs.NFLAT_GRADP,),
        )
        self.register_provider(nflat_gradp_provider)

        pressure_gradient_fields = factory.ProgramFieldProvider(
            func=mf.compute_pressure_gradient_downward_extrapolation_mask_distance.with_backend(
                self._backend
            ),
            deps={
                "z_mc": attrs.Z_MC,
                "c_lin_e": interpolation_attributes.C_LIN_E,
                "topography": "topography",
                "e_owner_mask": "e_owner_mask",
                "flat_idx_max": attrs.FLAT_IDX_MAX,
                "e_lev": "e_lev",
                "k_lev": "k_lev",
            },
            params={
                "horizontal_start_distance": self._grid.end_index(edge_domain(h_grid.Zone.NUDGING)),
                "horizontal_end_distance": self._grid.end_index(edge_domain(h_grid.Zone.LOCAL)),
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.NUDGING_LEVEL_2),
                    edge_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={"pg_exdist_dsl": attrs.PG_EXDIST_DSL},
            do_exchange=False,
        )
        self.register_provider(pressure_gradient_fields)

        compute_mask_prog_halo_c = factory.ProgramFieldProvider(
            func=mf.compute_mask_prog_halo_c.with_backend(self._backend),
            deps={
                "c_refin_ctrl": "c_refin_ctrl",
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.HALO),
                    cell_domain(h_grid.Zone.END),
                ),
            },
            fields={
                attrs.MASK_PROG_HALO_C: attrs.MASK_PROG_HALO_C,
            },
            do_exchange=False,
        )
        self.register_provider(compute_mask_prog_halo_c)

        compute_horizontal_mask_for_3d_divdamp = factory.ProgramFieldProvider(
            func=mf.compute_horizontal_mask_for_3d_divdamp.with_backend(self._backend),
            deps={
                "e_refin_ctrl": "e_refin_ctrl",
            },
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    edge_domain(h_grid.Zone.LOCAL),
                )
            },
            fields={attrs.HORIZONTAL_MASK_FOR_3D_DIVDAMP: attrs.HORIZONTAL_MASK_FOR_3D_DIVDAMP},
            params={
                "grf_nudge_start_e": refinement.get_nudging_refinement_value(dims.EdgeDim),  # type: ignore [attr-defined]
                "grf_nudgezone_width": gtx.int32(refinement.DEFAULT_GRF_NUDGEZONE_WIDTH),  # type: ignore [attr-defined]
            },
            do_exchange=True,
        )
        self.register_provider(compute_horizontal_mask_for_3d_divdamp)

        compute_zdiff_gradp_np = factory.NumpyDataProvider(
            func=functools.partial(
                compute_zdiff_gradp.compute_zdiff_gradp,
                exchange=self._exchange,
            ),
            deps={
                "z_mc": attrs.Z_MC,
                "c_lin_e": interpolation_attributes.C_LIN_E,
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "flat_idx": attrs.FLAT_IDX_MAX,
                "topography": "topography",
            },
            connectivities={"e2c": dims.E2CDim},
            domain=(dims.EdgeDim, dims.E2CDim, dims.KDim),
            fields=(
                attrs.ZDIFF_GRADP,
                attrs.VERTOFFSET_GRADP,
            ),
            params={
                "nlev": self._grid.num_levels,
                "horizontal_start": self._grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
                "horizontal_start_1": self._grid.start_index(
                    edge_domain(h_grid.Zone.NUDGING_LEVEL_2)
                ),
            },
        )
        self.register_provider(compute_zdiff_gradp_np)

        coeff_gradekin = factory.NumpyDataProvider(
            func=compute_coeff_gradekin.compute_coeff_gradekin,
            domain=(dims.EdgeDim, dims.E2CDim),
            fields=(attrs.COEFF_GRADEKIN,),
            deps={
                "edge_cell_length": geometry_attrs.EDGE_CELL_DISTANCE,
                "inv_dual_edge_length": f"inverse_of_{geometry_attrs.DUAL_EDGE_LENGTH}",
            },
            params={
                "horizontal_start": self._grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
            },
            do_exchange=True,
        )
        self.register_provider(coeff_gradekin)

        compute_wgtfacq_c = factory.NumpyDataProvider(
            func=weight_factors.compute_wgtfacq_c_dsl,
            domain=gtx.domain(
                {
                    dims.CellDim: (0, self._grid.num_cells),
                    dims.KDim: (self._grid.num_levels - 3, self._grid.num_levels),
                }
            ),
            fields=(attrs.WGTFACQ_C,),
            deps={"z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL},
            params={"nlev": self._grid.num_levels},
        )

        self.register_provider(compute_wgtfacq_c)

        compute_wgtfacq_e = factory.NumpyDataProvider(
            func=functools.partial(
                weight_factors.compute_wgtfacq_e_dsl,
                exchange=self._exchange,
            ),
            deps={
                "z_ifc": attrs.CELL_HEIGHT_ON_HALF_LEVEL,
                "c_lin_e": interpolation_attributes.C_LIN_E,
                "wgtfacq_c_dsl": attrs.WGTFACQ_C,
            },
            connectivities={"e2c": dims.E2CDim},
            domain=gtx.domain(
                {
                    dims.EdgeDim: (0, self._grid.num_edges),
                    dims.KDim: (self._grid.num_levels - 3, self._grid.num_levels),
                }
            ),
            fields=(attrs.WGTFACQ_E,),
            params={"n_edges": self._grid.num_edges, "nlev": self._grid.num_levels},
        )

        self.register_provider(compute_wgtfacq_e)

        compute_maxslp_maxhgtd = factory.ProgramFieldProvider(
            func=mf.compute_maxslp_maxhgtd.with_backend(self._backend),
            deps={
                "ddxn_z_full": attrs.DDXN_Z_FULL,
                "dual_edge_length": geometry_attrs.DUAL_EDGE_LENGTH,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LOCAL),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.MAXSLP: attrs.MAXSLP, attrs.MAXHGTD: attrs.MAXHGTD},
            do_exchange=True,
        )
        self.register_provider(compute_maxslp_maxhgtd)

        compute_weighted_cell_neighbor_sum = factory.ProgramFieldProvider(
            func=mf.compute_weighted_cell_neighbor_sum.with_backend(self._backend),
            deps={
                "maxslp": attrs.MAXSLP,
                "maxhgtd": attrs.MAXHGTD,
                "c_bln_avg": interpolation_attributes.C_BLN_AVG,
            },
            domain={
                dims.CellDim: (
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    cell_domain(h_grid.Zone.END),
                ),
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={attrs.MAXSLP_AVG: attrs.MAXSLP_AVG, attrs.MAXHGTD_AVG: attrs.MAXHGTD_AVG},
            do_exchange=True,
        )
        self.register_provider(compute_weighted_cell_neighbor_sum)

        compute_max_nbhgt = factory.NumpyDataProvider(
            func=compute_diffusion_metrics.compute_max_nbhgt_array_ns,
            deps={
                "z_mc": attrs.Z_MC,
            },
            connectivities={"c2e2c": dims.C2E2CDim},
            domain=(dims.CellDim,),
            fields=(attrs.MAX_NBHGT,),
            params={
                "nlev": self._grid.num_levels,
            },
            do_exchange=True,
        )
        self.register_provider(compute_max_nbhgt)

        compute_diffusion_mask_and_coef = factory.NumpyDataProvider(
            func=compute_diffusion_metrics.compute_diffusion_mask_and_coef,
            deps={
                "z_mc": attrs.Z_MC,
                "max_nbhgt": attrs.MAX_NBHGT,
                "c_owner_mask": "c_owner_mask",
                "maxslp_avg": attrs.MAXSLP_AVG,
                "maxhgtd_avg": attrs.MAXHGTD_AVG,
            },
            connectivities={"c2e2c": dims.C2E2CDim},
            domain=(dims.CellDim, dims.KDim),
            fields=(attrs.ZD_DIFFCOEF,),
            params={
                "thslp_zdiffu": self._config.thslp_zdiffu,
                "thhgtd_zdiffu": self._config.thhgtd_zdiffu,
                "cell_nudging": self._grid.start_index(
                    h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING)
                ),
                "nlev": self._grid.num_levels,
            },
        )

        self.register_provider(compute_diffusion_mask_and_coef)

        compute_diffusion_intcoef_and_vertoffset = factory.NumpyDataProvider(
            func=compute_diffusion_metrics.compute_diffusion_intcoef_and_vertoffset,
            deps={
                "z_mc": attrs.Z_MC,
                "max_nbhgt": attrs.MAX_NBHGT,
                "c_owner_mask": "c_owner_mask",
                "maxslp_avg": attrs.MAXSLP_AVG,
                "maxhgtd_avg": attrs.MAXHGTD_AVG,
            },
            connectivities={"c2e2c": dims.C2E2CDim},
            domain=(dims.CellDim, dims.C2E2CDim, dims.KDim),
            fields=(
                attrs.ZD_INTCOEF,
                attrs.ZD_VERTOFFSET,
            ),
            params={
                "thslp_zdiffu": self._config.thslp_zdiffu,
                "thhgtd_zdiffu": self._config.thhgtd_zdiffu,
                "cell_nudging": self._grid.start_index(
                    h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING)
                ),
                "nlev": self._grid.num_levels,
            },
        )

        self.register_provider(compute_diffusion_intcoef_and_vertoffset)

        compute_advection_deepatmo_fields = factory.ProgramFieldProvider(
            func=compute_advection_metrics.compute_advection_deepatmo_fields.with_backend(
                self._backend
            ),
            domain={
                dims.KDim: (
                    vertical_domain(v_grid.Zone.TOP),
                    vertical_domain(v_grid.Zone.BOTTOM),
                ),
            },
            fields={
                attrs.DEEPATMO_DIVH: attrs.DEEPATMO_DIVH,
                attrs.DEEPATMO_DIVZL: attrs.DEEPATMO_DIVZL,
                attrs.DEEPATMO_DIVZU: attrs.DEEPATMO_DIVZU,
            },
            deps={
                "height_u": "height_u",
                "height_l": "height_l",
            },
            params={"grid_sphere_radius": constants.EARTH_RADIUS},
            do_exchange=False,
        )

        self.register_provider(compute_advection_deepatmo_fields)

    def get_int32(self, name: str) -> gtx.int32:
        return gtx.int32(self.get(name, factory.RetrievalType.SCALAR))

    @property
    def metadata(self) -> dict[str, model.FieldMetaData]:
        return self._attrs

    @property
    def backend(self) -> gtx_typing.Backend | None:
        return self._backend

    @property
    def grid(self) -> icon.IconGrid:
        return self._grid

    @property
    def vertical_grid(self) -> v_grid.VerticalGrid:
        return self._vertical_grid
