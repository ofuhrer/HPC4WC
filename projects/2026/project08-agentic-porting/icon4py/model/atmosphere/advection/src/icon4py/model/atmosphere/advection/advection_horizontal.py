# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from abc import ABC, abstractmethod

import gt4py.next as gtx

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.atmosphere.advection.stencils.apply_positive_definite_horizontal_multiplicative_flux_factor import (
    apply_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.atmosphere.advection.stencils.compute_barycentric_backtrajectory_alt import (
    compute_barycentric_backtrajectory_alt,
)
from icon4py.model.atmosphere.advection.stencils.compute_edge_tangential import (
    compute_edge_tangential,
)
from icon4py.model.atmosphere.advection.stencils.compute_horizontal_tracer_flux_from_linear_coefficients_alt import (
    compute_horizontal_tracer_flux_from_linear_coefficients_alt,
)
from icon4py.model.atmosphere.advection.stencils.compute_positive_definite_horizontal_multiplicative_flux_factor import (
    compute_positive_definite_horizontal_multiplicative_flux_factor,
)
from icon4py.model.atmosphere.advection.stencils.copy_cell_kdim_field import copy_cell_kdim_field
from icon4py.model.atmosphere.advection.stencils.integrate_tracer_horizontally import (
    integrate_tracer_horizontally,
)
from icon4py.model.atmosphere.advection.stencils.reconstruct_linear_coefficients_svd import (
    reconstruct_linear_coefficients_svd,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.utils import data_allocation as data_alloc


"""Advection components related to horizontal transport."""

log = logging.getLogger(__name__)


class HorizontalFluxLimiter(ABC):
    """Class that limits the horizontal finite volume numerical flux."""

    @abstractmethod
    def apply_flux_limiter(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None: ...


class NoLimiter(HorizontalFluxLimiter):
    """Do not apply any limiting."""

    def apply_flux_limiter(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None: ...


class PositiveDefinite(HorizontalFluxLimiter):
    """Class that implements a positive definite horizontal flux limiter."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        interpolation_state: advection_states.AdvectionInterpolationState,
        backend: gtx.typing.Backend | None,
        exchange: decomposition.ExchangeRuntime,
    ):
        self._grid = grid
        self._interpolation_state = interpolation_state
        self._backend = backend
        self._exchange = exchange

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        # edge indices
        edge_domain = h_grid.domain(dims.EdgeDim)

        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))

        # limiter fields
        self._r_m = data_alloc.zero_field(
            self._grid,
            dims.CellDim,
            dims.KDim,
            allocator=model_backends.get_allocator(self._backend),
        )

        # stencils
        self._compute_positive_definite_horizontal_multiplicative_flux_factor = (
            model_options.setup_program(
                backend=self._backend,
                program=compute_positive_definite_horizontal_multiplicative_flux_factor,
                constant_args={
                    "geofac_div": self._interpolation_state.geofac_div,
                    "dbl_eps": constants.DBL_EPS,
                },
                horizontal_sizes={
                    "horizontal_start": self._start_cell_lateral_boundary_level_2,
                    "horizontal_end": self._end_cell_local,
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(0),
                    "vertical_end": gtx.int32(self._grid.num_levels),
                },
                offset_provider=self._grid.connectivities,
            )
        )
        self._apply_positive_definite_horizontal_multiplicative_flux_factor = (
            model_options.setup_program(
                backend=self._backend,
                program=apply_positive_definite_horizontal_multiplicative_flux_factor,
                horizontal_sizes={
                    "horizontal_start": self._start_edge_lateral_boundary_level_5,
                    "horizontal_end": self._end_edge_halo,
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(0),
                    "vertical_end": gtx.int32(self._grid.num_levels),
                },
                offset_provider=self._grid.connectivities,
            )
        )

    def apply_flux_limiter(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        # compute multiplicative flux factor to guarantee no undershoot
        log.debug(
            "running stencil compute_positive_definite_horizontal_multiplicative_flux_factor - start"
        )
        self._compute_positive_definite_horizontal_multiplicative_flux_factor(
            p_cc=p_tracer_now,
            p_rhodz_now=rhodz_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            r_m=self._r_m,
            p_dtime=dtime,
        )
        log.debug(
            "running stencil compute_positive_definite_horizontal_multiplicative_flux_factor - end"
        )

        log.debug("communication of advection cell field: r_m - start")
        self._exchange.exchange(dims.CellDim, self._r_m, stream=decomposition.DEFAULT_STREAM)
        log.debug("communication of advection cell field: r_m - end")

        # limit outward fluxes
        log.debug(
            "running stencil apply_positive_definite_horizontal_multiplicative_flux_factor - start"
        )
        self._apply_positive_definite_horizontal_multiplicative_flux_factor(
            r_m=self._r_m,
            p_mflx_tracer_h=p_mflx_tracer_h,
        )
        log.debug(
            "running stencil apply_positive_definite_horizontal_multiplicative_flux_factor - end"
        )


class SemiLagrangianTracerFlux(ABC):
    """Class that defines the horizontal semi-Lagrangian tracer flux."""

    @abstractmethod
    def compute_tracer_flux(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_distv_bary_1: fa.EdgeKField[ta.anyfloat],
        p_distv_bary_2: fa.EdgeKField[ta.anyfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None: ...


class SecondOrderMiura(SemiLagrangianTracerFlux):
    """Class that computes a Miura-based second-order accurate tracer flux."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        least_squares_state: advection_states.AdvectionLeastSquaresState,
        backend: gtx.typing.Backend | None,
        horizontal_limiter: HorizontalFluxLimiter | None = None,
    ):
        self._grid = grid
        self._least_squares_state = least_squares_state
        self._backend = backend
        self._horizontal_limiter = horizontal_limiter or NoLimiter()

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._end_cell_halo = self._grid.end_index(cell_domain(h_grid.Zone.HALO))

        # edge indices
        edge_domain = h_grid.domain(dims.EdgeDim)
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))

        # reconstruction fields
        allocator = model_backends.get_allocator(self._backend)
        self._p_coeff_1 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._p_coeff_2 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._p_coeff_3 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )

        # stencils
        self._reconstruct_linear_coefficients_svd = model_options.setup_program(
            backend=self._backend,
            program=reconstruct_linear_coefficients_svd,
            horizontal_sizes={
                "horizontal_start": self._start_cell_lateral_boundary_level_2,
                "horizontal_end": self._end_cell_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._compute_horizontal_tracer_flux_from_linear_coefficients_alt = (
            model_options.setup_program(
                backend=self._backend,
                program=compute_horizontal_tracer_flux_from_linear_coefficients_alt,
                horizontal_sizes={
                    "horizontal_start": self._start_edge_lateral_boundary_level_5,
                    "horizontal_end": self._end_edge_halo,
                },
                vertical_sizes={
                    "vertical_start": gtx.int32(0),
                    "vertical_end": gtx.int32(self._grid.num_levels),
                },
                offset_provider=self._grid.connectivities,
            )
        )

    def compute_tracer_flux(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        p_distv_bary_1: fa.EdgeKField[ta.anyfloat],
        p_distv_bary_2: fa.EdgeKField[ta.anyfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal tracer flux computation - start")

        # linear reconstruction using singular value decomposition
        log.debug("running stencil reconstruct_linear_coefficients_svd - start")
        self._reconstruct_linear_coefficients_svd(
            p_cc=p_tracer_now,
            lsq_pseudoinv_1=self._least_squares_state.lsq_pseudoinv_1,
            lsq_pseudoinv_2=self._least_squares_state.lsq_pseudoinv_2,
            p_coeff_1_dsl=self._p_coeff_1,
            p_coeff_2_dsl=self._p_coeff_2,
            p_coeff_3_dsl=self._p_coeff_3,
        )
        log.debug("running stencil reconstruct_linear_coefficients_svd - end")

        # compute reconstructed tracer value at each barycenter and corresponding flux at each edge
        log.debug(
            "running stencil compute_horizontal_tracer_flux_from_linear_coefficients_alt - start"
        )
        self._compute_horizontal_tracer_flux_from_linear_coefficients_alt(
            z_lsq_coeff_1=self._p_coeff_1,
            z_lsq_coeff_2=self._p_coeff_2,
            z_lsq_coeff_3=self._p_coeff_3,
            distv_bary_1=p_distv_bary_1,
            distv_bary_2=p_distv_bary_2,
            p_mass_flx_e=prep_adv.mass_flx_me,
            p_vn=prep_adv.vn_traj,
            p_out_e=p_mflx_tracer_h,
        )
        log.debug(
            "running stencil compute_horizontal_tracer_flux_from_linear_coefficients_alt - end"
        )

        self._horizontal_limiter.apply_flux_limiter(
            p_tracer_now=p_tracer_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            rhodz_now=rhodz_now,
            dtime=dtime,
        )

        log.debug("horizontal tracer flux computation - end")


class HorizontalAdvection(ABC):
    """Class that does one horizontal advection step."""

    @abstractmethod
    def run(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        """
        Run a horizontal advection step.

        Args:
            prep_adv: input argument, data class that contains precalculated advection fields
            p_tracer_now: input argument, field that contains current tracer mass fraction
            p_tracer_new: output argument, field that contains new tracer mass fraction
            rhodz_now: input argument, field that contains current air mass in each layer
            rhodz_new: input argument, field that contains new air mass in each layer
            p_mflx_tracer_h: output argument, field that contains new horizontal tracer mass flux
            dtime: input argument, the time step

        """
        ...


class NoAdvection(HorizontalAdvection):
    """Class that implements disabled horizontal advection."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        backend: gtx.typing.Backend | None,
    ):
        log.debug("horizontal advection class init - start")

        # input arguments
        self._backend = model_options.customize_backend(program=None, backend=backend)

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        # stencils
        self._copy_cell_kdim_field = model_options.setup_program(
            backend=self._backend,
            program=copy_cell_kdim_field,
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(grid.num_levels),
            },
            offset_provider=grid.connectivities,
        )

        log.debug("horizontal advection class init - end")

    def run(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal advection run - start")

        log.debug("running stencil copy_cell_kdim_field - start")
        self._copy_cell_kdim_field(
            field_in=p_tracer_now,
            field_out=p_tracer_new,
        )
        log.debug("running stencil copy_cell_kdim_field - end")
        log.debug("horizontal advection run - end")


class FiniteVolume(HorizontalAdvection):
    """Class that defines a finite volume horizontal advection scheme."""

    def run(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal advection run - start")

        self._compute_numerical_flux(
            prep_adv=prep_adv,
            p_tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            dtime=dtime,
        )

        self._update_unknowns(
            p_tracer_now=p_tracer_now,
            p_tracer_new=p_tracer_new,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            p_mflx_tracer_h=p_mflx_tracer_h,
            dtime=dtime,
        )
        log.debug("horizontal advection run - end")

    @abstractmethod
    def _compute_numerical_flux(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None: ...

    @abstractmethod
    def _update_unknowns(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None: ...


class SemiLagrangian(FiniteVolume):
    """Class that does one horizontal semi-Lagrangian finite volume advection step."""

    def __init__(
        self,
        *,
        tracer_flux: SemiLagrangianTracerFlux,
        grid: icon_grid.IconGrid,
        interpolation_state: advection_states.AdvectionInterpolationState,
        metric_state: advection_states.AdvectionMetricState,
        edge_params: grid_states.EdgeParams,
        cell_params: grid_states.CellParams,
        backend: gtx.typing.Backend | None,
    ):
        log.debug("horizontal advection class init - start")

        # input arguments
        self._tracer_flux = tracer_flux
        self._grid = grid
        self._interpolation_state = interpolation_state
        self._metric_state = metric_state
        self._edge_params = edge_params
        self._cell_params = cell_params
        self._backend = backend

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        # edge indices
        edge_domain = h_grid.domain(dims.EdgeDim)
        self._start_edge_lateral_boundary_level_2 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._start_edge_lateral_boundary_level_5 = self._grid.start_index(
            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
        )
        self._end_edge_halo = self._grid.end_index(edge_domain(h_grid.Zone.HALO))

        # backtrajectory fields
        allocator = model_backends.get_allocator(self._backend)
        self._z_real_vt = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self._p_distv_bary_1 = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )
        self._p_distv_bary_2 = data_alloc.zero_field(
            self._grid, dims.EdgeDim, dims.KDim, allocator=allocator
        )

        # stencils
        self._compute_edge_tangential = model_options.setup_program(
            backend=self._backend,
            program=compute_edge_tangential,
            constant_args={
                "ptr_coeff": self._interpolation_state.rbf_vec_coeff_e,
            },
            horizontal_sizes={
                "horizontal_start": self._start_edge_lateral_boundary_level_2,
                "horizontal_end": self._end_edge_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        self._compute_barycentric_backtrajectory_alt = model_options.setup_program(
            backend=self._backend,
            program=compute_barycentric_backtrajectory_alt,
            constant_args={
                "pos_on_tplane_e_1": self._interpolation_state.pos_on_tplane_e_1,
                "pos_on_tplane_e_2": self._interpolation_state.pos_on_tplane_e_2,
                "primal_normal_cell_1": self._edge_params.primal_normal_cell[0],
                "dual_normal_cell_1": self._edge_params.dual_normal_cell[0],
                "primal_normal_cell_2": self._edge_params.primal_normal_cell[1],
                "dual_normal_cell_2": self._edge_params.dual_normal_cell[1],
            },
            horizontal_sizes={
                "horizontal_start": self._start_edge_lateral_boundary_level_5,
                "horizontal_end": self._end_edge_halo,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": self._grid.num_levels,
            },
            offset_provider=self._grid.connectivities,
        )
        self._integrate_tracer_horizontally = model_options.setup_program(
            backend=self._backend,
            program=integrate_tracer_horizontally,
            constant_args={
                "deepatmo_divh": self._metric_state.deepatmo_divh,
                "geofac_div": self._interpolation_state.geofac_div,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": self._grid.num_levels,
            },
            offset_provider=self._grid.connectivities,
        )

        log.debug("horizontal advection class init - end")

    def _compute_numerical_flux(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal numerical flux computation - start")

        ## tracer-independent part

        # compute tangential velocity
        log.debug("running stencil compute_edge_tangential - start")
        self._compute_edge_tangential(
            p_vn_in=prep_adv.vn_traj,
            p_vt_out=self._z_real_vt,
        )
        log.debug("running stencil compute_edge_tangential - end")

        # backtrajectory calculation
        log.debug("running stencil compute_barycentric_backtrajectory_alt - start")
        self._compute_barycentric_backtrajectory_alt(
            p_vn=prep_adv.vn_traj,
            p_vt=self._z_real_vt,
            p_distv_bary_1=self._p_distv_bary_1,
            p_distv_bary_2=self._p_distv_bary_2,
            p_dthalf=0.5 * dtime,
        )
        log.debug("running stencil compute_barycentric_backtrajectory_alt - end")

        ## tracer-specific part

        self._tracer_flux.compute_tracer_flux(
            prep_adv=prep_adv,
            p_tracer_now=p_tracer_now,
            p_mflx_tracer_h=p_mflx_tracer_h,
            p_distv_bary_1=self._p_distv_bary_1,
            p_distv_bary_2=self._p_distv_bary_2,
            rhodz_now=rhodz_now,
            dtime=dtime,
        )

        log.debug("horizontal numerical flux computation - end")

    def _update_unknowns(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
        dtime: ta.wpfloat,
    ) -> None:
        log.debug("horizontal unknowns update - start")

        # update tracer mass fraction
        log.debug("running stencil integrate_tracer_horizontally - start")
        self._integrate_tracer_horizontally(
            p_mflx_tracer_h=p_mflx_tracer_h,
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            tracer_new_hor=p_tracer_new,
            p_dtime=dtime,
        )
        log.debug("running stencil integrate_tracer_horizontally - end")

        log.debug("horizontal unknowns update - end")
