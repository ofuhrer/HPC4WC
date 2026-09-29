# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import abc
import logging

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

from icon4py.model.atmosphere.advection import advection_states
from icon4py.model.atmosphere.advection.stencils.compute_ppm4gpu_courant_number import (
    compute_ppm4gpu_courant_number,
)
from icon4py.model.atmosphere.advection.stencils.compute_ppm4gpu_fractional_flux import (
    compute_ppm4gpu_fractional_flux,
)
from icon4py.model.atmosphere.advection.stencils.compute_ppm4gpu_integer_flux import (
    compute_ppm4gpu_integer_flux,
)
from icon4py.model.atmosphere.advection.stencils.compute_ppm4gpu_parabola_coefficients import (
    compute_ppm4gpu_parabola_coefficients,
)
from icon4py.model.atmosphere.advection.stencils.compute_ppm_quadratic_face_values import (
    compute_ppm_quadratic_face_values,
)
from icon4py.model.atmosphere.advection.stencils.compute_ppm_quartic_face_values import (
    compute_ppm_quartic_face_values,
)
from icon4py.model.atmosphere.advection.stencils.compute_ppm_slope import compute_ppm_slope
from icon4py.model.atmosphere.advection.stencils.compute_vertical_parabola_limiter_condition import (
    compute_vertical_parabola_limiter_condition,
)
from icon4py.model.atmosphere.advection.stencils.compute_vertical_tracer_flux_upwind import (
    compute_vertical_tracer_flux_upwind,
)
from icon4py.model.atmosphere.advection.stencils.copy_cell_kdim_field import copy_cell_kdim_field
from icon4py.model.atmosphere.advection.stencils.copy_cell_kdim_field_koff_minus1 import (
    copy_cell_kdim_field_koff_minus1,
)
from icon4py.model.atmosphere.advection.stencils.copy_cell_kdim_field_koff_plus1 import (
    copy_cell_kdim_field_koff_plus1,
)
from icon4py.model.atmosphere.advection.stencils.init_constant_cell_kdim_field import (
    init_constant_cell_kdim_field,
)
from icon4py.model.atmosphere.advection.stencils.integrate_tracer_vertically import (
    integrate_tracer_vertically,
)
from icon4py.model.atmosphere.advection.stencils.limit_vertical_parabola_semi_monotonically import (
    limit_vertical_parabola_semi_monotonically,
)
from icon4py.model.atmosphere.advection.stencils.limit_vertical_slope_semi_monotonically import (
    limit_vertical_slope_semi_monotonically,
)
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.utils import data_allocation as data_alloc


"""
Advection components related to vertical transport.
"""

log = logging.getLogger(__name__)


class BoundaryConditions(abc.ABC):
    """Class that sets the upper and lower boundary conditions."""

    @abc.abstractmethod
    def run(
        self,
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None:
        """
        Set the vertical boundary conditions.

        Args:
            p_mflx_tracer_v: output argument, field that contains new vertical tracer mass flux
            horizontal_start: input argument, horizontal start index
            horizontal_end: input argument, horizontal end index

        """
        ...


class NoFluxCondition(BoundaryConditions):
    """Class that sets the upper and lower boundary fluxes to zero."""

    def __init__(self, grid: icon_grid.IconGrid, backend: gtx_typing.Backend | None):
        # input arguments
        self._grid = grid
        self._backend = backend

        # stencils
        self._init_constant_cell_kdim_field = init_constant_cell_kdim_field.with_backend(
            self._backend
        )

    def run(
        self,
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None:
        log.debug("vertical boundary conditions computation - start")

        # set upper boundary conditions
        log.debug("running stencil init_constant_cell_kdim_field - start")
        self._init_constant_cell_kdim_field(
            field=p_mflx_tracer_v,
            value=0.0,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=1,
            offset_provider=self._grid.connectivities,
        )
        log.debug("running stencil init_constant_cell_kdim_field - end")

        # set lower boundary conditions
        log.debug("running stencil init_constant_cell_kdim_field - start")
        self._init_constant_cell_kdim_field(
            field=p_mflx_tracer_v,
            value=0.0,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=self._grid.num_levels,
            vertical_end=self._grid.num_levels + 1,
            offset_provider=self._grid.connectivities,
        )
        log.debug("running stencil init_constant_cell_kdim_field - end")

        log.debug("vertical boundary conditions computation - end")


class VerticalLimiter(abc.ABC):
    """Class that limits the vertical reconstructed fields and the fluxes."""

    @abc.abstractmethod
    def limit_slope(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        z_slope: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None: ...

    @abc.abstractmethod
    def limit_parabola(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_face: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        p_face_up: fa.CellKField[ta.wpfloat],
        p_face_low: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None: ...

    @abc.abstractmethod
    def limit_fluxes(
        self,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None: ...


class NoLimiter(VerticalLimiter):
    """Class that implements no vertical parabola limiter."""

    def __init__(self, grid: icon_grid.IconGrid, backend: gtx_typing.Backend | None):
        # input arguments
        self._grid = grid
        self._backend = backend

        # fields
        self._l_limit = data_alloc.zero_field(
            self._grid,
            dims.CellDim,
            dims.KDim,
            allocator=model_backends.get_allocator(self._backend),
        )

        # stencils
        self._copy_cell_kdim_field = model_options.setup_program(
            backend=backend,
            program=copy_cell_kdim_field,
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._copy_cell_kdim_field_koff_plus1 = model_options.setup_program(
            backend=backend,
            program=copy_cell_kdim_field_koff_plus1,
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

    def limit_slope(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        z_slope: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None: ...

    def limit_parabola(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_face: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        p_face_up: fa.CellKField[ta.wpfloat],
        p_face_low: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None:
        # simply copy to up/low face values
        log.debug("running stencil copy_cell_kdim_field - start")
        self._copy_cell_kdim_field(
            field_in=p_face,
            field_out=p_face_up,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil copy_cell_kdim_field - end")

        log.debug("running stencil copy_cell_kdim_field_koff_plus1 - start")
        self._copy_cell_kdim_field_koff_plus1(
            field_in=p_face,
            field_out=p_face_low,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil copy_cell_kdim_field_koff_plus1 - end")

    def limit_fluxes(
        self,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None: ...


class SemiMonotonicLimiter(VerticalLimiter):
    """Class that implements a semi-monotonic vertical parabola limiter."""

    def __init__(self, grid: icon_grid.IconGrid, backend: gtx_typing.Backend | None):
        # input arguments
        self._grid = grid
        self._backend = backend

        # fields
        allocator = model_backends.get_allocator(self._backend)
        self._k_field = data_alloc.index_field(
            self._grid, dims.KDim, extend={dims.KDim: 1}, dtype=gtx.int32, allocator=allocator
        )  # TODO(dastrm): should be KHalfDim
        self._l_limit = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=gtx.int32, allocator=allocator
        )

        # stencils
        self._limit_vertical_slope_semi_monotonically = model_options.setup_program(
            backend=backend,
            program=limit_vertical_slope_semi_monotonically,
            constant_args={
                "elev": self._grid.num_levels - 1,
                "k": self._k_field,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._compute_vertical_parabola_limiter_condition = model_options.setup_program(
            backend=backend,
            program=compute_vertical_parabola_limiter_condition,
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._limit_vertical_parabola_semi_monotonically = model_options.setup_program(
            backend=backend,
            program=limit_vertical_parabola_semi_monotonically,
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

    def limit_slope(
        self,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        z_slope: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None:
        log.debug("running stencil limit_vertical_slope_semi_monotonically - start")
        self._limit_vertical_slope_semi_monotonically(
            p_cc=p_tracer_now,
            z_slope=z_slope,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil limit_vertical_slope_semi_monotonically - end")

    def limit_parabola(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_face: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        p_face_up: fa.CellKField[ta.wpfloat],
        p_face_low: fa.CellKField[ta.wpfloat],
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None:
        # compute semi-monotonic limiter condition
        log.debug("running stencil compute_vertical_parabola_limiter_condition - start")
        self._compute_vertical_parabola_limiter_condition(
            p_face=p_face,
            p_cc=p_tracer_now,
            l_limit=self._l_limit,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil compute_vertical_parabola_limiter_condition - end")

        # apply semi-monotonic limiter condition and store to up/low face values
        log.debug("running stencil limit_vertical_parabola_semi_monotonically - start")
        self._limit_vertical_parabola_semi_monotonically(
            l_limit=self._l_limit,
            p_face=p_face,
            p_cc=p_tracer_now,
            p_face_up=p_face_up,
            p_face_low=p_face_low,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil limit_vertical_parabola_semi_monotonically - end")

    def limit_fluxes(
        self,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
    ) -> None: ...


class VerticalAdvection(abc.ABC):
    """Class that does one vertical advection step."""

    @abc.abstractmethod
    def run(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool = False,
    ) -> None:
        """
        Run a vertical advection step.

        Args:
            prep_adv: input argument, data class that contains precalculated advection fields
            p_tracer_now: input argument, field that contains current tracer mass fraction
            p_tracer_new: output argument, field that contains new tracer mass fraction
            rhodz_now: input argument, field that contains current air mass in each layer
            rhodz_new: input argument, field that contains new air mass in each layer
            p_mflx_tracer_v: output argument, field that contains new vertical tracer mass flux
            dtime: input argument, the time step
            even_timestep: input argument, determines whether halo points are included

        Note:
            Originally, if even step, vertical transport includes all halo points in order to avoid an additional synchronization step, i.e.
                if lstep_even: i_rlstart  = 2,                i_rlend = min_rlcell
                else:          i_rlstart  = grf_bdywidth_c+1, i_rlend = min_rlcell_int
            Horizontal advection is always called with the same indices though, i.e.
                i_rlstart = grf_bdywidth_c+1, i_rlend = min_rlcell_int
        """
        ...


class NoAdvection(VerticalAdvection):
    """Class that implements disabled vertical advection."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        backend: gtx_typing.Backend | None,
    ):
        log.debug("vertical advection class init - start")

        # input arguments
        self._grid = grid
        self._backend = backend

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        # stencils
        self._copy_cell_kdim_field = model_options.setup_program(
            backend=backend,
            program=copy_cell_kdim_field,
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        log.debug("vertical advection class init - end")

    def _get_horizontal_start_end(self, even_timestep: bool) -> tuple[gtx.int32, gtx.int32]:
        if even_timestep:
            horizontal_start = self._start_cell_lateral_boundary_level_2
            horizontal_end = self._end_cell_end
        else:
            horizontal_start = self._start_cell_nudging
            horizontal_end = self._end_cell_local

        return horizontal_start, horizontal_end

    def run(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool = False,
    ) -> None:
        log.debug("vertical advection run - start")

        horizontal_start, horizontal_end = self._get_horizontal_start_end(
            even_timestep=even_timestep
        )

        log.debug("running stencil copy_cell_kdim_field - start")
        self._copy_cell_kdim_field(
            field_in=p_tracer_now,
            field_out=p_tracer_new,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil copy_cell_kdim_field - end")
        log.debug("vertical advection run - end")


class FiniteVolume(VerticalAdvection):
    """Class that defines a finite volume vertical advection scheme."""

    def run(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool = False,
    ) -> None:
        log.debug("vertical advection run - start")

        self._compute_numerical_flux(
            prep_adv=prep_adv,
            p_tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_v=p_mflx_tracer_v,
            dtime=dtime,
            even_timestep=even_timestep,
        )

        self._update_unknowns(
            p_tracer_now=p_tracer_now,
            p_tracer_new=p_tracer_new,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            p_mflx_tracer_v=p_mflx_tracer_v,
            dtime=dtime,
            even_timestep=even_timestep,
        )

        log.debug("vertical advection run - end")

    @abc.abstractmethod
    def _compute_numerical_flux(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool,
    ) -> None: ...

    @abc.abstractmethod
    def _update_unknowns(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool,
    ) -> None: ...


class FirstOrderUpwind(FiniteVolume):
    """Class that does one vertical first-order accurate upwind finite volume advection step."""

    def __init__(
        self,
        boundary_conditions: BoundaryConditions,
        grid: icon_grid.IconGrid,
        metric_state: advection_states.AdvectionMetricState,
        backend: gtx_typing.Backend | None,
    ):
        log.debug("vertical advection class init - start")

        # input arguments
        self._boundary_conditions = boundary_conditions
        self._grid = grid
        self._metric_state = metric_state
        self._backend = backend

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))
        # misc
        self._ivadv_tracer = 1
        self._iadv_slev_jt = 0

        # fields
        self._k_field = data_alloc.index_field(
            self._grid,
            dims.KDim,
            extend={dims.KDim: 1},
            dtype=gtx.int32,
            allocator=model_backends.get_allocator(self._backend),
        )  # TODO(dastrm): should be KHalfDim

        # stencils
        self._compute_vertical_tracer_flux_upwind = model_options.setup_program(
            backend=backend,
            program=compute_vertical_tracer_flux_upwind,
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._init_constant_cell_kdim_field = init_constant_cell_kdim_field.with_backend(
            self._backend
        )
        self._integrate_tracer_vertically = model_options.setup_program(
            backend=backend,
            program=integrate_tracer_vertically,
            constant_args={
                "deepatmo_divzl": self._metric_state.deepatmo_divzl,
                "deepatmo_divzu": self._metric_state.deepatmo_divzu,
                "k": self._k_field,
                "ivadv_tracer": self._ivadv_tracer,
                "iadv_slev_jt": self._iadv_slev_jt,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        log.debug("vertical advection class init - end")

    def _get_horizontal_start_end(self, even_timestep: bool) -> tuple[gtx.int32, gtx.int32]:
        if even_timestep:
            horizontal_start = self._start_cell_lateral_boundary_level_2
            horizontal_end = self._end_cell_end
        else:
            horizontal_start = self._start_cell_nudging
            horizontal_end = self._end_cell_local

        return horizontal_start, horizontal_end

    def _compute_numerical_flux(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool,
    ) -> None:
        log.debug("vertical numerical flux computation - start")

        horizontal_start, horizontal_end = self._get_horizontal_start_end(
            even_timestep=even_timestep
        )
        log.debug("running stencil compute_vertical_tracer_flux_upwind - start")
        self._compute_vertical_tracer_flux_upwind(
            p_cc=p_tracer_now,
            p_mflx_contra_v=prep_adv.mass_flx_ic,
            p_upflux=p_mflx_tracer_v,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil compute_vertical_tracer_flux_upwind - end")

        # set boundary conditions
        self._boundary_conditions.run(
            p_mflx_tracer_v=p_mflx_tracer_v,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )

        log.debug("vertical numerical flux computation - end")

    def _update_unknowns(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool,
    ) -> None:
        log.debug("vertical unknowns update - start")

        horizontal_start, horizontal_end = self._get_horizontal_start_end(
            even_timestep=even_timestep
        )

        # update tracer mass fraction
        log.debug("running stencil integrate_tracer_vertically - start")
        self._integrate_tracer_vertically(
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_v=p_mflx_tracer_v,
            rhodz_new=rhodz_new,
            tracer_new=p_tracer_new,
            p_dtime=dtime,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil integrate_tracer_vertically - end")
        log.debug("vertical unknowns update - end")


class PiecewiseParabolicMethod(FiniteVolume):
    """Class that does one vertical PPM finite volume advection step."""

    def __init__(
        self,
        *,
        boundary_conditions: BoundaryConditions,
        vertical_limiter: VerticalLimiter,
        grid: icon_grid.IconGrid,
        metric_state: advection_states.AdvectionMetricState,
        backend: gtx_typing.Backend | None,
    ):
        log.debug("vertical advection class init - start")

        # input arguments
        self._boundary_conditions = boundary_conditions
        self._vertical_limiter = vertical_limiter
        self._grid = grid
        self._metric_state = metric_state
        self._backend = backend

        # cell indices
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_lateral_boundary_level_2 = self._grid.start_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))
        self._end_cell_end = self._grid.end_index(cell_domain(h_grid.Zone.END))

        # fields
        allocator = model_backends.get_allocator(self._backend)
        self._k_field = data_alloc.index_field(
            self._grid, dims.KDim, extend={dims.KDim: 1}, dtype=gtx.int32, allocator=allocator
        )  # TODO(dastrm): should be KHalfDim
        self._z_cfl = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        )  # TODO(dastrm): should be KHalfDim
        self._z_slope = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._z_face = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=allocator
        )  # TODO(dastrm): should be KHalfDim
        self._z_face_up = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._z_face_low = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._z_delta_q = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, allocator=allocator
        )
        self._z_a1 = data_alloc.zero_field(self._grid, dims.CellDim, dims.KDim, allocator=allocator)

        # misc
        self._slev = 0
        self._slevp1_ti = 1
        self._elev = self._grid.num_levels - 1
        self._nlev = self._grid.num_levels - 1
        self._ivadv_tracer = 1
        self._iadv_slev_jt = 0

        # stencils
        self._init_constant_cell_kdim_field = model_options.setup_program(
            backend=self._backend,
            program=init_constant_cell_kdim_field,
            constant_args={
                "value": 0.0,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )

        self._compute_ppm4gpu_courant_number = model_options.setup_program(
            backend=self._backend,
            program=compute_ppm4gpu_courant_number,
            constant_args={
                "k": self._k_field,
                "slevp1_ti": self._slevp1_ti,
                "nlev": self._nlev,
                "dbl_eps": constants.DBL_EPS,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._compute_ppm_slope = model_options.setup_program(
            backend=self._backend,
            program=compute_ppm_slope,
            constant_args={
                "p_cellhgt_mc_now": self._metric_state.ddqz_z_full,
                "elev": self._elev,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._compute_ppm_quadratic_face_values = model_options.setup_program(
            backend=self._backend,
            program=compute_ppm_quadratic_face_values,
            constant_args={
                "p_cellhgt_mc_now": self._metric_state.ddqz_z_full,
            },
            offset_provider=self._grid.connectivities,
        )
        self._compute_ppm_quartic_face_values = model_options.setup_program(
            backend=self._backend,
            program=compute_ppm_quartic_face_values,
            constant_args={
                "p_cellhgt_mc_now": self._metric_state.ddqz_z_full,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(2),
                "vertical_end": gtx.int32(self._grid.num_levels - 1),
            },
            offset_provider=self._grid.connectivities,
        )
        self._copy_cell_kdim_field = model_options.setup_program(
            backend=self._backend,
            program=copy_cell_kdim_field,
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(1),
            },
            offset_provider=self._grid.connectivities,
        )

        self._copy_cell_kdim_field_koff_minus1 = model_options.setup_program(
            backend=self._backend,
            program=copy_cell_kdim_field_koff_minus1,
            vertical_sizes={
                "vertical_start": gtx.int32(self._grid.num_levels),
                "vertical_end": gtx.int32(self._grid.num_levels + 1),
            },
            offset_provider=self._grid.connectivities,
        )

        self._compute_ppm4gpu_parabola_coefficients = model_options.setup_program(
            backend=self._backend,
            program=compute_ppm4gpu_parabola_coefficients,
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._compute_ppm4gpu_fractional_flux = model_options.setup_program(
            backend=self._backend,
            program=compute_ppm4gpu_fractional_flux,
            constant_args={
                "k": self._k_field,
                "slev": self._slev,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._compute_ppm4gpu_integer_flux = model_options.setup_program(
            backend=self._backend,
            program=compute_ppm4gpu_integer_flux,
            constant_args={
                "k": self._k_field,
                "slev": self._slev,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(1),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )
        self._integrate_tracer_vertically = model_options.setup_program(
            backend=self._backend,
            program=integrate_tracer_vertically,
            constant_args={
                "deepatmo_divzl": self._metric_state.deepatmo_divzl,
                "deepatmo_divzu": self._metric_state.deepatmo_divzu,
                "k": self._k_field,
                "ivadv_tracer": self._ivadv_tracer,
                "iadv_slev_jt": self._iadv_slev_jt,
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self._grid.num_levels),
            },
            offset_provider=self._grid.connectivities,
        )

        log.debug("vertical advection class init - end")

    def _get_horizontal_start_end(self, even_timestep: bool) -> tuple[gtx.int32, gtx.int32]:
        if even_timestep:
            horizontal_start = self._start_cell_lateral_boundary_level_2
            horizontal_end = self._end_cell_end
        else:
            horizontal_start = self._start_cell_nudging
            horizontal_end = self._end_cell_local

        return horizontal_start, horizontal_end

    def _compute_numerical_flux(
        self,
        *,
        prep_adv: advection_states.AdvectionPrepAdvState,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool,
    ) -> None:
        log.debug("vertical numerical flux computation - start")

        horizontal_start, horizontal_end = self._get_horizontal_start_end(
            even_timestep=even_timestep
        )

        ## compute density-weighted Courant number

        log.debug("running stencil init_constant_cell_kdim_field - start")
        self._init_constant_cell_kdim_field(
            field=self._z_cfl,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil init_constant_cell_kdim_field - end")

        log.debug("running stencil compute_ppm4gpu_courant_number - start")
        self._compute_ppm4gpu_courant_number(
            p_mflx_contra_v=prep_adv.mass_flx_ic,
            p_cellmass_now=rhodz_now,
            z_cfl=self._z_cfl,
            p_dtime=dtime,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil compute_ppm4gpu_courant_number - end")

        ## reconstruct face values

        # compute slope
        log.debug("running stencil compute_ppm_slope - start")
        self._compute_ppm_slope(
            p_cc=p_tracer_now,
            z_slope=self._z_slope,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil compute_ppm_slope - end")

        # limit slope
        self._vertical_limiter.limit_slope(
            p_tracer_now=p_tracer_now,
            z_slope=self._z_slope,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )

        # compute second highest face value
        log.debug("running stencil compute_ppm_quadratic_face_values - start")
        self._compute_ppm_quadratic_face_values(
            p_cc=p_tracer_now,
            p_face=self._z_face,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=1,
            vertical_end=2,
        )
        log.debug("running stencil compute_ppm_quadratic_face_values - end")

        # compute second lowest face value
        log.debug("running stencil compute_ppm_quadratic_face_values - start")
        self._compute_ppm_quadratic_face_values(
            p_cc=p_tracer_now,
            p_face=self._z_face,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=self._grid.num_levels - 1,
            vertical_end=self._grid.num_levels,
        )
        log.debug("running stencil compute_ppm_quadratic_face_values - end")

        # compute highest face value
        log.debug("running stencil copy_cell_kdim_field - start")
        self._copy_cell_kdim_field(
            field_in=p_tracer_now,
            field_out=self._z_face,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil copy_cell_kdim_field - end")

        # compute lowest face value
        log.debug("running stencil copy_cell_kdim_field_koff_minus1 - start")
        self._copy_cell_kdim_field_koff_minus1(
            field_in=p_tracer_now,
            field_out=self._z_face,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil copy_cell_kdim_field_koff_minus1 - end")

        # compute all other face values
        log.debug("running stencil compute_ppm_quartic_face_values - start")
        self._compute_ppm_quartic_face_values(
            p_cc=p_tracer_now,
            z_slope=self._z_slope,
            p_face=self._z_face,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil compute_ppm_quartic_face_values - end")

        ## limit reconstruction

        self._vertical_limiter.limit_parabola(
            p_tracer_now=p_tracer_now,
            p_face=self._z_face,
            p_face_up=self._z_face_up,
            p_face_low=self._z_face_low,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )

        ## compute fractional numerical flux

        log.debug("running stencil compute_ppm4gpu_parabola_coefficients - start")
        self._compute_ppm4gpu_parabola_coefficients(
            z_face_up=self._z_face_up,
            z_face_low=self._z_face_low,
            p_cc=p_tracer_now,
            z_delta_q=self._z_delta_q,
            z_a1=self._z_a1,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil compute_ppm4gpu_parabola_coefficients - end")

        log.debug("running stencil compute_ppm4gpu_fractional_flux - start")
        self._compute_ppm4gpu_fractional_flux(
            p_cc=p_tracer_now,
            p_cellmass_now=rhodz_now,
            z_cfl=self._z_cfl,
            z_delta_q=self._z_delta_q,
            z_a1=self._z_a1,
            p_upflux=p_mflx_tracer_v,
            p_dtime=dtime,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil compute_ppm4gpu_fractional_flux - end")

        ## compute integer numerical flux

        log.debug("running stencil compute_ppm4gpu_integer_flux - start")
        self._compute_ppm4gpu_integer_flux(
            p_cc=p_tracer_now,
            p_cellmass_now=rhodz_now,
            z_cfl=self._z_cfl,
            p_upflux=p_mflx_tracer_v,
            p_dtime=dtime,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil compute_ppm4gpu_integer_flux - end")

        ## set boundary conditions
        self._boundary_conditions.run(
            p_mflx_tracer_v=p_mflx_tracer_v,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )

        ## apply flux limiter
        self._vertical_limiter.limit_fluxes(
            horizontal_start=horizontal_start, horizontal_end=horizontal_end
        )

        log.debug("vertical numerical flux computation - end")

    def _update_unknowns(
        self,
        *,
        p_tracer_now: fa.CellKField[ta.wpfloat],
        p_tracer_new: fa.CellKField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        p_mflx_tracer_v: fa.CellKField[ta.wpfloat],  # TODO(dastrm): should be KHalfDim
        dtime: ta.wpfloat,
        even_timestep: bool,
    ) -> None:
        log.debug("vertical unknowns update - start")

        horizontal_start, horizontal_end = self._get_horizontal_start_end(
            even_timestep=even_timestep
        )

        # update tracer mass fraction
        log.debug("running stencil integrate_tracer_vertically - start")
        self._integrate_tracer_vertically(
            tracer_now=p_tracer_now,
            rhodz_now=rhodz_now,
            p_mflx_tracer_v=p_mflx_tracer_v,
            rhodz_new=rhodz_new,
            tracer_new=p_tracer_new,
            p_dtime=dtime,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
        )
        log.debug("running stencil integrate_tracer_vertically - end")
        log.debug("vertical unknowns update - end")
