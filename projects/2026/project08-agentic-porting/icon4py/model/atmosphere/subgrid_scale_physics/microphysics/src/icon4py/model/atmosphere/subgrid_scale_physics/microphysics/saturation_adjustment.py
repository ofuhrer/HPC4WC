# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Final

import icon4py.model.common.dimension as dims
import icon4py.model.common.utils as common_utils
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics.stencils import (
    saturation_adjustment_stencils as satad_stencils,
)
from icon4py.model.common import field_type_aliases as fa, model_options, type_alias as ta
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid, vertical as v_grid
    from icon4py.model.common.states import model


@dataclasses.dataclass(frozen=True)
class SaturationAdjustmentConfig:
    #: in ICON, 10 is always used for max iteration when subroutine satad_v_3D is called.
    max_iter: int = 10
    #: in ICON, 1.e-3 is always used for the tolerance when subroutine satad_v_3D is called.
    tolerance: ta.wpfloat = 1.0e-3


@dataclasses.dataclass
class MetricStateSaturationAdjustment:
    ddqz_z_full: fa.CellKField[ta.wpfloat]


class ConvergenceError(Exception):
    pass


#: CF attributes of saturation adjustment input variables
_SATURATION_ADJUST_INPUT_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    air_density=dict(
        standard_name="air_density",
        long_name="density",
        units="kg m-3",
        icon_var_name="rho",
    ),
    temperature=dict(
        standard_name="air_temperature",
        long_name="air temperature",
        units="K",
        icon_var_name="temp",
    ),
    specific_humidity=dict(
        standard_name="specific_humidity",
        long_name="ratio of water vapor mass to total moist air parcel mass",
        units="1",
        icon_var_name="qv",
    ),
    specific_cloud=dict(
        standard_name="specific_cloud_content",
        long_name="ratio of cloud water mass to total moist air parcel mass",
        units="1",
        icon_var_name="qc",
    ),
)


#: CF attributes of saturation adjustment output variables
_SATURATION_ADJUST_OUTPUT_ATTRIBUTES: Final[dict[str, model.FieldMetaData]] = dict(
    tend_temperature_due_to_satad=dict(
        standard_name="tendency_of_air_temperature_due_to_saturation_adjustment",
        long_name="tendency of air temperature due to saturation adjustment",
        units="K s-1",
    ),
    tend_specific_humidity_due_to_satad=dict(
        standard_name="tendency_of_specific_humidity_due_to_saturation_adjustment",
        long_name="tendency of ratio of water vapor mass to total moist air parcel mass due to saturation adjustment",
        units="s-1",
    ),
    tend_specific_cloud_due_to_satad=dict(
        standard_name="tendency_of_specific_cloud_content_due_to_saturation_adjustment",
        long_name="tendency of ratio of cloud water mass to total moist air parcel mass due to saturation adjustment",
        units="s-1",
    ),
)


class SaturationAdjustment:
    def __init__(
        self,
        *,
        config: SaturationAdjustmentConfig,
        grid: icon_grid.IconGrid,
        vertical_params: v_grid.VerticalGrid,
        metric_state: MetricStateSaturationAdjustment,
        backend: gtx_typing.Backend | None,
    ):
        self._backend = backend
        self.config = config
        self._grid = grid
        self._vertical_params: v_grid.VerticalGrid = vertical_params
        self._metric_state: MetricStateSaturationAdjustment = metric_state
        self._xp = data_alloc.import_array_ns(self._backend)

        self._allocate_local_variables()
        self._determine_horizontal_domains()
        self._initialize_gt4py_programs()

    # TODO(OngChia): add in input and output data properties, and refactor this component to follow the physics component protocol.
    def input_properties(self) -> dict[str, model.FieldMetaData]:
        raise NotImplementedError

    def output_properties(self) -> dict[str, model.FieldMetaData]:
        raise NotImplementedError

    def _allocate_local_variables(self):
        #: it was originally named as tworkold in ICON. Old temperature before iteration.
        self._temperature1 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        #: it was originally named as twork in ICON. New temperature before iteration.
        self._temperature2 = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        #: A mask that indicates whether the grid cell is subsaturated or not.
        self._subsaturated_mask = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=bool, allocator=self._backend
        )
        #: A mask that indicates whether next Newton iteration is required.
        self._newton_iteration_mask = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=bool, allocator=self._backend
        )
        #: latent heat vaporization / dry air heat capacity at constant volume
        self._lwdocvd = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )

    def _initialize_gt4py_programs(self):
        self._compute_subsaturated_case_and_initialize_newton_iterations = (
            model_options.setup_program(
                backend=self._backend,
                program=satad_stencils.compute_subsaturated_case_and_initialize_newton_iterations,
                constant_args={
                    "tolerance": self.config.tolerance,
                },
                horizontal_sizes={
                    "horizontal_start": self._start_cell_nudging,
                    "horizontal_end": self._end_cell_local,
                },
                vertical_sizes={
                    "vertical_start": self._vertical_params.kstart_moist,
                    "vertical_end": self._grid.num_levels,
                },
            )
        )
        self._update_temperature_by_newton_iteration = model_options.setup_program(
            backend=self._backend,
            program=satad_stencils.update_temperature_by_newton_iteration,
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": self._vertical_params.kstart_moist,
                "vertical_end": self._grid.num_levels,
            },
        )
        self._compute_newton_iteration_mask_and_copy_temperature_on_converged_cells = model_options.setup_program(
            backend=self._backend,
            program=satad_stencils.compute_newton_iteration_mask_and_copy_temperature_on_converged_cells,
            constant_args={
                "tolerance": self.config.tolerance,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": self._vertical_params.kstart_moist,
                "vertical_end": self._grid.num_levels,
            },
        )
        self._update_temperature_qv_qc_tendencies = model_options.setup_program(
            backend=self._backend,
            program=satad_stencils.update_temperature_qv_qc_tendencies,
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": self._vertical_params.kstart_moist,
                "vertical_end": self._grid.num_levels,
            },
        )

    def _determine_horizontal_domains(self):
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    def _not_converged(self) -> bool:
        return self._xp.any(
            self._newton_iteration_mask.ndarray[
                self._start_cell_nudging : self._end_cell_local,
                self._vertical_params.kstart_moist : self._grid.num_levels,
            ]
        )

    def run(
        self,
        *,
        dtime: ta.wpfloat,
        rho: fa.CellKField[ta.wpfloat],
        temperature: fa.CellKField[ta.wpfloat],
        qv: fa.CellKField[ta.wpfloat],
        qc: fa.CellKField[ta.wpfloat],
        temperature_tendency: fa.CellKField[ta.wpfloat],
        qv_tendency: fa.CellKField[ta.wpfloat],
        qc_tendency: fa.CellKField[ta.wpfloat],
    ):
        """
        Adjust saturation at each grid point.
        Saturation adjustment condenses/evaporates specific humidity (qv) into/from
        cloud water content (qc) such that a gridpoint is just saturated. Temperature (t)
        is adapted accordingly and pressure adapts itself in ICON.

        Saturation adjustment at constant total density (adjustment of T and p accordingly)
        assuming chemical equilibrium of water and vapor. For the heat capacity of
        of the total system (dry air, vapor, and hydrometeors) the value of dry air
        is taken, which is a common approximation and introduces only a small error.

        Originally inspired from satad_v_3D of ICON.

        Args:
            dtime: time step [s]
            rho: air density [kg m-3]
            temperature: air temperature [K]
            qv: specific humidity [kg kg-1]
            qc: specific cloud water content [kg kg-1]
            temperature_tendency: air temperature tendency [K s-1]
            qv_tendency: specific humidity tendency [s-1]
            qc_tendency: specific cloud water content tendency [s-1]
        """

        temperature_pair = common_utils.TimeStepPair(self._temperature1, self._temperature2)

        self._compute_subsaturated_case_and_initialize_newton_iterations(
            temperature=temperature,
            qv=qv,
            qc=qc,
            rho=rho,
            subsaturated_mask=self._subsaturated_mask,
            lwdocvd=self._lwdocvd,
            current_temperature=temperature_pair.current,
            next_temperature=temperature_pair.next,
            newton_iteration_mask=self._newton_iteration_mask,
        )

        # TODO(OngChia): this is inspired by the cpu version of the original ICON saturation_adjustment code. Consider to refactor this code when break and for loop features are ready in gt4py.
        num_iter = 0
        while self._not_converged():
            if num_iter > self.config.max_iter:
                raise ConvergenceError(
                    f"Maximum iteration of saturation adjustment ({self.config.max_iter}) is not enough. The max absolute error is {self._xp.abs(self._temperature1.ndarray - self._temperature2.ndarray).max()} . Please raise max_iter"
                )

            self._update_temperature_by_newton_iteration(
                temperature=temperature,
                qv=qv,
                rho=rho,
                newton_iteration_mask=self._newton_iteration_mask,
                lwdocvd=self._lwdocvd,
                next_temperature=temperature_pair.next,
                current_temperature=temperature_pair.current,
            )

            self._compute_newton_iteration_mask_and_copy_temperature_on_converged_cells(
                current_temperature=temperature_pair.current,
                next_temperature=temperature_pair.next,
                newton_iteration_mask=self._newton_iteration_mask,
            )

            temperature_pair.swap()
            num_iter = num_iter + 1

        self._update_temperature_qv_qc_tendencies(
            dtime=dtime,
            temperature=temperature,
            current_temperature=temperature_pair.current,
            qv=qv,
            qc=qc,
            rho=rho,
            subsaturated_mask=self._subsaturated_mask,
            temperature_tendency=temperature_tendency,
            qv_tendency=qv_tendency,
            qc_tendency=qc_tendency,
        )
