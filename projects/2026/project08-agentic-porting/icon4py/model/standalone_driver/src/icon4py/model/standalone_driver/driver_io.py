# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Driver-side glue between the model state and the ``icon4py.model.common.io`` module.

- assembles the prognostic model state into the ``dict[str, xarray.DataArray]`` consumed
  by ``IOMonitor.store`` (:func:`prognostic_state_to_dataarrays`),
- computes the standard diagnostic output fields (u, v, temperature, virtual temperature,
  pressure) from the prognostic state (:class:`DiagnosticsComputer`) and assembles them
  (:func:`diagnostic_fields_to_dataarrays`),
- builds an ``IOMonitor`` that writes all requested fields to one file
  (:func:`create_io_monitor`).
"""

import datetime
import pathlib
import uuid
from typing import Any, Final

import gt4py.next as gtx
import xarray as xr

from icon4py.model.common import dimension as dims, time, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.diagnostic_calculations import pressure as pressure_diagnostics
from icon4py.model.common.diagnostic_calculations.stencils import diagnose_temperature
from icon4py.model.common.grid import base as grid_base, horizontal as h_grid, vertical as v_grid
from icon4py.model.common.interpolation.stencils import edge_2_cell_vector_rbf_interpolation as rbf
from icon4py.model.common.io import io as common_io, utils as io_utils
from icon4py.model.common.states import data as state_data, prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc


#: File-name stub for the output file (a counter + ``.nc`` is appended).
DEFAULT_OUTPUT_FILENAME: Final[str] = "icon4py_output"


# --------------------------------------------------------------------------------------
# Prognostic fields
# --------------------------------------------------------------------------------------


#: Default prognostic output variables, selected by CF name from the
#: ``states.data.PROGNOSTIC_CF_ATTRIBUTES`` catalog (which also holds fields the driver does
#: not output, e.g. ``tangential_velocity``). The metadata, the state attribute
#: (``icon_var_name``) and the vertical placement (``is_on_half_levels``) all come from that
#: catalog; this list only selects which entries to emit.
PROGNOSTIC_VARIABLES: Final[list[str]] = [
    "air_density",
    "exner_function",
    "virtual_potential_temperature",
    "upward_air_velocity",
    "normal_velocity",
]


def prognostic_state_to_dataarrays(
    prognostic_state: prognostics.PrognosticState,
    variables: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """Assemble a CF/UGRID-annotated model-state dict from a ``PrognosticState``."""
    selected = PROGNOSTIC_VARIABLES if variables is None else variables

    state: dict[str, xr.DataArray] = {}
    for name in selected:
        try:
            metadata = state_data.PROGNOSTIC_CF_ATTRIBUTES[name]
        except KeyError as err:
            raise ValueError(
                f"Unknown prognostic output variable '{name}'. "
                f"Known variables are: {PROGNOSTIC_VARIABLES}."
            ) from err
        field = getattr(prognostic_state, metadata["icon_var_name"])
        state[name] = io_utils.to_data_array(
            field,
            metadata,
            is_on_half_levels=metadata.get("is_on_half_levels", False),
            to_host=True,
        )
    return state


# --------------------------------------------------------------------------------------
# Diagnostic fields
# --------------------------------------------------------------------------------------


DIAGNOSTIC_VARIABLES: Final[list[str]] = [
    "eastward_wind",
    "northward_wind",
    "temperature",
    "virtual_temperature",
    "pressure",
]

#: All output variables (prognostic + diagnostic), written together into the same file.
DEFAULT_OUTPUT_VARIABLES: Final[list[str]] = [*PROGNOSTIC_VARIABLES, *DIAGNOSTIC_VARIABLES]


class DiagnosticsComputer:
    """Computes the diagnostic output fields from the prognostic state.

    The ~14 scratch/output buffers are allocated **once** and reused on every
    :meth:`compute` call, avoiding per-step allocation (significant on GPU backends,
    since output is written every step). The static fields (``ddqz_z_full`` and the cell
    rbf coefficients) are passed to :meth:`compute` so callers fetch them from their field
    factories; tests can pass artificial fields.

    TODO(kotsaloscv): refactor once the driver groups model state -- derive these diagnostic
    buffers from that shared grouping instead of recomputing them here. Physics relies
    heavily on the same diagnostic variables, so they should be shared rather than
    duplicated across IO and physics.
    """

    def __init__(self, *, grid: grid_base.Grid, backend: gtx.typing.Backend | None) -> None:
        self._grid = grid
        self._backend = backend
        self._num_levels = grid.num_levels
        cell_domain = h_grid.domain(dims.CellDim)
        self._end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))
        self._cell_lateral_boundary_level_2 = grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )

        def _zero_full() -> gtx.Field:
            return data_alloc.zero_field(
                grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
            )

        def _zero_interface() -> gtx.Field:
            return data_alloc.zero_field(
                grid,
                dims.CellDim,
                dims.KDim,
                dtype=ta.wpfloat,
                extend={dims.KDim: 1},
                allocator=backend,
            )

        # dry air: all hydrometeors stay zero (never written, so allocated once)
        self._qv, self._qc, self._qi, self._qr, self._qs, self._qg = (
            _zero_full() for _ in range(6)
        )
        self._temperature = _zero_full()
        self._virtual_temperature = _zero_full()
        self._u = _zero_full()
        self._v = _zero_full()
        self._pressure = _zero_full()
        # Typed as Any: gt4py's NDArrayObject protocol does not expose __setitem__, so the
        # in-place buffer fills below would not type-check against the precise Field type.
        self._pressure_on_cells_half_levels: Any = _zero_interface()
        self._surface_pressure: Any = data_alloc.zero_field(
            grid, dims.CellDim, dtype=ta.wpfloat, allocator=backend
        )

    def compute(
        self,
        prognostic_state: prognostics.PrognosticState,
        *,
        ddqz_z_full: gtx.Field,
        rbf_vec_coeff_c1: gtx.Field,
        rbf_vec_coeff_c2: gtx.Field,
    ) -> dict[str, gtx.Field]:
        """Diagnose temperature/virtual temperature from ``theta_v``/``exner``, the cell
        winds by RBF interpolation of ``vn``, and pressure by vertical integration from the
        diagnosed surface pressure (dry-air path). Buffers are overwritten in place.

        Returns:
            ``{eastward_wind, northward_wind, temperature, virtual_temperature, pressure}``.
        """
        backend = self._backend
        num_levels = self._num_levels
        end_cell_end = self._end_cell_end

        diagnose_temperature.diagnose_virtual_temperature_and_temperature.with_backend(backend)(
            qv=self._qv,
            qc=self._qc,
            qi=self._qi,
            qr=self._qr,
            qs=self._qs,
            qg=self._qg,
            theta_v=prognostic_state.theta_v,
            exner=prognostic_state.exner,
            virtual_temperature=self._virtual_temperature,
            temperature=self._temperature,
            horizontal_start=0,
            horizontal_end=end_cell_end,
            vertical_start=0,
            vertical_end=num_levels,
            offset_provider={},
        )

        rbf.edge_2_cell_vector_rbf_interpolation.with_backend(backend)(
            p_e_in=prognostic_state.vn,
            ptr_coeff_1=rbf_vec_coeff_c1,
            ptr_coeff_2=rbf_vec_coeff_c2,
            p_u_out=self._u,
            p_v_out=self._v,
            horizontal_start=self._cell_lateral_boundary_level_2,
            horizontal_end=end_cell_end,
            vertical_start=0,
            vertical_end=num_levels,
            offset_provider={"C2E2C2E": self._grid.get_connectivity("C2E2C2E")},
        )

        pressure_diagnostics.diagnose_pressure_surface_to_top(
            grid=self._grid,
            backend=backend,
            exner=prognostic_state.exner,
            virtual_temperature=self._virtual_temperature,
            ddqz_z_full=ddqz_z_full,
            surface_pressure=self._surface_pressure,
            pressure=self._pressure,
            pressure_on_cells_half_levels=self._pressure_on_cells_half_levels,
        )

        return {
            "eastward_wind": self._u,
            "northward_wind": self._v,
            "temperature": self._temperature,
            "virtual_temperature": self._virtual_temperature,
            "pressure": self._pressure,
        }


def diagnostic_fields_to_dataarrays(
    diagnostic_fields: dict[str, gtx.Field],
) -> dict[str, xr.DataArray]:
    """Assemble CF/UGRID-annotated DataArrays from computed diagnostic fields."""
    state: dict[str, xr.DataArray] = {}
    for name, field in diagnostic_fields.items():
        metadata = state_data.DIAGNOSTIC_CF_ATTRIBUTES[name]
        state[name] = io_utils.to_data_array(
            field,
            metadata,
            is_on_half_levels=metadata.get("is_on_half_levels", False),
            to_host=True,
        )
    return state


# --------------------------------------------------------------------------------------
# Monitor factory
# --------------------------------------------------------------------------------------


def create_io_monitor(
    *,
    output_path: pathlib.Path,
    grid_file_path: pathlib.Path,
    grid: grid_base.Grid,
    vertical_grid: v_grid.VerticalGrid,
    dtime: datetime.timedelta,
    variables: list[str] | None = None,
    output_interval: common_io.OutputInterval = time.NumTimeSteps(1),
    process_props: decomposition_defs.ProcessProperties | None = None,
) -> common_io.IOMonitor:
    """Build a single-node ``IOMonitor`` with one field group holding all output fields.

    ``output_interval`` is either a number of model steps or a simulation-time delta
    (normalized to steps using ``dtime``); it defaults to every step.

    ``process_props`` is currently unused: IO is single-node only. It is kept on the
    signature so the distributed path (per-rank IO setup) can be wired in without a
    signature change.
    """
    del process_props  # reserved for the distributed IO path; unused while single-node
    output_variables = DEFAULT_OUTPUT_VARIABLES if variables is None else variables

    field_groups = [
        common_io.FieldGroupIOConfig(
            output_interval=output_interval,
            filename=DEFAULT_OUTPUT_FILENAME,
            variables=output_variables,
            nc_title="ICON4Py output",
            nc_comment="Fields computed by ICON4Py.",
        )
    ]

    config = common_io.IOConfig(output_path=str(output_path), field_groups=field_groups)
    return common_io.IOMonitor(
        config=config,
        vertical_size=vertical_grid,
        horizontal_size=grid.config.horizontal_config,
        grid_file_name=grid_file_path,
        # Grid.id holds the file's `uuidOfHGrid` as a string; the IO layer wants a UUID.
        grid_id=uuid.UUID(grid.id),
        dtime=dtime,
    )
