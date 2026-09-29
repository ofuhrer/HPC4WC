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
import pathlib
from typing import TYPE_CHECKING

import serialbox  # type: ignore[import-untyped]

from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


log = logging.getLogger(__name__)


@dataclasses.dataclass
class FromFileConfig:
    """Parameters for the file-based initial condition."""

    #: Path to the serialised data directory (typically ``<experiment>/ser_data``).
    data_path: pathlib.Path
    #: Number of tracer species stored in the snapshot (0 means no tracers).
    ntracer: int = 0


def _fill_prognostics_from_serialbox(
    *,
    data_path: pathlib.Path,
    rank: int,
    grid: icon_grid.IconGrid,
    backend: gtx_typing.Backend | None,
    prognostic_state: prognostics.PrognosticState,
    ntracer: int,
) -> None:
    """Fill a pre-allocated PrognosticState from a serialbox snapshot."""
    allocator = model_backends.get_allocator(backend)
    array_ns = data_alloc.import_array_ns(allocator)

    fname = f"icon_pydycore_rank{rank}"
    ser = serialbox.Serializer(serialbox.OpenModeKind.Read, str(data_path), fname)
    sp = ser.savepoint["prognostics"].id[1].location["initial-state"].as_savepoint()
    log.debug("Reading prognostics initial-state from %s / %s", data_path, fname)

    num_cells = grid.num_cells
    num_edges = grid.num_edges

    def read_cell_k(name: str) -> data_alloc.NDArray:
        return array_ns.asarray(array_ns.squeeze(ser.read(name, sp).astype(float))[:num_cells, :])

    def read_edge_k(name: str) -> data_alloc.NDArray:
        return array_ns.asarray(array_ns.squeeze(ser.read(name, sp).astype(float))[:num_edges, :])

    prognostic_state.rho.ndarray[:, :] = read_cell_k("rho_now")
    prognostic_state.exner.ndarray[:, :] = read_cell_k("exner_now")
    prognostic_state.theta_v.ndarray[:, :] = read_cell_k("theta_v_now")
    prognostic_state.vn.ndarray[:, :] = read_edge_k("vn_now")
    prognostic_state.w.ndarray[:, :] = read_cell_k("w_now")

    if ntracer > 0:
        tracers_raw = array_ns.squeeze(ser.read("tracers_now", sp).astype(float))
        for i, tracer in enumerate(prognostic_state.tracer.active_fields()):
            tracer.field.ndarray[:, :] = array_ns.asarray(tracers_raw[:num_cells, :, i])


def read_from_file(
    *,
    config: FromFileConfig,
    grid: icon_grid.IconGrid,
    prognostic_state_now: prognostics.PrognosticState,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
) -> None:
    """Initialise prognostic state from a serialised ICON initial-condition snapshot."""
    _fill_prognostics_from_serialbox(
        data_path=config.data_path,
        rank=exchange.my_rank(),
        grid=grid,
        backend=backend,
        prognostic_state=prognostic_state_now,
        ntracer=config.ntracer,
    )
