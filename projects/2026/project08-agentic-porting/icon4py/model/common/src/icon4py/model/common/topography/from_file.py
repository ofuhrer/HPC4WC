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
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import grid_manager as gm


log = logging.getLogger(__name__)


@dataclasses.dataclass
class FromFileConfig:
    """Parameters for the file-based topography."""

    #: Path to the serialised data directory (typically ``<experiment>/ser_data``).
    data_path: pathlib.Path


def read_from_file(
    *,
    config: FromFileConfig,
    grid_manager: gm.GridManager,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
) -> data_alloc.NDArray:
    """Initialise prognostic state from a serialised ICON initial-condition snapshot.

    Reads ``prognostics / initial-state`` from the serialbox archive located at
    ``parameters.data_path``.  All other physics arguments are accepted but
    ignored so that this function shares the same call signature as the
    analytical initial-condition functions (``jablonowski_williamson``,
    ``gauss3d``, …) and can be stored transparently in
    `~icon4py.model.common.initial_condition.create`.
    """

    allocator = model_backends.get_allocator(backend)
    array_ns = data_alloc.import_array_ns(allocator)

    fname = f"icon_pydycore_rank{exchange.my_rank()}"
    data_path = config.data_path

    ser = serialbox.Serializer(serialbox.OpenModeKind.Read, str(data_path), fname)
    sp = ser.savepoint["smooth-topo-savepoint"].as_savepoint()
    log.debug("Reading prognostics initial-state from %s / %s", data_path, fname)

    nc = grid_manager.grid.num_cells
    return array_ns.asarray(array_ns.squeeze(ser.read("topography", sp).astype(float))[:nc])
