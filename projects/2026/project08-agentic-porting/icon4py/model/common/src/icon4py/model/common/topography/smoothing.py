# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import data_allocation as data_alloc


def compute_nabla2_on_cell(
    psi_c: data_alloc.NDArray,
    geofac_n2s: data_alloc.NDArray,
    c2e2co: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """Compute the Laplacian (nabla squared) of a scalar field defined on cell centres."""
    array_ns = data_alloc.array_namespace(psi_c)
    return array_ns.sum(psi_c[c2e2co] * geofac_n2s, axis=1)


def smooth_topography(
    *,
    topography: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    geofac_n2s: data_alloc.NDArray,
    c2e2co: data_alloc.NDArray,
    exchange: decomposition.ExchangeRuntime,
    num_iterations: int = 25,
) -> data_alloc.NDArray:
    """Compute the smoothed (laplacian-filtered) topography needed by the SLEVE coordinate."""
    array_ns = data_alloc.array_namespace(topography)
    smooth_topo = topography.copy()
    # TODO(jcanton): if the input topography is properly exchanged, which it should, this is not needed here.
    exchange.exchange(dims.CellDim, smooth_topo, stream=decomposition.BLOCK)

    for _ in range(num_iterations):
        nabla2_topo = compute_nabla2_on_cell(smooth_topo, geofac_n2s, c2e2co)
        array_ns.add(smooth_topo, 0.125 * nabla2_topo * cell_areas, out=smooth_topo)
        exchange.exchange(dims.CellDim, smooth_topo, stream=decomposition.BLOCK)

    return smooth_topo
