# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

from gt4py.next import common as gtx_common

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import horizontal as h_grid, icon


logger = logging.getLogger(__name__)


def print_grid_decomp_info(
    *,
    icon_grid: icon.IconGrid,
    process_props: definitions.ProcessProperties,
    decomposition_info: definitions.DecompositionInfo,
    num_cells: int,
    num_edges: int,
    num_verts: int,
) -> None:
    logger.info(
        "icon_grid:cell_start for rank %s is.... %s",
        process_props.rank,
        "\n".join(
            [
                f"{k!s}  - {icon_grid.start_index(k)}"
                for k in h_grid.get_domains_for_dim(dims.CellDim)
            ]
        ),
    )
    logger.info(
        "icon_grid:cell_end for rank %s is.... %s",
        process_props.rank,
        "\n".join(
            [f"{k!s}  - {icon_grid.end_index(k)}" for k in h_grid.get_domains_for_dim(dims.CellDim)]
        ),
    )
    logger.info(
        "icon_grid:vert_start for rank %s is.... %s",
        process_props.rank,
        "\n".join(
            [
                f"{k!s}  - {icon_grid.start_index(k)}"
                for k in h_grid.get_domains_for_dim(dims.VertexDim)
            ]
        ),
    )
    logger.info(
        "icon_grid:vert_end for rank %s is.... %s",
        process_props.rank,
        "\n".join(
            [
                f"{k!s}  - {icon_grid.end_index(k)}"
                for k in h_grid.get_domains_for_dim(dims.VertexDim)
            ]
        ),
    )
    logger.info(
        "icon_grid:edge_start for rank %s is.... %s",
        process_props.rank,
        "\n".join(
            [
                f"{k!s}  - {icon_grid.start_index(k)}"
                for k in h_grid.get_domains_for_dim(dims.EdgeDim)
            ]
        ),
    )
    logger.info(
        "icon_grid:edge_end for rank %s is.... %s",
        process_props.rank,
        "\n".join(
            [f"{k!s}  - {icon_grid.end_index(k)}" for k in h_grid.get_domains_for_dim(dims.EdgeDim)]
        ),
    )

    for offset, connectivity in icon_grid.connectivities.items():
        if gtx_common.is_neighbor_table(connectivity):
            logger.debug(
                f"icon_grid:{offset} for rank {process_props.rank} is.... {connectivity.asnumpy()}"
            )

    logger.info(
        "c_glb_index for rank %s is.... %s",
        process_props.rank,
        decomposition_info.global_index(dims.CellDim)[0:num_cells],
    )
    logger.info(
        "e_glb_index for rank %s is.... %s",
        process_props.rank,
        decomposition_info.global_index(dims.EdgeDim)[0:num_edges],
    )
    logger.info(
        "v_glb_index for rank %s is.... %s",
        process_props.rank,
        decomposition_info.global_index(dims.VertexDim)[0:num_verts],
    )

    logger.info(
        "c_owner_mask for rank %s is.... %s",
        process_props.rank,
        decomposition_info.owner_mask(dims.CellDim)[0:num_cells],
    )
    logger.info(
        "e_owner_mask for rank %s is.... %s",
        process_props.rank,
        decomposition_info.owner_mask(dims.EdgeDim)[0:num_edges],
    )
    logger.info(
        "v_owner_mask for rank %s is.... %s",
        process_props.rank,
        decomposition_info.owner_mask(dims.VertexDim)[0:num_verts],
    )

    logger.info(
        f"local cells = {decomposition_info.global_index(dims.CellDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(dims.EdgeDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(dims.VertexDim, definitions.DecompositionInfo.EntryType.ALL).shape}"
    )
    logger.info(
        f"rank={process_props.rank}/{process_props.comm_size}:  GHEX context setup: from {process_props.comm_name} with {process_props.comm_size} nodes"
    )
