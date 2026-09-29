# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import logging
from typing import Annotated

import numpy as np
from gt4py import next as gtx
from gt4py.next.type_system import type_specifications as ts

import icon4py.model.common.grid.states as grid_states
from icon4py.bindings import (
    common as wrapper_common,
    debug_utils as wrapper_debug_utils,
    icon4py_export,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, model_backends
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import icon as icon_grid, vertical
from icon4py.model.common.type_alias import wpfloat
from icon4py.tools import py2fgen


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GridState:
    grid: icon_grid.IconGrid
    vertical_grid: vertical.VerticalGrid
    edge_geometry: grid_states.EdgeParams
    cell_geometry: grid_states.CellParams
    exchange_runtime: decomposition_defs.ExchangeRuntime


grid_state: GridState | None = None  # TODO(havogt): remove module global state

type NumpyInt32Array1D = Annotated[
    np.ndarray,
    py2fgen.ArrayParamDescriptor(
        rank=1,
        dtype=ts.ScalarKind.INT32,
        memory_space=py2fgen.MemorySpace.HOST,
        is_optional=False,
    ),
]

type NumpyBoolArray1D = Annotated[
    np.ndarray,
    py2fgen.ArrayParamDescriptor(
        rank=1,
        dtype=ts.ScalarKind.BOOL,
        memory_space=py2fgen.MemorySpace.HOST,
        is_optional=False,
    ),
]


@icon4py_export.export
def grid_init(  # noqa: PLR0917 [too-many-positional-arguments]
    cell_starts: NumpyInt32Array1D,
    cell_ends: NumpyInt32Array1D,
    vertex_starts: NumpyInt32Array1D,
    vertex_ends: NumpyInt32Array1D,
    edge_starts: NumpyInt32Array1D,
    edge_ends: NumpyInt32Array1D,
    c2e: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], gtx.int32],
    e2c: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.int32],
    c2e2c: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], gtx.int32],
    e2c2e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], gtx.int32],
    e2v: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2VDim], gtx.int32],
    v2e: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], gtx.int32],
    v2c: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], gtx.int32],
    e2c2v: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.int32],
    c2v: gtx.Field[gtx.Dims[dims.CellDim, dims.C2VDim], gtx.int32],
    c_owner_mask: NumpyBoolArray1D,
    e_owner_mask: NumpyBoolArray1D,
    v_owner_mask: NumpyBoolArray1D,
    c_glb_index: NumpyInt32Array1D,
    e_glb_index: NumpyInt32Array1D,
    v_glb_index: NumpyInt32Array1D,
    tangent_orientation: fa.EdgeField[wpfloat],
    inverse_primal_edge_lengths: fa.EdgeField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    edge_areas: fa.EdgeField[wpfloat],
    f_e: fa.EdgeField[wpfloat],
    cell_center_lat: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    cell_center_lon: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    cell_areas: gtx.Field[gtx.Dims[dims.CellDim], gtx.float64],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.float64],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.float64],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.float64],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], gtx.float64],
    primal_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    primal_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    dual_normal_cell_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    dual_normal_cell_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], gtx.float64],
    edge_center_lat: fa.EdgeField[wpfloat],
    edge_center_lon: fa.EdgeField[wpfloat],
    primal_normal_x: fa.EdgeField[wpfloat],
    primal_normal_y: fa.EdgeField[wpfloat],
    vct_a: gtx.Field[gtx.Dims[dims.KDim], gtx.float64],
    lowest_layer_thickness: gtx.float64,
    model_top_height: gtx.float64,
    stretch_factor: gtx.float64,
    flat_height: gtx.float64,
    rayleigh_damping_height: gtx.float64,
    mean_cell_area: gtx.float64,
    comm_id: gtx.int32,
    num_vertices: gtx.int32,
    num_cells: gtx.int32,
    num_edges: gtx.int32,
    vertical_size: gtx.int32,
    limited_area: bool,
    backend: gtx.int32,
) -> None:
    on_gpu = c2e.array_ns != np  # TODO(havogt): expose `on_gpu` from py2fgen
    actual_backend = wrapper_common.select_backend(
        wrapper_common.BackendIntEnum(backend), on_gpu=on_gpu
    )
    allocator = model_backends.get_allocator(actual_backend)

    if comm_id is None:
        process_props = decomposition_defs.SingleNodeProcessProperties()
        exchange_runtime = decomposition_defs.SingleNodeExchange()
    else:
        # Set MultiNodeExchange as exchange runtime
        (
            process_props,
            decomposition_info,
            exchange_runtime,
        ) = wrapper_common.construct_decomposition(
            c_glb_index=c_glb_index,
            e_glb_index=e_glb_index,
            v_glb_index=v_glb_index,
            c_owner_mask=c_owner_mask,
            e_owner_mask=e_owner_mask,
            v_owner_mask=v_owner_mask,
            num_cells=num_cells,
            num_edges=num_edges,
            num_vertices=num_vertices,
            comm_id=comm_id,
        )

    grid = wrapper_common.construct_icon_grid(
        cell_starts=cell_starts,
        cell_ends=cell_ends,
        vertex_starts=vertex_starts,
        vertex_ends=vertex_ends,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        c2e=c2e.ndarray,
        e2c=e2c.ndarray,
        c2e2c=c2e2c.ndarray,
        e2c2e=e2c2e.ndarray,
        e2v=e2v.ndarray,
        v2e=v2e.ndarray,
        v2c=v2c.ndarray,
        e2c2v=e2c2v.ndarray,
        c2v=c2v.ndarray,
        grid_id="icon_grid",
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
        distributed=not process_props.is_single_rank(),
        allocator=allocator,
    )

    if comm_id is not None:
        wrapper_debug_utils.print_grid_decomp_info(
            icon_grid=grid,
            process_props=process_props,
            decomposition_info=decomposition_info,
            num_cells=num_cells,
            num_edges=num_edges,
            num_verts=num_vertices,
        )

    # Vertical grid config
    vertical_config = vertical.VerticalGridConfig(
        num_levels=vertical_size,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=rayleigh_damping_height,
        flat_height=flat_height,
        # TODO(havogt): pass all other params from Fortran and ensure
        # we don't miss new ones in the future
    )

    # Vertical parameters
    vertical_grid = vertical.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=None,
    )

    # Edge geometry
    edge_params = grid_states.EdgeParams(
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_lengths=inv_dual_edge_length,
        inverse_vertex_vertex_lengths=inv_vert_vert_length,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_cell_x=dual_normal_cell_x,
        dual_normal_cell_y=dual_normal_cell_y,
        edge_areas=edge_areas,
        coriolis_frequency=f_e,
        edge_center_lat=edge_center_lat,
        edge_center_lon=edge_center_lon,
        primal_normal_x=primal_normal_x,
        primal_normal_y=primal_normal_y,
    )

    # Cell geometry
    cell_params = grid_states.CellParams(
        cell_center_lat=cell_center_lat,
        cell_center_lon=cell_center_lon,
        area=cell_areas,
        mean_cell_area=mean_cell_area,
    )

    global grid_state  # noqa: PLW0603 [global-statement]
    grid_state = GridState(
        grid=grid,
        vertical_grid=vertical_grid,
        edge_geometry=edge_params,
        cell_geometry=cell_params,
        exchange_runtime=exchange_runtime,
    )
