# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import functools
import logging
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import TYPE_CHECKING, Annotated, Final

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
from gt4py import eve
from gt4py._core import definitions as gt4py_definitions
from gt4py.next.type_system import type_specifications as ts

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import definitions, mpi_decomposition
from icon4py.model.common.grid import base, horizontal as h_grid, icon
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.common.utils.data_allocation import adjust_fortran_indices
from icon4py.tools import py2fgen


if TYPE_CHECKING:
    xp: Final = np
else:
    try:
        import cupy as cp

        xp = cp
    except ImportError:
        cp = None
        xp = np


type NDArray = np.ndarray | xp.ndarray

# TODO(havogt): import needed to register MultNodeRun in get_process_properties, does the pattern make sense?
assert hasattr(mpi_decomposition, "get_multinode_properties")

log = logging.getLogger(__name__)

type Int32Array1D = Annotated[
    data_alloc.NDArray,
    py2fgen.ArrayParamDescriptor(
        rank=1,
        dtype=ts.ScalarKind.INT32,
        memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
        is_optional=False,
    ),
]

type OptionalInt32Array1D = Annotated[
    data_alloc.NDArray | None,
    py2fgen.ArrayParamDescriptor(
        rank=1,
        dtype=ts.ScalarKind.INT32,
        memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
        is_optional=True,
    ),
]

type OptionalInt32Array2D = Annotated[
    data_alloc.NDArray | None,
    py2fgen.ArrayParamDescriptor(
        rank=2,
        dtype=ts.ScalarKind.INT32,
        memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
        is_optional=True,
    ),
]

type Float64Array1D = Annotated[
    data_alloc.NDArray,
    py2fgen.ArrayParamDescriptor(
        rank=1,
        dtype=ts.ScalarKind.FLOAT64,
        memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
        is_optional=False,
    ),
]

type Float64Array2D = Annotated[
    data_alloc.NDArray,
    py2fgen.ArrayParamDescriptor(
        rank=2,
        dtype=ts.ScalarKind.FLOAT64,
        memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
        is_optional=False,
    ),
]

type Float64Array3D = Annotated[
    data_alloc.NDArray,
    py2fgen.ArrayParamDescriptor(
        rank=3,
        dtype=ts.ScalarKind.FLOAT64,
        memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
        is_optional=False,
    ),
]

type OptionalFloat64Array1D = Annotated[
    data_alloc.NDArray | None,
    py2fgen.ArrayParamDescriptor(
        rank=1,
        dtype=ts.ScalarKind.FLOAT64,
        memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
        is_optional=True,
    ),
]

type OptionalFloat64Array2D = Annotated[
    data_alloc.NDArray | None,
    py2fgen.ArrayParamDescriptor(
        rank=2,
        dtype=ts.ScalarKind.FLOAT64,
        memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
        is_optional=True,
    ),
]


class BackendIntEnum(eve.IntEnum):
    DEFAULT = 0
    DACE = 1
    GTFN = 2


def select_backend(
    selector: BackendIntEnum, on_gpu: bool
) -> gtx_typing.Backend | model_backends.BackendDescriptor:
    backend_descriptor: model_backends.BackendDescriptor = {}
    backend_descriptor["device"] = model_backends.GPU if on_gpu else model_backends.CPU
    if selector == BackendIntEnum.DEFAULT:
        return backend_descriptor
    if selector == BackendIntEnum.DACE:
        backend_descriptor["backend_factory"] = model_backends.make_custom_dace_backend
        return backend_descriptor
    if selector == BackendIntEnum.GTFN:
        backend_descriptor["backend_factory"] = model_backends.make_custom_gtfn_backend
        return backend_descriptor

    raise ValueError(f"Invalid backend selector: {selector}")


def cached_dummy_field_factory(
    allocator: gtx_typing.Allocator | None,
) -> Callable[[str, gtx.Domain, gt4py_definitions.DType], gtx.Field]:
    # curried to exclude non-hashable backend from cache
    @functools.lru_cache(maxsize=20)
    def impl(_name: str, domain: gtx.Domain, dtype: gt4py_definitions.DType) -> gtx.Field:
        # _name is used to differentiate between different dummy fields
        return gtx.zeros(domain, dtype=dtype, allocator=allocator)

    return impl


def shrink_to_dimension(
    sizes: dict[gtx.Dimension, int], tables: dict[gtx.FieldOffset, NDArray]
) -> dict[gtx.FieldOffset, NDArray]:
    """Shrink the neighbor tables from nproma size to the actual size of the grid."""
    return {k: v[: sizes[k.target[0]]] for k, v in tables.items()}


def add_origin(xp: ModuleType, table: NDArray) -> NDArray:
    return xp.column_stack((xp.arange(table.shape[0], dtype=xp.int32), table))


def get_nproma(tables: Iterable[NDArray]) -> int:
    tables = list(tables)
    nproma = tables[0].shape[0]
    if not all(table.shape[0] == nproma for table in tables):
        raise ValueError("All connectivity tables must have the same number of rows (nproma).")
    return nproma


def construct_icon_grid(
    *,
    cell_starts: np.ndarray,
    cell_ends: np.ndarray,
    vertex_starts: np.ndarray,
    vertex_ends: np.ndarray,
    edge_starts: np.ndarray,
    edge_ends: np.ndarray,
    c2e: NDArray,
    e2c: NDArray,
    c2e2c: NDArray,
    e2c2e: NDArray,
    e2v: NDArray,
    v2e: NDArray,
    v2c: NDArray,
    e2c2v: NDArray,
    c2v: NDArray,
    grid_id: str,
    num_vertices: int,
    num_cells: int,
    num_edges: int,
    vertical_size: int,
    limited_area: bool,
    distributed: bool,
    allocator: gtx_typing.Allocator | None,
) -> icon.IconGrid:
    log.debug("Constructing ICON Grid in Python...")
    log.debug("num_cells:%s", num_cells)
    log.debug("num_edges:%s", num_edges)
    log.debug("num_vertices:%s", num_vertices)
    log.debug("num_levels:%s", vertical_size)

    log.debug("Offsetting Fortran connectivitity arrays by 1")

    xp = data_alloc.import_array_ns(allocator)
    start_indices = {
        # TODO(halungge): ICON Fortran has 0 values in these arrays in some places possibly where they don't use them.
        # We should investigate where we access these values.
        dims.CellDim: np.maximum(0, adjust_fortran_indices(cell_starts)),
        dims.EdgeDim: np.maximum(0, adjust_fortran_indices(edge_starts)),
        dims.VertexDim: np.maximum(0, adjust_fortran_indices(vertex_starts)),
    }
    end_indices = {dims.CellDim: cell_ends, dims.EdgeDim: edge_ends, dims.VertexDim: vertex_ends}

    c2e = adjust_fortran_indices(c2e)
    c2v = adjust_fortran_indices(c2v)
    v2c = adjust_fortran_indices(v2c)
    e2v = adjust_fortran_indices(e2v)[
        :, 0:2
    ]  # slicing required for e2v as input data is actually e2c2v
    c2e2c = adjust_fortran_indices(c2e2c)
    v2e = adjust_fortran_indices(v2e)
    e2c2v = adjust_fortran_indices(e2c2v)
    e2c = adjust_fortran_indices(e2c)
    e2c2e = adjust_fortran_indices(e2c2e)

    # stacked arrays
    c2e2c0 = add_origin(xp, c2e2c)
    e2c2e0 = add_origin(xp, e2c2e)

    config = base.GridConfig(
        horizontal_config=base.HorizontalGridSize(
            num_vertices=num_vertices,
            num_cells=num_cells,
            num_edges=num_edges,
        ),
        vertical_size=vertical_size,
        limited_area=limited_area,
        distributed=distributed,
        keep_skip_values=False,
    )

    neighbor_tables = {
        dims.C2E: c2e,
        dims.C2V: c2v,
        dims.E2C: e2c,
        dims.E2C2E: e2c2e,
        dims.C2E2C: c2e2c,
        dims.C2E2CO: c2e2c0,
        dims.E2C2EO: e2c2e0,
        dims.V2E: v2e,
        dims.E2V: e2v,
        dims.E2C2V: e2c2v,
        dims.V2C: v2c,
    }

    neighbor_tables = shrink_to_dimension(
        sizes={dims.EdgeDim: num_edges, dims.VertexDim: num_vertices, dims.CellDim: num_cells},
        tables=neighbor_tables,
    )
    domain_bounds_constructor = functools.partial(
        h_grid.get_start_end_idx_from_icon_arrays,
        start_indices=start_indices,
        end_indices=end_indices,
    )
    start_index, end_index = icon.get_start_and_end_index(domain_bounds_constructor)

    return icon.icon_grid(
        id_=grid_id,
        allocator=allocator,
        config=config,
        neighbor_tables=neighbor_tables,
        start_index=start_index,
        end_index=end_index,
        grid_params=icon.GridParams(),
    )


def construct_decomposition(
    *,
    c_glb_index: np.ndarray,
    e_glb_index: np.ndarray,
    v_glb_index: np.ndarray,
    c_owner_mask: np.ndarray,
    e_owner_mask: np.ndarray,
    v_owner_mask: np.ndarray,
    num_cells: int,
    num_edges: int,
    num_vertices: int,
    comm_id: int,
) -> tuple[
    definitions.ProcessProperties, definitions.DecompositionInfo, definitions.ExchangeRuntime
]:
    log.debug("Offsetting Fortran connectivitity arrays by 1")
    c_glb_index = adjust_fortran_indices(c_glb_index)
    e_glb_index = adjust_fortran_indices(e_glb_index)
    v_glb_index = adjust_fortran_indices(v_glb_index)

    c_owner_mask = c_owner_mask[:num_cells]
    e_owner_mask = e_owner_mask[:num_edges]
    v_owner_mask = v_owner_mask[:num_vertices]

    decomposition_info = (
        definitions.DecompositionInfo()
        # TODO (halungge): last argument is called `decomp_domain` in icon, it is not needed in the granules should we pass it nevertheless?
        .set_dimension(dims.CellDim, c_glb_index, c_owner_mask, None)
        .set_dimension(dims.EdgeDim, e_glb_index, e_owner_mask, None)
        .set_dimension(dims.VertexDim, v_glb_index, v_owner_mask, None)
    )
    process_props = definitions.get_process_properties(definitions.MultiNodeRun(), comm_id)
    exchange = definitions.create_exchange(process_props, decomposition_info)

    return process_props, decomposition_info, exchange
