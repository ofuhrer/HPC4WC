# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Union

import numpy as np
from gt4py import next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.decomposition.definitions import Reductions, SingleNodeExchange
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc


try:
    import ghex  # type: ignore [import-not-found]
    import mpi4py
    from ghex.context import make_context  # type: ignore [import-not-found]
    from ghex.unstructured import (  # type: ignore [import-not-found]
        DomainDescriptor,
        HaloGenerator,
        make_communication_object,
        make_field_descriptor,
        make_pattern,
    )
    from ghex.util import Architecture  # type: ignore [import-not-found]

    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = True

    import_error: ImportError | None = None

except ImportError as e:
    mpi4py = None  # type: ignore[assignment]
    ghex = None
    unstructured = None
    import_error = e

CommId = Union[int, "mpi4py.MPI.Comm", None]
log = logging.getLogger(__name__)


def init_mpi() -> None:
    from mpi4py import MPI  # noqa: PLC0415 [import-outside-top-level]

    if not MPI.Is_initialized():
        log.info("initializing MPI")
        MPI.Init()


def finalize_mpi() -> None:
    from mpi4py import MPI  # noqa: PLC0415 [import-outside-top-level]

    if not MPI.Is_finalized():
        log.info("finalizing MPI")
        MPI.Finalize()


def _get_process_properties(with_mpi: bool = False, comm_id: CommId = None) -> Any:
    def _get_current_comm_or_comm_world(comm_id: CommId) -> mpi4py.MPI.Comm:
        if isinstance(comm_id, int):
            comm = mpi4py.MPI.Comm.f2py(comm_id)
        elif isinstance(comm_id, mpi4py.MPI.Comm):
            comm = comm_id
        else:
            comm = mpi4py.MPI.COMM_WORLD
        return comm

    if with_mpi:
        init_mpi()
        current_comm = _get_current_comm_or_comm_world(comm_id)
        return MPICommProcessProperties(current_comm)


@decomp_defs.get_process_properties.register(decomp_defs.MultiNodeRun)
def get_multinode_properties(
    s: decomp_defs.MultiNodeRun, comm_id: CommId = None
) -> decomp_defs.ProcessProperties:
    return _get_process_properties(with_mpi=True, comm_id=comm_id)


@dataclass(frozen=True)
class MPICommProcessProperties(decomp_defs.ProcessProperties):
    comm: mpi4py.MPI.Comm

    @functools.cached_property
    def rank(self) -> int:  # type: ignore [override]
        return self.comm.Get_rank()

    @functools.cached_property
    def comm_name(self) -> str:  # type: ignore [override]
        return self.comm.Get_name()

    @functools.cached_property
    def comm_size(self) -> int:  # type: ignore [override]
        return self.comm.Get_size()


class GHexMultiNodeExchange(decomp_defs.ExchangeRuntime):
    def __init__(
        self,
        process_props: decomp_defs.ProcessProperties,
        domain_decomposition: decomp_defs.DecompositionInfo,
    ):
        self._context = make_context(process_props.comm, False)
        self._domain_id_gen = decomp_defs.DomainDescriptorIdGenerator(process_props)
        self._decomposition_info = domain_decomposition
        self._domain_descriptors = {
            dim: self._create_domain_descriptor(dim) for dim in dims.horizontal_dims()
        }
        self._field_size: dict[gtx.Dimension, int] = {
            dim: self._decomposition_info.global_index(dim).shape[0]
            for dim in dims.horizontal_dims()
        }
        log.info(f"domain descriptors for dimensions {self._domain_descriptors.keys()} initialized")
        self._patterns = {dim: self._create_pattern(dim) for dim in dims.horizontal_dims()}
        log.info(f"patterns for dimensions {self._patterns.keys()} initialized ")
        self._comm = make_communication_object(self._context)

        self._applied_patterns_cache: dict = {}

        log.info("communication object initialized")

    def _domain_descriptor_info(self, descr: DomainDescriptor) -> str:
        return f" domain_descriptor=[id='{descr.domain_id()}', size='{descr.size()}', inner_size='{descr.inner_size()}' (halo size='{descr.size() - descr.inner_size()}')"

    def get_size(self) -> int:
        return self._context.size()

    def my_rank(self) -> int:
        return self._context.rank()

    def _create_domain_descriptor(self, dim: gtx.Dimension) -> DomainDescriptor:
        all_global = self._decomposition_info.global_index(
            dim, decomp_defs.DecompositionInfo.EntryType.ALL
        )
        local_halo = self._decomposition_info.local_index(
            dim, decomp_defs.DecompositionInfo.EntryType.HALO
        )
        # first arg is the domain ID which builds up an MPI Tag.
        # if those ids are not different for all domain descriptors the system might deadlock
        # if two parallel exchanges with the same domain id are done
        domain_desc = DomainDescriptor(
            self._domain_id_gen(), all_global.tolist(), local_halo.tolist()
        )
        log.debug(
            f"domain descriptor for dim='{dim.value}' with properties {self._domain_descriptor_info(domain_desc)} created"
        )
        return domain_desc

    def _create_pattern(self, horizontal_dim: gtx.Dimension) -> DomainDescriptor:
        assert horizontal_dim.kind == gtx.DimensionKind.HORIZONTAL

        global_halo_idx = self._decomposition_info.global_index(
            horizontal_dim, decomp_defs.DecompositionInfo.EntryType.HALO
        )
        halo_generator = HaloGenerator.from_gids(global_halo_idx)
        log.debug(f"halo generator for dim='{horizontal_dim.value}' created")
        pattern = make_pattern(
            self._context,
            halo_generator,
            [self._domain_descriptors[horizontal_dim]],
        )
        log.debug(
            f"pattern for dim='{horizontal_dim.value}' and {self._domain_descriptor_info(self._domain_descriptors[horizontal_dim])} created"
        )
        return pattern

    def _slice_field_based_on_dim(self, field: gtx.Field, dim: gtx.Dimension) -> data_alloc.NDArray:
        """
        Slices the field based on the dimension passed in.

        This operation is *necessary* for the use inside FORTRAN as there fields are larger than the grid (nproma size). where it does not do anything in a purely Python setup.
        the granule context where fields otherwise have length nproma.
        """
        if dim.kind == gtx.DimensionKind.HORIZONTAL:
            return field.ndarray[: self._field_size[dim]]
        else:
            raise ValueError(f"Unknown dimension {dim}")

    def _make_field_descriptor(self, dim: gtx.Dimension, array: data_alloc.NDArray) -> Any:
        return make_field_descriptor(
            self._domain_descriptors[dim],
            array,
            arch=Architecture.CPU if isinstance(array, np.ndarray) else Architecture.GPU,
        )

    def _get_applied_pattern(self, dim: gtx.Dimension, f: gtx.Field | data_alloc.NDArray) -> str:
        if isinstance(f, gtx.Field):
            assert hasattr(f, "__gt_buffer_info__")
            # dimension and buffer_info uniquely identifies the exchange pattern
            # TODO(havogt): the cache is never cleared, consider using functools.lru_cache in a bigger refactoring.
            key = (dim, f.__gt_buffer_info__.hash_key)
            try:
                return self._applied_patterns_cache[key]
            except KeyError:
                assert dim in f.domain.dims
                array = self._slice_field_based_on_dim(f, dim)
                self._applied_patterns_cache[key] = self._patterns[dim](
                    self._make_field_descriptor(dim, array)
                )
                return self._applied_patterns_cache[key]
        else:
            assert f.ndim in (1, 2), "Buffers must be 1d or 2d"
            return self._patterns[dim](self._make_field_descriptor(dim, f))

    def start(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field | data_alloc.NDArray,
        stream: decomp_defs.StreamLike = decomp_defs.DEFAULT_STREAM,
    ) -> MultiNodeResult:
        """Synchronize with `stream` and start the halo exchange of `*fields`."""
        assert dim in dims.horizontal_dims(), (
            f"first dimension must be one of ({list(dims.horizontal_dims())})"
        )

        applied_patterns = [self._get_applied_pattern(dim, f) for f in fields]
        if not ghex.__config__["gpu"]:
            # No GPU support fall back to the regular exchange function.
            handle = self._comm.exchange(applied_patterns)
        else:
            assert stream is not None
            handle = self._comm.schedule_exchange(
                patterns=applied_patterns,
                stream=stream,
            )
        log.debug(f"exchange for {len(fields)} fields of dimension ='{dim.value}' initiated.")
        return MultiNodeResult(handle, applied_patterns)

    def exchange(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field | data_alloc.NDArray,
        stream: decomp_defs.StreamLike | decomp_defs.BlockType = decomp_defs.DEFAULT_STREAM,
    ) -> None:
        # Fall back to the default implementation provided by the protocol.
        super().exchange(dim, *fields, stream=stream)
        log.debug(f"exchange for {len(fields)} fields of dimension ='{dim.value}' done.")


@dataclass
class HaloExchangeWait(decomp_defs.HaloExchangeWaitRuntime):
    exchange_object: GHexMultiNodeExchange

    def __call__(
        self,
        communication_handle: decomp_defs.ExchangeResult,
        stream: decomp_defs.StreamLike | decomp_defs.BlockType = decomp_defs.DEFAULT_STREAM,
    ) -> None:
        communication_handle.finish(stream=stream)


@decomp_defs.create_halo_exchange_wait.register(GHexMultiNodeExchange)
def create_multinode_halo_exchange_wait(runtime: GHexMultiNodeExchange) -> HaloExchangeWait:
    return HaloExchangeWait(runtime)


@dataclass
class MultiNodeResult(decomp_defs.ExchangeResult):
    handle: Any
    pattern_refs: Any

    def finish(
        self,
        stream: decomp_defs.StreamLike | decomp_defs.BlockType = decomp_defs.DEFAULT_STREAM,
    ) -> None:
        """Finish the initiated halo exchange and either block or schedule completion on `stream`."""
        if (not ghex.__config__["gpu"]) or stream is decomp_defs.BLOCK:
            # No GPU support or blocking wait requested -> use normal `wait()`.
            self.handle.wait()

        else:
            # Stream given, perform a scheduled wait.
            self.handle.schedule_wait(stream)

    def is_ready(self) -> bool:
        return self.handle.is_ready()


@decomp_defs.create_exchange.register(MPICommProcessProperties)
def create_multinode_node_exchange(
    process_props: MPICommProcessProperties, decomp_info: decomp_defs.DecompositionInfo
) -> decomp_defs.ExchangeRuntime:
    if process_props.comm_size > 1:
        return GHexMultiNodeExchange(process_props, decomp_info)
    else:
        return SingleNodeExchange()


@dataclasses.dataclass
class GlobalReductions(Reductions):
    """
    MPI-aware global reductions.

    Owner masks from the decomposition info are stored internally, keyed by the
    horizontal dimension size (num_cells, num_edges, num_vertices). The correct
    mask is resolved from buffer.shape[0], ensuring only owned (non-halo)
    elements participate in the reduction.
    """

    # TODO (jcanton,msimberg,nfarabullini): the reductions may be better if
    # receiving Fields as arguments instead of NDArray, such that they get
    # domain info that can be used for masks

    process_props: decomp_defs.ProcessProperties
    _owner_masks: dict[int, data_alloc.NDArray] = dataclasses.field(default_factory=dict)

    def __init__(
        self,
        process_props: decomp_defs.ProcessProperties,
        decomposition_info: decomp_defs.DecompositionInfo,
    ) -> None:
        self.process_props = process_props
        self._owner_masks = {}
        for dim in (dims.CellDim, dims.EdgeDim, dims.VertexDim):
            mask = decomposition_info.owner_mask(dim)
            size = mask.shape[0]
            if size in self._owner_masks:
                raise ValueError(
                    f"Ambiguous horizontal dimension size {size}: multiple dimensions "
                    f"have the same local size. Cannot auto-resolve owner mask."
                )
            self._owner_masks[size] = mask

    def _prepare_buffer(self, buffer: data_alloc.NDArray) -> data_alloc.NDArray:
        if len(buffer.shape) > 0:
            owner_mask = self._resolve_owner_mask(buffer)
            return buffer[owner_mask]
        return buffer

    def _resolve_owner_mask(self, buffer: data_alloc.NDArray) -> data_alloc.NDArray:
        """Resolve the 1D owner mask for the buffer's first dimension.

        The returned mask is always 1D (num_horizontal,). When used for
        indexing, NumPy's boolean indexing with a 1D mask on a
        multi-dimensional array selects along the first axis, so
        ``buffer[mask]`` works correctly for both 1D buffers of shape
        ``(num_horizontal,)`` and 2D buffers of shape
        ``(num_horizontal, K)``.
        """
        first_dim_size = buffer.shape[0]
        if first_dim_size not in self._owner_masks:
            raise ValueError(
                f"Cannot resolve owner mask: buffer's first dimension size "
                f"({first_dim_size}) does not match any known horizontal "
                f"dimension (known sizes: {list(self._owner_masks.keys())})."
            )
        return self._owner_masks[first_dim_size]

    @staticmethod
    def _min_identity(dtype: np.dtype, array_ns: ModuleType = np) -> data_alloc.NDArray:
        if array_ns.issubdtype(dtype, array_ns.integer):
            return array_ns.asarray([dtype.type(array_ns.iinfo(dtype).max)])
        elif array_ns.issubdtype(dtype, array_ns.floating):
            return array_ns.asarray([dtype.type(array_ns.inf)])
        else:
            raise TypeError(f"Unsupported dtype for min identity: {dtype}")

    @staticmethod
    def _max_identity(dtype: np.dtype, array_ns: ModuleType = np) -> data_alloc.NDArray:
        if array_ns.issubdtype(dtype, array_ns.integer):
            return array_ns.asarray([dtype.type(array_ns.iinfo(dtype).min)])
        elif array_ns.issubdtype(dtype, array_ns.floating):
            return array_ns.asarray([dtype.type(-array_ns.inf)])
        else:
            raise TypeError(f"Unsupported dtype for max identity: {dtype}")

    @staticmethod
    def _sum_identity(dtype: np.dtype, array_ns: ModuleType = np) -> data_alloc.NDArray:
        return array_ns.asarray([dtype.type(0)])

    def _reduce(
        self,
        buffer: data_alloc.NDArray,
        local_reduction: Callable[[data_alloc.NDArray], data_alloc.ScalarT],
        global_reduction: mpi4py.MPI.Op,
    ) -> state_utils.ScalarType:
        array_ns = data_alloc.array_namespace(buffer)
        local_red_val = local_reduction(buffer)
        recv_buffer = array_ns.empty(1, dtype=buffer.dtype)
        if hasattr(
            array_ns, "cuda"
        ):  # https://mpi4py.readthedocs.io/en/stable/tutorial.html#gpu-aware-mpi-python-gpu-arrays
            array_ns.cuda.runtime.deviceSynchronize()
        self.process_props.comm.Allreduce(local_red_val, recv_buffer, global_reduction)
        return recv_buffer.item()

    def _calc_buffer_size(
        self,
        buffer: data_alloc.NDArray,
    ) -> state_utils.ScalarType:
        array_ns = data_alloc.array_namespace(buffer)
        return self._reduce(array_ns.asarray([buffer.size]), array_ns.sum, mpi4py.MPI.SUM)

    def min(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType:
        buffer = self._prepare_buffer(buffer)
        array_ns = data_alloc.array_namespace(buffer)
        if self._calc_buffer_size(buffer) == 0:
            raise ValueError("global_min requires a non-empty buffer")
        return self._reduce(
            buffer if buffer.size != 0 else self._min_identity(buffer.dtype, array_ns),
            array_ns.min,
            mpi4py.MPI.MIN,
        )

    def max(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType:
        buffer = self._prepare_buffer(buffer)
        array_ns = data_alloc.array_namespace(buffer)
        if self._calc_buffer_size(buffer) == 0:
            raise ValueError("global_max requires a non-empty buffer")
        return self._reduce(
            buffer if buffer.size != 0 else self._max_identity(buffer.dtype, array_ns),
            array_ns.max,
            mpi4py.MPI.MAX,
        )

    def sum(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType:
        buffer = self._prepare_buffer(buffer)
        array_ns = data_alloc.array_namespace(buffer)
        if self._calc_buffer_size(buffer) == 0:
            raise ValueError("global_sum requires a non-empty buffer")
        return self._reduce(
            buffer if buffer.size != 0 else self._sum_identity(buffer.dtype, array_ns),
            array_ns.sum,
            mpi4py.MPI.SUM,
        )

    def mean(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType:
        buffer = self._prepare_buffer(buffer)
        array_ns = data_alloc.array_namespace(buffer)
        global_buffer_size = self._calc_buffer_size(buffer)
        if global_buffer_size == 0:
            raise ValueError("global_mean requires a non-empty buffer")

        return (
            self._reduce(
                (buffer if buffer.size != 0 else self._sum_identity(buffer.dtype, array_ns)),
                array_ns.sum,
                mpi4py.MPI.SUM,
            )
            / global_buffer_size
        )


@decomp_defs.create_reduction.register(MPICommProcessProperties)
def create_global_reduction(
    process_props: MPICommProcessProperties, decomposition_info: decomp_defs.DecompositionInfo
) -> Reductions:
    return GlobalReductions(process_props, decomposition_info)
