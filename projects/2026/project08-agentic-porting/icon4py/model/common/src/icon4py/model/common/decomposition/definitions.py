# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import dataclasses
import functools
import logging
import warnings
from enum import Enum
from typing import Any, ClassVar, Literal, Protocol, overload, runtime_checkable

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, utils
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)

# TODO(all): Currently this is the only place that uses/needs stream. Move them to an
#   appropriate location once the need arises.


class CupyLikeStream(Protocol):
    """The type follows the CuPy convention of a stream.

    This means they have an attribute `ptr` that returns the address of the
    underlying GPU stream.
    See: https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html#cupy-cuda-stream

    Todo:
        Drop once we fully translated to CuPy 13.
    """

    @property
    def ptr(self) -> int: ...


class CudaStreamProtocol(Protocol):
    """The type follows the CUDA stream protocol.

    This means it provides a method called `__cuda_stream__()` returning a pair of
    integers. The first is the protocol version and the second value is the
    address of the stream.
    See: https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
    """

    def __cuda_stream__(self) -> tuple[int, int]: ...


type StreamLike = CupyLikeStream | CudaStreamProtocol


@dataclasses.dataclass(frozen=True)
class Stream:
    """Stream object used in ICON4Py

    Args:
        ptr: The address of the underlying stream.
    """

    ptr: int

    def __cuda_stream__(self) -> tuple[int, int]:
        return 0, self.ptr


DEFAULT_STREAM = Stream(0)
"""Default stream of the device.

Its availability is not tied to a particular device, it is thus also present in a
purely CPU setting, where it is safe to use and usually represents fully blocking
semantic.
"""


class BlockType:
    _instance = None

    def __new__(cls) -> BlockType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


BLOCK = BlockType()
"""
Constant used by `ExchangeResult.finish()` to indicate that blocking semantics should
be used, i.e. wait until the exchange has fully finished not until it has merely been
scheduled on the device.
"""


class ProcessProperties(Protocol):
    comm: Any
    rank: int
    comm_name: str
    comm_size: int

    def is_single_rank(self) -> bool:
        return self.comm_size == 1


@dataclasses.dataclass(frozen=True, init=False)
class SingleNodeProcessProperties(ProcessProperties):
    comm: None
    rank: int
    comm_name: str
    comm_size: int

    def __init__(self) -> None:
        object.__setattr__(self, "comm", None)
        object.__setattr__(self, "rank", 0)
        object.__setattr__(self, "comm_name", "")
        object.__setattr__(self, "comm_size", 1)


class DomainDescriptorIdGenerator:
    _counter = 0
    _roundtrips = 0

    def __init__(self, process_props: ProcessProperties):
        self._comm_size = process_props.comm_size
        self._roundtrips = process_props.rank
        self._base = self._roundtrips * self._comm_size

    def __call__(self) -> int:
        next_id = self._base + self._counter
        if self._counter + 1 >= self._comm_size:
            self._roundtrips = self._roundtrips + self._comm_size
            self._base = self._roundtrips * self._comm_size
            self._counter = 0
        else:
            self._counter = self._counter + 1
        return next_id


class DecompositionInfo:
    def __init__(
        self,
    ) -> None:
        self._global_index: dict[gtx.Dimension, data_alloc.NDArray] = {}
        self._halo_levels: dict[gtx.Dimension, data_alloc.NDArray] = {}
        self._owner_mask: dict[gtx.Dimension, data_alloc.NDArray] = {}

    class EntryType(int, Enum):
        ALL = 0
        OWNED = 1
        HALO = 2
        HALO_LEVEL_1 = 11
        HALO_LEVEL_2 = 12

    @utils.chainable
    def set_dimension(
        self,
        dim: gtx.Dimension,
        global_index: data_alloc.NDArray,
        owner_mask: data_alloc.NDArray,
        halo_levels: data_alloc.NDArray | None,
    ) -> None:
        self._global_index[dim] = global_index
        self._owner_mask[dim] = owner_mask
        assert halo_levels is None or (halo_levels != DecompositionFlag.UNDEFINED.value).all()
        self._halo_levels[dim] = halo_levels

    def is_distributed(self) -> bool:
        return max(self._halo_levels[dims.CellDim]).item() > DecompositionFlag.OWNED

    def local_index(
        self, dim: gtx.Dimension, entry_type: EntryType = EntryType.ALL
    ) -> data_alloc.NDArray:
        match entry_type:
            case DecompositionInfo.EntryType.ALL:
                return self._to_local_index(dim)
            case DecompositionInfo.EntryType.HALO:
                index = self._to_local_index(dim)
                mask = self._owner_mask[dim]
                return index[~mask]
            case DecompositionInfo.EntryType.HALO_LEVEL_1:
                index = self._to_local_index(dim)
                mask = self.halo_level_mask(dim=dim, level=DecompositionFlag.FIRST_HALO_LEVEL)
                return index[mask]
            case DecompositionInfo.EntryType.HALO_LEVEL_2:
                index = self._to_local_index(dim)
                mask = self.halo_level_mask(dim=dim, level=DecompositionFlag.SECOND_HALO_LEVEL)
                return index[mask]
            case DecompositionInfo.EntryType.OWNED:
                index = self._to_local_index(dim)
                mask = self._owner_mask[dim]
                return index[mask]

    def _to_local_index(self, dim: gtx.Dimension) -> data_alloc.NDArray:
        data = self._global_index[dim]
        assert data.ndim == 1
        return data_alloc.array_namespace(data).arange(data.shape[0])

    def owner_mask(self, dim: gtx.Dimension) -> data_alloc.NDArray:
        return self._owner_mask[dim]

    def global_index(
        self,
        dim: gtx.Dimension,
        entry_type: DecompositionInfo.EntryType = EntryType.ALL,
    ) -> data_alloc.NDArray:
        match entry_type:
            case DecompositionInfo.EntryType.ALL:
                return self._global_index[dim]
            case DecompositionInfo.EntryType.OWNED:
                return self._global_index[dim][self._owner_mask[dim]]
            case DecompositionInfo.EntryType.HALO:
                return self._global_index[dim][~self._owner_mask[dim]]
            case DecompositionInfo.EntryType.HALO_LEVEL_1:
                mask = self.halo_level_mask(dim=dim, level=DecompositionFlag.FIRST_HALO_LEVEL)
                return self._global_index[dim][mask]
            case DecompositionInfo.EntryType.HALO_LEVEL_2:
                mask = self.halo_level_mask(dim=dim, level=DecompositionFlag.SECOND_HALO_LEVEL)
                return self._global_index[dim][mask]
            case _:
                raise NotImplementedError()

    def get_horizontal_size(self) -> base.HorizontalGridSize:
        return base.HorizontalGridSize(
            num_cells=self.global_index(dims.CellDim, self.EntryType.ALL).shape[0],
            num_edges=self.global_index(dims.EdgeDim, self.EntryType.ALL).shape[0],
            num_vertices=self.global_index(dims.VertexDim, self.EntryType.ALL).shape[0],
        )

    def get_halo_size(self, dim: gtx.Dimension, flag: DecompositionFlag) -> int:
        level_mask = self.halo_level_mask(dim, flag)
        return data_alloc.array_namespace(level_mask).count_nonzero(level_mask)

    def halo_levels(self, dim: gtx.Dimension) -> data_alloc.NDArray:
        return self._halo_levels[dim]

    def halo_level_mask(self, dim: gtx.Dimension, level: DecompositionFlag) -> data_alloc.NDArray:
        levels = self._halo_levels[dim]
        return data_alloc.array_namespace(levels).where(levels == level.value, True, False)


class ExchangeResult(Protocol):
    def finish(
        self,
        stream: StreamLike | BlockType = DEFAULT_STREAM,
    ) -> None:
        """Wait on the halo exchange.

        Finishes the halo exchange represented by this `ExchangeResult`, see
        `ExchangeRuntime.start()` for more.
        When the function returns the exchange has not necessarily completed yet,
        but has been scheduled on the device using stream `stream`. This means
        that all further work submitted to `stream` will wait until the exchange
        has completed. By default `DEFAULT_STREAM` is used.

        In case `stream` is `BLOCK` the function will only return after the exchange
        has been completed.

        For fields located on the host all values of `stream` are interpreted as `BLOCK`.
        """
        ...

    def is_ready(self) -> bool:
        """Check if communication has been finished.

        For an exchange involving device memory, calling this function is equivalent
        to the call `self.finish(stream=BLOCK)`.
        """
        ...


@runtime_checkable
class ExchangeRuntime(Protocol):
    @overload
    def start(
        self,
        dim: gtx.Dimension,
        *buffers: data_alloc.NDArray,
        stream: StreamLike = DEFAULT_STREAM,
    ) -> ExchangeResult: ...

    @overload
    def start(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field,
        stream: StreamLike = DEFAULT_STREAM,
    ) -> ExchangeResult: ...

    @abc.abstractmethod
    def start(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field | data_alloc.NDArray,
        stream: StreamLike = DEFAULT_STREAM,
    ) -> ExchangeResult:
        """Initiate a halo exchanges.

        The exchange will synchronize with `stream`, i.e. not start before all work
        previously submitted to `stream` has finished. `stream` defaults to
        `DEFAULT_STREAM`. To complete the exchange `finish()` must be called on the
        returned `ExchangeResult`. There is also the `exchange()` function which
        combines these two steps into one.

        Note:
            - For fields on the host the exchange will begin immediately, regardless
                which stream has been passed.
            - Depending on the backend the memory of `*fields` might still be accessed
                even after this function returns.
        """
        ...

    @overload
    def exchange(
        self,
        dim: gtx.Dimension,
        *buffers: data_alloc.NDArray,
        stream: StreamLike | BlockType = DEFAULT_STREAM,
    ) -> None: ...

    @overload
    def exchange(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field,
        stream: StreamLike | BlockType = DEFAULT_STREAM,
    ) -> None: ...

    def exchange(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field | data_alloc.NDArray,
        stream: StreamLike | BlockType = DEFAULT_STREAM,
    ) -> None:
        """Perform a full halo exchange.

        The exchange will synchronize with `stream`, i.e. not start before tasks
        previously submitted to it are done. The function returns before the exchange
        has been completed, but it will synchronize with `stream`, i.e. work submitted
        to `stream` will not start before the exchange has finished.

        It is possible to perform the exchange in two steps. The first step is done
        by calling `self.start()`. The second step is done by calling `finish()` on
        the handle returned by `self.start()`.

        In case `stream` is `BLOCK` the function will only return once the exchange
        has been completed entirely and the send part will be performed as if
        `DEFAULT_STREAM` was passed.

        Note:
            The protocol supplies a default implementation.
        """
        ex_req = self.start(
            dim,
            *fields,
            stream=(DEFAULT_STREAM if stream is BLOCK else stream),  # type: ignore[arg-type]
        )
        ex_req.finish(stream)

    @overload
    def __call__(
        self,
        *fields: gtx.Field | data_alloc.NDArray,
        dim: gtx.Dimension,
        full_exchange: Literal[True],
        stream: StreamLike | BlockType = DEFAULT_STREAM,
    ) -> None: ...

    @overload
    def __call__(
        self,
        *fields: gtx.Field | data_alloc.NDArray,
        dim: gtx.Dimension,
        full_exchange: Literal[False],
        stream: StreamLike = DEFAULT_STREAM,
    ) -> ExchangeResult: ...

    def __call__(
        self,
        *fields: gtx.Field | data_alloc.NDArray,
        dim: gtx.Dimension,
        full_exchange: bool = True,
        stream: StreamLike | BlockType = DEFAULT_STREAM,
    ) -> None | ExchangeResult:
        """Performs either a full exchange or a partial exchange.

        If `full_exchange` is `True` then this function is equivalent to
        `self.exchange()` otherwise it behaves as `self.start()` and the exchange
        result object is returned.

        Note:
            - This function is deprecated and should no longer be used.
            - The order of `*fields` and `dim` is reversed compared to `exchange()`.
            - The protocol supplies a default implementation.
        """
        ex_req = self.start(
            dim,
            *fields,
            stream=(DEFAULT_STREAM if stream is BLOCK else stream),  # type: ignore[arg-type]
        )
        if not full_exchange:
            return ex_req
        ex_req.finish(stream=stream)
        return None

    def get_size(self) -> int: ...

    def my_rank(self) -> int: ...

    def __str__(self) -> str:
        return f"{self.__class__} (rank = {self.my_rank()} / {self.get_size()})"


@dataclasses.dataclass
class SingleNodeExchange(ExchangeRuntime):
    _warning_emitted: ClassVar[bool] = False

    @classmethod
    def _warn_if_used(cls, *, stacklevel: int = 3) -> None:
        """
        Debugging helper: emit a single warning when SingleNodeExchange is used.
        Not called in production code.  Enable manually during development to
        spot unintended single-node exchange usage.
        """
        if not cls._warning_emitted:
            warnings.warn(
                "***** SingleNodeExchange is in use; HALO EXCHANGE IS RUNNING IN SINGLE-NODE *****",
                RuntimeWarning,
                stacklevel=stacklevel,
            )
            cls._warning_emitted = True

    def start(
        self,
        dim: gtx.Dimension,
        *fields: gtx.Field | data_alloc.NDArray,
        stream: StreamLike = DEFAULT_STREAM,
    ) -> ExchangeResult:
        return SingleNodeResult()

    def my_rank(self) -> int:
        return 0

    def get_size(self) -> int:
        return 1


class HaloExchangeWaitRuntime(Protocol):
    """Protocol for halo exchange wait."""

    def __call__(
        self,
        communication_handle: ExchangeResult,
        stream: StreamLike | BlockType = DEFAULT_STREAM,
    ) -> None:
        """Calls `finish()` on the provided communication handle.

        Args:
            stream: The stream forwarded to the `wait()` call, defaults to `DEFAULT_STREAM`.
        """
        # NOTE: Refactor the code such that there should be a default implementation
        #   of this function or such that it is no longer needed.
        ...


@dataclasses.dataclass
class HaloExchangeWait(HaloExchangeWaitRuntime):
    exchange_object: SingleNodeExchange

    def __call__(
        self,
        communication_handle: ExchangeResult,
        stream: StreamLike | BlockType = DEFAULT_STREAM,
    ) -> None:
        communication_handle.finish(stream=stream)


@functools.singledispatch
def create_halo_exchange_wait(runtime: ExchangeRuntime) -> HaloExchangeWaitRuntime:
    raise TypeError(f"Unknown ExchangeRuntime type ({type(runtime)})")


@create_halo_exchange_wait.register(SingleNodeExchange)
def create_single_node_halo_exchange_wait(runtime: SingleNodeExchange) -> HaloExchangeWait:
    return HaloExchangeWait(runtime)


class SingleNodeResult(ExchangeResult):
    def finish(self, stream: StreamLike | BlockType = DEFAULT_STREAM) -> None:
        pass

    def is_ready(self) -> bool:
        return True


class RunType:
    """Base type for marker types used to initialize the parallel or single node properties."""

    pass


class MultiNodeRun(RunType):
    """
    Mark multinode run.

    Dummy marker type used to initialize a multinode run and initialize
    construction multinode ProcessProperties.
    """

    pass


class SingleNodeRun(RunType):
    """
    Mark single node run.

    Dummy marker type used to initialize a single node run and initialize
    construction SingleNodeProcessProperties.
    """

    pass


class Reductions(Protocol):
    """
    Protocol for global reduction operations across distributed ranks.
    """

    def min(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType: ...

    def max(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType: ...

    def sum(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType: ...

    def mean(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType: ...


class SingleNodeReductions(Reductions):
    """Reductions for single-rank (non-distributed) runs.

    Single-rank runs have no halos, so all elements are owned and no masking
    is needed. The entire buffer is reduced directly.
    """

    def min(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType:
        array_ns = data_alloc.array_namespace(buffer)
        return array_ns.min(buffer).item()

    def max(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType:
        array_ns = data_alloc.array_namespace(buffer)
        return array_ns.max(buffer).item()

    def sum(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType:
        array_ns = data_alloc.array_namespace(buffer)
        return array_ns.sum(buffer).item()

    def mean(self, buffer: data_alloc.NDArray) -> state_utils.ScalarType:
        array_ns = data_alloc.array_namespace(buffer)
        return array_ns.sum(buffer).item() / buffer.size


def get_runtype(with_mpi: bool = False) -> RunType:
    if with_mpi:
        return MultiNodeRun()
    else:
        return SingleNodeRun()


@functools.singledispatch
def get_process_properties(runtime: RunType, comm_id: int | None = None) -> ProcessProperties:
    raise TypeError(f"Cannot define ProcessProperties for ({type(runtime)})")


@get_process_properties.register(SingleNodeRun)
def get_single_node_properties(s: SingleNodeRun, comm_id: int | None = None) -> ProcessProperties:
    return SingleNodeProcessProperties()


@functools.singledispatch
def create_exchange(
    process_props: ProcessProperties, decomp_info: DecompositionInfo
) -> ExchangeRuntime:
    """
    Create an Exchange depending on the runtime size.

    Depending on the number of processor a SingleNode version is returned or a GHEX context created and a Multinode returned.
    """
    raise NotImplementedError(f"Unknown ProcessProperties type ({type(process_props)})")


@create_exchange.register(SingleNodeProcessProperties)
def create_single_node_exchange(
    process_props: SingleNodeProcessProperties, decomp_info: DecompositionInfo
) -> ExchangeRuntime:
    return SingleNodeExchange()


@functools.singledispatch
def create_reduction(
    process_props: ProcessProperties, decomposition_info: DecompositionInfo
) -> Reductions:
    """
    Create a Global Reduction depending on the runtime size.

    Depending on the number of processor a SingleNode version is returned or a GHEX context created and a Multinode returned.
    """
    raise NotImplementedError(f"Unknown ProcessProperties type ({type(process_props)})")


@create_reduction.register(SingleNodeProcessProperties)
def create_single_reduction_exchange(
    process_props: SingleNodeProcessProperties, decomposition_info: DecompositionInfo
) -> Reductions:
    return SingleNodeReductions()


class DecompositionFlag(int, Enum):
    UNDEFINED = -1
    OWNED = 0
    """used for locally owned cells, vertices, edges"""

    FIRST_HALO_LEVEL = 1
    """
    used for:
    - cells that share 1 edge with an OWNED cell
    - vertices that are on OWNED cell, but not owned (because they belong to the neighboring rank)
    - edges that are on OWNED cell, but not owned (because they belong to the neighboring rank)
    """

    SECOND_HALO_LEVEL = 2
    """
    used for:
    - cells that share one vertex with an OWNED cell
    - vertices that are on a cell(FIRST_HALO_LEVEL) but not on an owned cell
    - edges that have _exactly_ one vertex shared with and OWNED Cell
    """

    THIRD_HALO_LEVEL = 3
    """
    This type does not exist in ICON. It denotes the "closing/far" edges of the SECOND_HALO_LINE cells
    used for:
    - cells (NOT USED)
    - vertices (NOT USED)
    - edges that are only on the cell(SECOND_HALO_LEVEL)
    """


class ParallelLogger(logging.Filter):
    def __init__(
        self,
        process_properties: ProcessProperties | None = None,
        print_distributed_debug_msg: bool = False,
    ) -> None:
        super().__init__()
        self._rank_info_str = ""
        self._print_distributed_debug_msg = print_distributed_debug_msg
        self._rank_id = 0
        if process_properties and process_properties.comm_size > 1:
            self._rank_info_str = f"rank={process_properties.rank}/{process_properties.comm_size} [{process_properties.comm_name}] "
            self._rank_id = process_properties.rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank_info_str = self._rank_info_str
        if self._rank_id == 0:
            return True
        return record.levelno >= logging.WARNING or (
            self._print_distributed_debug_msg and record.levelno == logging.DEBUG
        )


single_node_exchange = SingleNodeExchange()
single_node_reductions = SingleNodeReductions()
