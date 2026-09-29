# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
from collections.abc import Callable
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next import backend as gtx_backend
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from icon4py.model.common import model_backends


log = logging.getLogger(__name__)


def dict_values_to_list(d: dict[str, Any]) -> dict[str, list]:
    return {k: [v] for k, v in d.items()}


def get_dace_options(
    program_name: str, **backend_descriptor: Any
) -> model_backends.BackendDescriptor:
    is_rocm_device = backend_descriptor.get("device") == model_backends.DeviceType.ROCM
    optimization_args = backend_descriptor.get("optimization_args", {})
    optimization_hooks = optimization_args.get("optimization_hooks", {})
    if program_name in [
        "vertically_implicit_solver_at_corrector_step",
        "vertically_implicit_solver_at_predictor_step",
    ]:
        if gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowStep not in optimization_hooks:
            # Enable pass that removes access node (next_w) copies for vertically implicit solver programs
            optimization_hooks[gtx_transformations.GT4PyAutoOptHook.TopLevelDataFlowStep] = (
                lambda sdfg: sdfg.apply_transformations_repeated(
                    gtx_transformations.RemoveAccessNodeCopies(),
                    validate=False,
                    validate_all=False,
                )
            )
        if "scan_loop_unrolling" not in optimization_args:
            optimization_args["scan_loop_unrolling"] = True
        if "scan_loop_unrolling_factor" not in optimization_args:
            optimization_args["scan_loop_unrolling_factor"] = 0
    # TODO(havogt): Eventually the option `use_zero_origin` should be removed and the default behavior should be `use_zero_origin=False`.
    # We keep it `True` for 'compute_rho_theta_pgrad_and_update_vn' as performance drops,
    # due to it falling into a less optimized code generation (on santis).
    if program_name == "compute_rho_theta_pgrad_and_update_vn":
        backend_descriptor["use_zero_origin"] = True
    if program_name == "graupel_run":
        optimization_args["fuse_tasklets"] = True
        if not is_rocm_device:
            optimization_args["gpu_maxnreg"] = 80
            optimization_args["gpu_block_size_2d"] = (64, 6)
        optimization_args["gpu_memory_pool"] = False
        optimization_args["make_persistent"] = True
    if optimization_hooks:
        optimization_args["optimization_hooks"] = optimization_hooks
    if optimization_args:
        backend_descriptor["optimization_args"] = optimization_args
    return backend_descriptor


def get_gtfn_options(
    program_name: str, **backend_descriptor: Any
) -> model_backends.BackendDescriptor:
    return backend_descriptor


def get_options(program_name: str, **backend_descriptor: Any) -> model_backends.BackendDescriptor:
    if "backend_factory" not in backend_descriptor:
        # here we could set a backend_factory per program
        backend_descriptor["backend_factory"] = model_backends.make_custom_dace_backend
    if backend_descriptor["backend_factory"] == model_backends.make_custom_dace_backend:
        backend_descriptor = get_dace_options(program_name, **backend_descriptor)
    if backend_descriptor["backend_factory"] == model_backends.make_custom_gtfn_backend:
        backend_descriptor = get_gtfn_options(program_name, **backend_descriptor)

    return backend_descriptor


def customize_backend(
    program: gtx_typing.Program | gtx.typing.FieldOperator | None,
    backend: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
) -> gtx_typing.Backend | None:
    program_name = program.__name__ if program is not None else ""
    if backend is None or isinstance(backend, gtx_backend.Backend):
        backend_name = backend.name if backend is not None else "embedded"
        log.info(f"Using non-custom backend '{backend_name}' for '{program_name}'.")
        return backend  # type: ignore[return-value]

    backend_descriptor = (
        {"device": backend} if isinstance(backend, model_backends.DeviceType) else backend
    )
    backend_descriptor = get_options(program_name, **backend_descriptor)
    backend_descriptor["device"] = backend_descriptor.get(
        "device", model_backends.CPU
    )  # set default device
    backend_factory = backend_descriptor.pop(
        "backend_factory", model_backends.make_custom_dace_backend
    )
    custom_backend = backend_factory(**backend_descriptor)
    log.info(
        f"Using custom backend '{custom_backend.name}' for '{program_name}' with options: {backend_descriptor}."
    )
    return custom_backend


def setup_program(
    *,
    program: gtx_typing.Program,
    backend: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
    constant_args: dict[str, gtx.Field | gtx_typing.Scalar] | None = None,
    variants: dict[str, list[gtx_typing.Scalar]] | None = None,
    horizontal_sizes: dict[str, gtx.int32] | None = None,
    vertical_sizes: dict[str, gtx.int32] | None = None,
    offset_provider: gtx_typing.OffsetProvider | None = None,
) -> Callable[..., None]:
    """
    This function processes arguments to the GT4Py program. It
    - binds arguments that don't change during model run ('constant_args', 'horizontal_sizes', "vertical_sizes');
    - inlines scalar arguments into the GT4Py program at compile-time (via GT4Py's 'compile').
    Args:
        - backend: GT4Py backend,
        - program: GT4Py program,
        - constant_args: constant fields and scalars,
        - variants: list of all scalars potential values from which one is selected at run time,
        - horizontal_sizes: horizontal domain bounds,
        - vertical_sizes: vertical domain bounds,
        - offset_provider: GT4Py offset_provider,
    """
    constant_args = {} if constant_args is None else constant_args
    variants = {} if variants is None else variants
    horizontal_sizes = {} if horizontal_sizes is None else horizontal_sizes
    vertical_sizes = {} if vertical_sizes is None else vertical_sizes
    offset_provider = {} if offset_provider is None else offset_provider

    backend = customize_backend(program, backend)

    bound_static_args = {k: v for k, v in constant_args.items() if gtx.is_scalar_type(v)}
    static_args_program = program.with_backend(backend)
    if backend is not None:
        static_args_program = static_args_program.with_compilation_options(enable_jit=False)
        static_args_program.compile(
            **dict_values_to_list(horizontal_sizes),
            **dict_values_to_list(vertical_sizes),
            **variants,
            **dict_values_to_list(bound_static_args),
            offset_provider=offset_provider,
        )

    return functools.partial(
        static_args_program,
        **constant_args,
        **horizontal_sizes,
        **vertical_sizes,
        offset_provider=offset_provider,
    )
