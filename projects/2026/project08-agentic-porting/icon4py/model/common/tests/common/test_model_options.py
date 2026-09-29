# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import typing

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common import field_type_aliases as fa, model_backends
from icon4py.model.common.model_options import customize_backend, setup_program


@gtx.field_operator  # type: ignore[call-overload]
def field_op_return_field(field: fa.CellKField[float], factor: float) -> fa.CellKField[float]:
    return field + factor


@gtx.program  # type: ignore[call-overload]
def program_return_field(field: fa.CellKField[float], factor: float):  # type: ignore[no-untyped-def]
    field_op_return_field(field, factor, out=field)


@pytest.mark.parametrize(
    "backend_factory, expected_backend",
    [
        (
            model_backends.make_custom_gtfn_backend,
            model_backends.make_custom_gtfn_backend(device=model_backends.CPU),
        ),
        (
            model_backends.make_custom_dace_backend,
            model_backends.make_custom_dace_backend(device=model_backends.CPU),
        ),
    ],
)
def test_custom_backend_options(backend_factory: typing.Callable, expected_backend: str) -> None:
    backend_options: dict = {
        "backend_factory": backend_factory,
        "device": model_backends.CPU,
    }
    backend = customize_backend(None, backend_options)
    # TODO(havogt): test should be improved to work without string comparison
    assert repr(expected_backend) == repr(backend)


def test_custom_backend_device() -> None:
    device = model_backends.CPU
    backend = customize_backend(None, device)
    default_backend = model_backends.make_custom_dace_backend(device=device)
    # TODO(havogt): test should be improved to work without string comparison
    assert repr(default_backend) == repr(backend)


@pytest.mark.parametrize(
    "backend",
    [
        model_backends.make_custom_dace_backend(device=model_backends.CPU),  # conrete backend
        model_backends.CPU,
        {"backend_factory": model_backends.make_custom_dace_backend, "device": model_backends.CPU},
        {"backend_factory": model_backends.make_custom_dace_backend},
        {"device": model_backends.CPU},
    ],
)
def test_setup_program_defaults(
    backend: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
) -> None:
    testee = setup_program(backend=backend, program=program_return_field)
    expected_backend = model_backends.make_custom_dace_backend(device=model_backends.CPU)
    expected_program = functools.partial(
        program_return_field.with_backend(expected_backend)
        .with_compilation_options(enable_jit=False)
        .compile(
            offset_provider={},
        ),
        offset_provider={},
    )
    # TODO(havogt): test should be improved to work without string comparison
    assert repr(testee) == repr(expected_program)


@pytest.mark.parametrize(
    "backend_params, expected_backend",
    [
        (model_backends.BACKENDS["embedded"], model_backends.BACKENDS["embedded"]),
        (
            {
                "backend_factory": model_backends.make_custom_dace_backend,
                "device": model_backends.GPU,
            },
            model_backends.make_custom_dace_backend(device=model_backends.GPU),
        ),
        (
            {"backend_factory": model_backends.make_custom_dace_backend},
            model_backends.make_custom_dace_backend(device=model_backends.CPU),
        ),
        (
            {"device": model_backends.GPU},
            model_backends.make_custom_dace_backend(device=model_backends.GPU),
        ),
    ],
)
def test_setup_program_specify_inputs(
    backend_params: gtx_typing.Backend
    | model_backends.DeviceType
    | model_backends.BackendDescriptor
    | None,
    expected_backend: gtx_typing.Backend | None,
) -> None:
    testee = setup_program(backend=backend_params, program=program_return_field)
    if expected_backend is None:
        expected_program = functools.partial(
            program_return_field.with_backend(expected_backend), offset_provider={}
        )
    else:
        expected_program = functools.partial(
            program_return_field.with_backend(expected_backend)
            .with_compilation_options(enable_jit=False)
            .compile(
                offset_provider={},
            ),
            offset_provider={},
        )
    # TODO(havogt): test should be improved to work without string comparison
    assert repr(testee) == repr(expected_program)
