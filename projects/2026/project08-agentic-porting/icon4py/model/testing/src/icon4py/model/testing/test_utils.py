# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import os
from collections.abc import Buffer
from typing import Any

import gt4py.next.typing as gtx_typing
import numpy as np
import numpy.testing as np_testing
import numpy.typing as npt
import pytest

from icon4py.model.common import model_backends, model_options
from icon4py.model.testing import config


def get_mpi_comparison_tolerance(
    backend: gtx_typing.Backend | None,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    """Return (atol, rtol) for single-rank vs multi-rank field comparisons.

    When ``ICON4PY_TEST_EXPECT_MPI_REPRODUCIBLE`` is set and the backend is a
    known-good CPU backend (``gtfn`` or ``dace``), tolerances are overridden to
    zero for bitwise-exact comparison. Otherwise the caller-supplied tolerances
    are returned unchanged.
    """
    if os.environ.get("ICON4PY_TEST_EXPECT_MPI_REPRODUCIBLE") == "1" and (
        model_backends.is_cpu_backend(backend) and (is_gtfn_backend(backend) or is_dace(backend))
    ):
        return 0.0, 0.0
    return atol, rtol


def dallclose(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    rtol: float = 1.0e-12,
    atol: float = 0.0,
    equal_nan: bool = False,
) -> bool:
    """
    'numpy.allclose', but with double precision default tolerances.
    """
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def assert_dallclose(
    actual: npt.ArrayLike,
    desired: npt.ArrayLike,
    *,
    rtol: float = 1.0e-12,
    atol: float = 0.0,
    equal_nan: bool = False,
    err_msg: str = "",
    verbose: bool = True,
) -> None:
    """
    'numpy.testing.assert_allclose', but with double precision default tolerances.
    """
    if config.DALLCLOSE_PRINT_INSTEAD_OF_FAIL:
        # Non-blocking version: prints max diff instead of raising errors.
        # Prints red if delta > 0, green otherwise.
        max_diff = np.max(np.abs(np.asarray(actual) - np.asarray(desired)))
        color = "\033[1;31m" if max_diff > 0 else "\033[32m"
        print(f"{color}{err_msg} max diff {max_diff}\033[0m")
    else:
        np_testing.assert_allclose(
            actual,  # type: ignore[arg-type]
            desired,  # type: ignore[arg-type]
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            err_msg=err_msg,
            verbose=verbose,
        )


def is_sorted(array: np.ndarray) -> bool:
    return bool((array[:-1] <= array[1:]).all())


def fingerprint_buffer(buffer: Buffer, *, digest_length: int = 8) -> str:
    return hashlib.md5(np.asarray(buffer, order="C")).hexdigest()[-digest_length:]


def get_fixture_value(name: str, item: pytest.Item) -> Any:
    if name in item.fixturenames:  # type: ignore[attr-defined]
        # Get the fixture value using the item's fixture manager
        return item._request.getfixturevalue(name)  # type: ignore[attr-defined]

    return None


def get_backend_fixture_value(item: pytest.Item) -> gtx_typing.Backend | None:
    backend = get_fixture_value("backend", item)
    if backend is not None:
        return backend
    backend_like = get_fixture_value("backend_like", item)
    if backend_like is not None:
        return model_options.customize_backend(None, backend_like)
    return None


def is_embedded(backend: gtx_typing.Backend | None) -> bool:
    return backend is None


def is_roundtrip(backend: gtx_typing.Backend | None) -> bool:
    return backend.name == "roundtrip" if backend else False


def is_python(backend: gtx_typing.Backend | None) -> bool:
    # want to exclude python backends:
    #   - cannot run on embedded: because of slicing
    #   - roundtrip is very slow on large grid
    return is_embedded(backend) or is_roundtrip(backend)


def is_dace(backend: gtx_typing.Backend | None) -> bool:
    return backend.name.startswith("run_dace_") if backend else False


def is_gtfn_backend(backend: gtx_typing.Backend | None) -> bool:
    return "gtfn" in backend.name if backend else False
