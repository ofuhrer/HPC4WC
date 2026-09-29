# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import time as _time


try:
    import cupy as cp  # type: ignore[import-not-found]

    def _sync() -> None:
        cp.cuda.runtime.deviceSynchronize()
except ImportError:
    cp = None

    def _sync() -> None:
        pass


def perf_counter() -> float:
    _sync()
    return _time.perf_counter()
