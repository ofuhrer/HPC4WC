# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import pathlib

import filelock


# Consider moving to common if this is needed outside of testing


def lock(
    directory: pathlib.Path | str, lockfile: str = "filelock.lock"
) -> contextlib.AbstractContextManager:
    """Create a lock for the given path."""
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory, got: {directory}")

    path = directory / lockfile
    return filelock.FileLock(str(path))
