#!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --group scripts python3
#
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Handle validation grid files for testing and benchmarking."""

from __future__ import annotations

import hashlib
import sys
from typing import TYPE_CHECKING

import typer


if TYPE_CHECKING:
    from icon4py.model.testing import definitions


cli = typer.Typer(
    name=__name__.split(".")[-1].replace("_", "-"), no_args_is_help=True, help=__doc__
)


def get_validation_grids() -> list[definitions.GridDescription]:
    # Import here to reduce startup time of the CLI
    from icon4py.model.testing import definitions  # noqa: PLC0415 [import-outside-top-level]

    return [
        definitions.Grids.R01B01_GLOBAL,
        definitions.Grids.R02B04_GLOBAL,
        definitions.Grids.MCH_CH_R04B09_DSL,
        definitions.Grids.MCH_OPR_R04B07_DOMAIN01,
        definitions.Grids.TORUS_50000x5000,
    ]  # change to MCH_OPR_R04B07_DOMAIN01


@cli.command(name="cache-key")
def cache_key() -> None:
    """Generate a cache key for the GitHub action cache based on grid file name and download URI."""

    from icon4py.model.testing import (  # noqa: PLC0415 [import-outside-top-level]
        datatest_utils as dt_utils,
        definitions,
    )

    d = "_".join(
        grid.name + dt_utils.get_grid_archive_url(definitions.TESTDATA_ROOT_URL, grid)
        for grid in get_validation_grids()
    )
    hexdigest = hashlib.md5(d.encode()).hexdigest()
    print(hexdigest)


@cli.command(name="download")
def download_validation_grids() -> None:
    """Effectively download the validation grid files."""
    from icon4py.model.testing import grid_utils  # noqa: PLC0415 [import-outside-top-level]

    for grid in get_validation_grids():
        print(f"downloading and unpacking {grid.name}")
        fname = grid_utils._download_grid_file(grid)
        print(f"done - downloaded {fname}")


if __name__ == "__main__":
    sys.exit(cli())
