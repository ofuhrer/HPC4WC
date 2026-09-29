# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#

"""Shared helpers for Python dev-scripts."""

from __future__ import annotations

import pathlib
import subprocess
from typing import Final

import rich
import typer


PY_SCRIPTS_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().absolute().parent.parent
SCRIPTS_DIR: Final[pathlib.Path] = PY_SCRIPTS_DIR.parent
REPO_ROOT: Final[pathlib.Path] = SCRIPTS_DIR.parent


def run_or_fail(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[bytes]:
    """
    Run *cmd* and exit with its return code on failure.

    All *kwargs* are forwarded to :func:`subprocess.run`.
    """
    try:
        result = subprocess.run(cmd, **kwargs, check=True)
    except subprocess.CalledProcessError as e:
        rich.print(
            f"[bold red]Error[/bold red] Command failed (rc={e.returncode}): {' '.join(cmd)}",
        )
        raise typer.Exit(e.returncode) from e
    return result
