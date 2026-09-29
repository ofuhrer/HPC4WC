# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import TYPE_CHECKING, Any

from icon4py.model.common.topography import from_file as from_file_topo
from icon4py.model.common.topography.analytical import (
    flat_topography as flat_topo,
    gaussian_hill as gausshill_topo,
    jablonowski_williamson as jw_topo,
)
from icon4py.model.common.utils import data_allocation as data_alloc, fortran_config


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition_defs
    from icon4py.model.common.grid import grid_manager as gm

log = logging.getLogger(__name__)


@dataclasses.dataclass
class TopographyConfig:
    config: (
        flat_topo.FlatTopographyConfig
        | jw_topo.JablonowskiWilliamsonConfig
        | gausshill_topo.GaussianHillConfig
        | from_file_topo.FromFileConfig
    )

    @classmethod
    def from_fortran_dict(
        cls,
        *,
        atm_dict: dict[str, Any],
        input_dict: dict[str, Any],
        data_path: pathlib.Path,
    ) -> TopographyConfig:
        run_nml = atm_dict.get("run_nml", {})
        if not run_nml.get("ltestcase", False):
            log.info("Reading topography from file")
            return cls(
                config=from_file_topo.FromFileConfig(
                    data_path=data_path / fortran_config.SER_DATA_SUBDIR,
                ),
            )

        testcase_nml = input_dict.get("nh_testcase_nml", {})
        test_name = testcase_nml.get("nh_test_name")
        config: (
            flat_topo.FlatTopographyConfig
            | jw_topo.JablonowskiWilliamsonConfig
            | gausshill_topo.GaussianHillConfig
        )  # mypy does not automatically catch type
        match test_name:
            case "APE_nwp" | "APE_aes" | "wk82":
                log.info("Flat topography")
                config = flat_topo.FlatTopographyConfig()
            case "jabw" | "jabw_s":
                log.info("Analytical topography for Jablonowski-Williamson test case")
                config = fortran_config.config_dataclass_from_dict(
                    jw_topo.JablonowskiWilliamsonConfig, testcase_nml
                )
            case "gauss3D":
                log.info("Analytical Gaussian hill topography")
                config = fortran_config.config_dataclass_from_dict(
                    gausshill_topo.GaussianHillConfig, testcase_nml
                )
            case name:
                raise ValueError(f"Unknown or missing test case name: {name!r}")

        return cls(config)


def create(
    *,
    config: TopographyConfig,
    grid_manager: gm.GridManager,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
) -> data_alloc.NDArray:
    """Create topography array by dispatching on the type of ``config.config``."""
    match config.config:
        case flat_topo.FlatTopographyConfig():
            return flat_topo.flat_topography(grid_manager=grid_manager)
        case jw_topo.JablonowskiWilliamsonConfig():
            return jw_topo.jablonowski_williamson(config=config.config, grid_manager=grid_manager)
        case gausshill_topo.GaussianHillConfig():
            return gausshill_topo.gaussian_hill(config=config.config, grid_manager=grid_manager)
        case from_file_topo.FromFileConfig():
            return from_file_topo.read_from_file(
                config=config.config,
                grid_manager=grid_manager,
                backend=backend,
                exchange=exchange,
            )
        case _:
            raise TypeError(f"Unknown topography config type: {type(config.config)!r}")
