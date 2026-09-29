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

from icon4py.model.common.initial_condition import from_file as from_file_ic
from icon4py.model.common.initial_condition.analytical import (
    gauss3d as gauss_ic,
    jablonowski_williamson as jw_ic,
)
from icon4py.model.common.utils import fortran_config


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition_defs
    from icon4py.model.common.grid import icon as icon_grid, vertical as v_grid
    from icon4py.model.common.states import prognostic_state as prognostics, static_fields

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InitialConditionConfig:
    config: jw_ic.JablonowskiWilliamsonConfig | gauss_ic.Gauss3DConfig | from_file_ic.FromFileConfig

    @classmethod
    def from_fortran_dict(
        cls,
        *,
        atm_dict: dict[str, Any],
        input_dict: dict[str, Any],
        data_path: pathlib.Path,
    ) -> InitialConditionConfig:
        run_nml = atm_dict.get("run_nml", {})
        if not run_nml.get("ltestcase", False):
            ntracer = fortran_config.list_to_value(run_nml.get("ntracer", 0))
            log.info("Reading initial condition from file")
            return cls(
                config=from_file_ic.FromFileConfig(
                    data_path=data_path / fortran_config.SER_DATA_SUBDIR,
                    ntracer=ntracer,
                ),
            )

        testcase_nml = input_dict.get("nh_testcase_nml", {})
        test_name = testcase_nml.get("nh_test_name")
        config: (
            jw_ic.JablonowskiWilliamsonConfig | gauss_ic.Gauss3DConfig
        )  # mypy does not automatically catch type
        match test_name:
            case "jabw" | "jabw_s" | "APE_nwp" | "APE_aes":
                log.info("Analytical initial condition for Jablonowski-Williamson test case")
                config = fortran_config.config_dataclass_from_dict(
                    jw_ic.JablonowskiWilliamsonConfig, testcase_nml
                )
                # Only the APE cases rescale qv to a prescribed global moisture content.
                config.normalize_global_moisture = test_name in ("APE_nwp", "APE_aes")
                # Fortran resets jw_up to 0 only for jabw_s; other cases keep the default (1.0).
                if test_name == "jabw_s":
                    config.baroclinic_amplitude = 0.0
            case (
                "gauss3D" | "wk82"
            ):  # TODO (jcanton): wk82 is just a placeholder until next PR, it is not actually used
                log.info("Analytical initial condition for Gauss 3D test case")
                config = fortran_config.config_dataclass_from_dict(
                    gauss_ic.Gauss3DConfig, testcase_nml
                )
            case name:
                raise ValueError(f"Unknown or missing test case name: {name!r}")

        return cls(config=config)


def create(
    *,
    config: InitialConditionConfig,
    vertical_config: v_grid.VerticalGridConfig,
    grid: icon_grid.IconGrid,
    static_fields: static_fields.StaticFieldFactories,
    prognostic_state_now: prognostics.PrognosticState,
    backend: gtx_typing.Backend | None,
    exchange: decomposition_defs.ExchangeRuntime,
    global_reductions: decomposition_defs.Reductions,
) -> None:
    """Fill a PrognosticState by dispatching on the type of ``config.config``."""
    match config.config:
        case jw_ic.JablonowskiWilliamsonConfig():
            jw_ic.jablonowski_williamson(
                config=config.config,
                vertical_config=vertical_config,
                grid=grid,
                static_fields=static_fields,
                prognostic_state_now=prognostic_state_now,
                backend=backend,
                exchange=exchange,
                global_reductions=global_reductions,
            )
        case gauss_ic.Gauss3DConfig():
            gauss_ic.gauss3d(
                config=config.config,
                vertical_config=vertical_config,
                grid=grid,
                static_fields=static_fields,
                prognostic_state_now=prognostic_state_now,
                backend=backend,
                exchange=exchange,
            )
        case from_file_ic.FromFileConfig():
            from_file_ic.read_from_file(
                config=config.config,
                grid=grid,
                prognostic_state_now=prognostic_state_now,
                backend=backend,
                exchange=exchange,
            )
        case _:
            raise TypeError(f"Unknown initial conditions config type: {type(config.config)!r}")
