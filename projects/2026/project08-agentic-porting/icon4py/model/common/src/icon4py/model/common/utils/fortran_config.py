# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import Any, Final


NAMELIST_INPUT_FNAME: Final = "NAMELIST_expname"
NAMELIST_ATM_FNAME: Final = "NAMELIST_ICON_output_atm"
NAMELIST_MASTER_FNAME: Final = "icon_master.namelist"

ATM_DICT_FNAME: Final = f"{NAMELIST_ATM_FNAME}.json"
MASTER_DICT_FNAME: Final = f"{NAMELIST_MASTER_FNAME}.json"
INPUT_DICT_FNAME: Final = f"{NAMELIST_INPUT_FNAME}.json"

SER_DATA_SUBDIR: Final = "ser_data"


def list_to_value[T](obj: list[T] | T) -> T:
    # Some parameters are allocated as `max_dom`-sized lists, with one value
    # per domain. ICON4Py (for now) only runs on one domain.
    # Most parameters have the same value for all elements, others (such as
    # num_levels) have a default value different from domain[0].
    # TODO (ricoh,jcanton): stop using this for per-tracer values when enabling
    # that functionality Tracers are an even different case where there is one
    # value per tracer, but with the current version of ICON4Py all tracers get
    # the same config.
    return obj[0] if isinstance(obj, list) else obj


def _translate_fields(
    source: dict[str, Any],
    name_map: dict[str, str],
    known_fields: set[str],
) -> dict[str, Any]:
    """Map Fortran namelist keys to Python field names, keeping only known dataclass fields."""
    params: dict[str, Any] = {}
    for key, value in source.items():
        python_name = name_map.get(key, key)
        if python_name in known_fields:
            params[python_name] = value
    return params


def config_dataclass_from_dict[T](cls: type[T], source: dict[str, Any]) -> T:
    """Construct a dataclass from a Fortran namelist dict.

    This is used by the topography and initial_condition Config classes which
    contain part (sometimes renamed) of the fortran namelist parameters.

    Unknown keys are ignored (e.g. topography params mixed into the same nml block).
    Missing keys fall back to the dataclass field defaults.
    Fortran→Python name translation is driven by the required ``fortran_name_map``
    class variable: ``{fortran_key: python_field_name}``.
    """
    name_map: dict[str, str] = cls.fortran_name_map  # type: ignore[attr-defined]
    known_fields = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    kwargs = _translate_fields(source, name_map, known_fields)
    return cls(**kwargs)
