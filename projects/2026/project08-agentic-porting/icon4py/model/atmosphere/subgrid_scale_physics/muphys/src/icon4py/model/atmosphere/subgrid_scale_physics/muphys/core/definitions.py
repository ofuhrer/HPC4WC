# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import NamedTuple

from icon4py.model.common import field_type_aliases as fa, type_alias as ta


class Q(NamedTuple):
    v: fa.CellKField[ta.wpfloat]  # Specific humidity
    c: fa.CellKField[ta.wpfloat]  # Specific cloud water content
    r: fa.CellKField[ta.wpfloat]  # Specific rain water
    s: fa.CellKField[ta.wpfloat]  # Specific snow water
    i: fa.CellKField[ta.wpfloat]  # Specific ice water content
    g: fa.CellKField[ta.wpfloat]  # Specific graupel water content


class Q_scalar(NamedTuple):
    v: ta.wpfloat  # Specific humidity
    c: ta.wpfloat  # Specific cloud water content
    r: ta.wpfloat  # Specific rain water
    s: ta.wpfloat  # Specific snow water
    i: ta.wpfloat  # Specific ice water content
    g: ta.wpfloat  # Specific graupel water content
