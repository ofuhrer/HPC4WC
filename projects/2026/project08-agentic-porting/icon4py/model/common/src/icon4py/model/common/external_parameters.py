# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from icon4py.model.common import field_type_aliases as fa


@dataclasses.dataclass
class ExternalParameters:
    """Dataclass containing external parameters."""

    topo_c: fa.CellField[float]
    topo_smt_c: fa.CellField[float]
