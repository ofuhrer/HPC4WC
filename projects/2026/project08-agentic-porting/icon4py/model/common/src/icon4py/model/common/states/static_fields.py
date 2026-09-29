# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple


if TYPE_CHECKING:
    from icon4py.model.common.grid import geometry as grid_geometry
    from icon4py.model.common.interpolation import interpolation_factory
    from icon4py.model.common.metrics import metrics_factory


class StaticFieldFactories(NamedTuple):
    """
    Factories of static fields for the driver and components.

    Attributes:
        geometry: grid geometry field factory that stores geometrical properties of a grid
        interpolation: interpolation field factory that stores pre-computed coefficients for interpolation employed in the model
        metrics: metrics field factory that stores pre-computed coefficients for numerical operations employed in the model
    """

    geometry: grid_geometry.GridGeometry
    interpolation: interpolation_factory.InterpolationFieldsFactory
    metrics: metrics_factory.MetricsFieldsFactory
