# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import pytest

from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.utils import device_utils
from icon4py.model.testing import test_utils


type Predicate[T] = Callable[[T], bool]


class ItemFilter(NamedTuple):
    """Test item filter definition."""

    condition: Predicate[pytest.Item]
    action: Callable[[], None]


item_marker_filters: dict[str, ItemFilter] = {
    pytest.mark.cpu_only.name: ItemFilter(
        condition=lambda item: device_utils.is_cupy_device(
            test_utils.get_backend_fixture_value(item)
        ),
        action=functools.partial(pytest.skip, "currently only runs on CPU"),
    ),
    pytest.mark.embedded_only.name: ItemFilter(
        condition=lambda item: (
            not test_utils.is_embedded(test_utils.get_backend_fixture_value(item))
        ),
        action=functools.partial(pytest.skip, "stencil runs only on embedded backend"),
    ),
    pytest.mark.embedded_remap_error.name: ItemFilter(
        condition=lambda item: test_utils.is_embedded(test_utils.get_backend_fixture_value(item)),
        action=functools.partial(
            pytest.xfail, "Embedded backend currently fails in remap function."
        ),
    ),
    pytest.mark.embedded_static_args.name: ItemFilter(
        condition=lambda item: test_utils.is_embedded(test_utils.get_backend_fixture_value(item)),
        action=functools.partial(
            pytest.xfail, " gt4py _compiled_programs returns error when backend is None."
        ),
    ),
    pytest.mark.uses_concat_where.name: ItemFilter(
        condition=lambda item: test_utils.is_embedded(test_utils.get_backend_fixture_value(item)),
        action=functools.partial(pytest.xfail, "Embedded backend does not support concat_where."),
    ),
    pytest.mark.gtfn_too_slow.name: ItemFilter(
        condition=lambda item: test_utils.is_gtfn_backend(
            test_utils.get_backend_fixture_value(item)
        ),
        action=functools.partial(pytest.skip, "GTFN compilation is too slow for this test."),
    ),
    pytest.mark.skip_value_error.name: ItemFilter(
        condition=lambda item: (
            (grid := test_utils.get_fixture_value("grid", item)).limited_area
            or (
                isinstance(grid, icon_grid.IconGrid)
                and grid.grid_params.geometry_type == icon_grid.GeometryType.ICOSAHEDRON
            )
        ),
        action=functools.partial(
            pytest.skip,
            "Stencil does not support domain containing skip values. Consider shrinking domain.",
        ),
    ),
}
