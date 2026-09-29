# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common.grid import geometry_attributes


@pytest.mark.parametrize(
    "name",
    (
        geometry_attributes.DUAL_EDGE_LENGTH,
        geometry_attributes.VERTEX_VERTEX_LENGTH,
        geometry_attributes.EDGE_LENGTH,
    ),
)
def test_data_for_inverse(name: str) -> None:
    metadata = geometry_attributes.attrs[name]
    metadata_of_inverse = geometry_attributes.metadata_for_inverse(metadata)
    assert "inverse_of" in metadata_of_inverse["standard_name"]
    assert metadata["standard_name"] in metadata_of_inverse["standard_name"]
    assert metadata["units"] in metadata_of_inverse["units"]
    assert metadata_of_inverse["units"].endswith("-1")
    assert metadata["dims"] == metadata_of_inverse["dims"]
    assert metadata["dtype"] == metadata["dtype"]
    assert "inv_" in metadata_of_inverse["icon_var_name"]
    assert metadata["icon_var_name"] == "".join(metadata_of_inverse["icon_var_name"].split("inv_"))
