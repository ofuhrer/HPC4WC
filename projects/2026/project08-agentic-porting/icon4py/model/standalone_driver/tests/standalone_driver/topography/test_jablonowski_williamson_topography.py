# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.common import model_backends, topography
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.topography.analytical import jablonowski_williamson as jw_topo
from icon4py.model.testing import definitions, grid_utils, test_utils
from icon4py.model.testing.fixtures import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    process_props,
    topography_savepoint,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    import icon4py.model.testing.serialbox as sb


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.JW])
def test_jablonowski_williamson_topography(
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
    topography_savepoint: sb.TopographySavepoint,
) -> None:
    gm = grid_utils.get_grid_manager_from_experiment(
        experiment=experiment,
        keep_skip_values=True,
        allocator=model_backends.get_allocator(backend),
    )
    config = topography.TopographyConfig(
        config=jw_topo.JablonowskiWilliamsonConfig(),
    )
    topo_c = topography.create(
        config=config, grid_manager=gm, backend=backend, exchange=decomp_defs.single_node_exchange
    )

    topo_c_ref = topography_savepoint.topo_c().asnumpy()

    assert test_utils.dallclose(
        topo_c,
        topo_c_ref,
    )
