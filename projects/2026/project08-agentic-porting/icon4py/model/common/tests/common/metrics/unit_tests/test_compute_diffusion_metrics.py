# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import pytest

import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.metrics.compute_diffusion_metrics import (
    compute_diffusion_intcoef_and_vertoffset,
    compute_diffusion_mask_and_coef,
)
from icon4py.model.common.metrics.metric_fields import (
    compute_max_nbhgt,
    compute_maxslp_maxhgtd,
    compute_weighted_cell_neighbor_sum,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, test_utils
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.level("unit")
@pytest.mark.embedded_remap_error
@pytest.mark.datatest
def test_compute_diffusion_mask_and_coeff(  # noqa: PLR0917 [too-many-positional-arguments]
    metrics_savepoint: sb.MetricSavepoint,
    experiment: definitions.Experiment,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: base_grid.Grid,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    maxslp_avg = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    maxhgtd_avg = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    maxslp = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    maxhgtd = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    max_nbhgt = data_alloc.zero_field(icon_grid, dims.CellDim, allocator=backend)

    c2e2c = icon_grid.get_connectivity(dims.C2E2C).asnumpy()
    c_bln_avg = interpolation_savepoint.c_bln_avg()
    z_mc = metrics_savepoint.z_mc()
    thslp_zdiffu = experiment.config.metrics.thslp_zdiffu
    thhgtd_zdiffu = experiment.config.metrics.thhgtd_zdiffu
    cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))

    cell_lateral = icon_grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    nlev = icon_grid.num_levels

    compute_maxslp_maxhgtd.with_backend(backend)(
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        dual_edge_length=grid_savepoint.dual_edge_length(),
        maxslp=maxslp,
        maxhgtd=maxhgtd,
        horizontal_start=cell_lateral,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={"C2E": icon_grid.get_connectivity("C2E")},
    )

    compute_weighted_cell_neighbor_sum.with_backend(backend)(
        maxslp=maxslp,
        maxhgtd=maxhgtd,
        c_bln_avg=c_bln_avg,
        maxslp_avg=maxslp_avg,
        maxhgtd_avg=maxhgtd_avg,
        horizontal_start=cell_lateral,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={
            "C2E2CO": icon_grid.get_connectivity("C2E2CO"),
        },
    )

    compute_max_nbhgt.with_backend(backend)(
        z_mc_nlev=gtx.as_field((dims.CellDim,), z_mc.asnumpy()[:, nlev - 1], allocator=backend),
        max_nbhgt=max_nbhgt,
        horizontal_start=cell_nudging,
        horizontal_end=icon_grid.num_cells,
        offset_provider={"C2E2C": icon_grid.get_connectivity("C2E2C")},
    )

    zd_diffcoef = compute_diffusion_mask_and_coef(
        c2e2c=c2e2c,
        z_mc=z_mc.asnumpy(),
        max_nbhgt=max_nbhgt.asnumpy(),
        c_owner_mask=grid_savepoint.c_owner_mask().asnumpy(),
        maxslp_avg=maxslp_avg.asnumpy(),
        maxhgtd_avg=maxhgtd_avg.asnumpy(),
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        cell_nudging=cell_nudging,
        nlev=nlev,
    )
    assert test_utils.dallclose(
        zd_diffcoef, metrics_savepoint.zd_diffcoef().asnumpy(), rtol=1.0e-11
    )


@pytest.mark.level("unit")
@pytest.mark.embedded_remap_error
@pytest.mark.datatest
def test_compute_diffusion_intcoef_and_vertoffset(  # noqa: PLR0917 [too-many-positional-arguments]
    metrics_savepoint: sb.MetricSavepoint,
    experiment: definitions.Experiment,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: base_grid.Grid,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    maxslp_avg = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    maxhgtd_avg = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    maxslp = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    maxhgtd = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    max_nbhgt = data_alloc.zero_field(icon_grid, dims.CellDim, allocator=backend)

    c2e2c = icon_grid.get_connectivity(dims.C2E2C).asnumpy()
    c_bln_avg = interpolation_savepoint.c_bln_avg()
    z_mc = metrics_savepoint.z_mc()
    thslp_zdiffu = experiment.config.metrics.thslp_zdiffu
    thhgtd_zdiffu = experiment.config.metrics.thhgtd_zdiffu

    cell_nudging = icon_grid.start_index(h_grid.domain(dims.CellDim)(h_grid.Zone.NUDGING))

    cell_lateral = icon_grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    nlev = icon_grid.num_levels

    compute_maxslp_maxhgtd.with_backend(backend)(
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        dual_edge_length=grid_savepoint.dual_edge_length(),
        maxslp=maxslp,
        maxhgtd=maxhgtd,
        horizontal_start=cell_lateral,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={"C2E": icon_grid.get_connectivity("C2E")},
    )

    compute_weighted_cell_neighbor_sum.with_backend(backend)(
        maxslp=maxslp,
        maxhgtd=maxhgtd,
        c_bln_avg=c_bln_avg,
        maxslp_avg=maxslp_avg,
        maxhgtd_avg=maxhgtd_avg,
        horizontal_start=cell_lateral,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlev,
        offset_provider={
            "C2E2CO": icon_grid.get_connectivity("C2E2CO"),
        },
    )

    compute_max_nbhgt.with_backend(backend)(
        z_mc_nlev=gtx.as_field((dims.CellDim,), z_mc.asnumpy()[:, nlev - 1], allocator=backend),
        max_nbhgt=max_nbhgt,
        horizontal_start=cell_nudging,
        horizontal_end=icon_grid.num_cells,
        offset_provider={"C2E2C": icon_grid.get_connectivity("C2E2C")},
    )

    zd_intcoef, zd_vertoffset = compute_diffusion_intcoef_and_vertoffset(
        c2e2c=c2e2c,
        z_mc=z_mc.asnumpy(),
        max_nbhgt=max_nbhgt.asnumpy(),
        c_owner_mask=grid_savepoint.c_owner_mask().asnumpy(),
        maxslp_avg=maxslp_avg.asnumpy(),
        maxhgtd_avg=maxhgtd_avg.asnumpy(),
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        cell_nudging=cell_nudging,
        nlev=nlev,
    )
    assert test_utils.dallclose(zd_vertoffset, metrics_savepoint.zd_vertoffset().asnumpy())
    assert test_utils.dallclose(zd_intcoef, metrics_savepoint.zd_intcoef().asnumpy())
