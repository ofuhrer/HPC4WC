# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import gt4py.next as gtx
import pytest

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import grid_refinement as refinement, horizontal
from icon4py.model.common.metrics import metric_fields as mf
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, test_utils as testing_helpers
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

cell_domain = horizontal.domain(dims.CellDim)
edge_domain = horizontal.domain(dims.EdgeDim)
vertex_domain = horizontal.domain(dims.VertexDim)


@pytest.mark.level("unit")
@pytest.mark.embedded_remap_error
@pytest.mark.datatest
def test_compute_ddq_z_half(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    ddq_z_half_ref = metrics_savepoint.ddqz_z_half()
    z_ifc = metrics_savepoint.z_ifc()

    nlevp1 = icon_grid.num_levels + 1
    z_mc = metrics_savepoint.z_mc()
    ddqz_z_half = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )

    mf.compute_ddqz_z_half.with_backend(backend=backend)(
        z_ifc=z_ifc,
        z_mc=z_mc,
        nlev=icon_grid.num_levels,
        ddqz_z_half=ddqz_z_half,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=nlevp1,
        offset_provider={},
    )

    assert testing_helpers.dallclose(ddqz_z_half.asnumpy(), ddq_z_half_ref.asnumpy())


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_compute_ddqz_z_full_and_inverse(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    z_ifc = metrics_savepoint.z_ifc()
    inv_ddqz_full_ref = metrics_savepoint.inv_ddqz_z_full()
    ddqz_z_full = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    inv_ddqz_z_full = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)

    mf.compute_ddqz_z_full_and_inverse.with_backend(backend)(
        z_ifc=z_ifc,
        ddqz_z_full=ddqz_z_full,
        inv_ddqz_z_full=inv_ddqz_z_full,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    assert testing_helpers.dallclose(inv_ddqz_z_full.asnumpy(), inv_ddqz_full_ref.asnumpy())


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_compute_scaling_factor_for_3d_divdamp(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    scalfac_dd3d_ref = metrics_savepoint.scalfac_dd3d()
    scaling_factor_for_3d_divdamp = data_alloc.zero_field(icon_grid, dims.KDim, allocator=backend)
    divdamp_trans_start = 12500.0
    divdamp_trans_end = 17500.0
    divdamp_type = 3

    mf.compute_scaling_factor_for_3d_divdamp.with_backend(backend=backend)(
        vct_a=grid_savepoint.vct_a(),
        scaling_factor_for_3d_divdamp=scaling_factor_for_3d_divdamp,
        divdamp_trans_start=divdamp_trans_start,
        divdamp_trans_end=divdamp_trans_end,
        divdamp_type=divdamp_type,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={},
    )

    assert testing_helpers.dallclose(
        scalfac_dd3d_ref.asnumpy(), scaling_factor_for_3d_divdamp.asnumpy()
    )


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_compute_rayleigh_w(
    icon_grid: base_grid.Grid,
    experiment: definitions.Experiment,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    rayleigh_w_ref = metrics_savepoint.rayleigh_w()
    vct_a_1 = grid_savepoint.vct_a().asnumpy()[0]
    rayleigh_w_full = data_alloc.zero_field(
        icon_grid, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )
    mf.compute_rayleigh_w.with_backend(backend=backend)(
        rayleigh_w=rayleigh_w_full,
        vct_a=grid_savepoint.vct_a(),
        damping_height=experiment.config.vertical_grid.rayleigh_damping_height,
        rayleigh_type=experiment.config.metrics.rayleigh_type,
        rayleigh_coeff=experiment.config.metrics.rayleigh_coeff,
        vct_a_1=vct_a_1,
        pi_const=math.pi,
        vertical_start=0,
        vertical_end=gtx.int32(grid_savepoint.nrdmax() + 1),
        offset_provider={},
    )

    assert testing_helpers.dallclose(rayleigh_w_full.asnumpy(), rayleigh_w_ref.asnumpy())


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_compute_coeff_dwdz(
    icon_grid: base_grid.Grid, metrics_savepoint: sb.MetricSavepoint, backend: gtx_typing.Backend
) -> None:
    coeff1_dwdz_ref = metrics_savepoint.coeff1_dwdz()
    coeff2_dwdz_ref = metrics_savepoint.coeff2_dwdz()

    coeff1_dwdz_full = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    coeff2_dwdz_full = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    ddqz_z_full = gtx.as_field(
        (dims.CellDim, dims.KDim),
        1 / metrics_savepoint.inv_ddqz_z_full().ndarray,
        allocator=backend,
    )

    mf.compute_coeff_dwdz.with_backend(backend=backend)(
        ddqz_z_full=ddqz_z_full,
        z_ifc=metrics_savepoint.z_ifc(),
        coeff1_dwdz=coeff1_dwdz_full,
        coeff2_dwdz=coeff2_dwdz_full,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=1,
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={},
    )

    assert testing_helpers.dallclose(coeff1_dwdz_full.asnumpy(), coeff1_dwdz_ref.asnumpy())
    assert testing_helpers.dallclose(coeff2_dwdz_full.asnumpy(), coeff2_dwdz_ref.asnumpy())


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_compute_exner_w_explicit_weight_parameter(
    icon_grid: base_grid.Grid, metrics_savepoint: sb.MetricSavepoint, backend: gtx_typing.Backend
) -> None:
    exner_w_explicit_weight_parameter_full = data_alloc.zero_field(
        icon_grid, dims.CellDim, allocator=backend
    )
    vwind_expl_wgt_ref = metrics_savepoint.vwind_expl_wgt()
    exner_w_implicit_weight_parameter = metrics_savepoint.vwind_impl_wgt()

    mf.compute_exner_w_explicit_weight_parameter.with_backend(backend)(
        exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
        exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter_full,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        offset_provider={"C2E": icon_grid.get_connectivity("C2E")},
    )

    assert testing_helpers.dallclose(
        exner_w_explicit_weight_parameter_full.asnumpy(), vwind_expl_wgt_ref.asnumpy()
    )


@pytest.mark.level("unit")
@pytest.mark.uses_concat_where
@pytest.mark.datatest
def test_compute_exner_exfac(
    grid_savepoint: sb.IconGridSavepoint,
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    horizontal_start = icon_grid.start_index(cell_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2))

    exner_exfac = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    max_slp = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    max_hgtd = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    mf._compute_maxslp_maxhgtd.with_backend(backend)(
        metrics_savepoint.ddxn_z_full(),
        grid_savepoint.dual_edge_length(),
        out=(max_slp, max_hgtd),
        offset_provider={"C2E": icon_grid.get_connectivity("C2E")},
        domain={
            dims.CellDim: (horizontal_start, icon_grid.num_cells),
            dims.KDim: (0, icon_grid.num_levels),
        },
    )

    exner_exfac_ref = metrics_savepoint.exner_exfac()
    mf.compute_exner_exfac.with_backend(backend)(
        maxslp=max_slp,
        maxhgtd=max_hgtd,
        exner_exfac=exner_exfac,
        exner_expol=experiment.config.metrics.exner_expol,
        lateral_boundary_level_2=horizontal_start,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(icon_grid.num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={},
    )

    assert testing_helpers.dallclose(exner_exfac.asnumpy(), exner_exfac_ref.asnumpy(), rtol=1.0e-10)


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_compute_exner_w_implicit_weight_parameter(  # noqa: PLR0917 [too-many-positional-arguments]
    icon_grid: base_grid.Grid,
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    z_ifc = metrics_savepoint.z_ifc()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    tangent_orientation = grid_savepoint.tangent_orientation()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    z_ddxn_z_half_e = data_alloc.zero_field(
        icon_grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )
    z_ddxt_z_half_e = data_alloc.zero_field(
        icon_grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )
    horizontal_start = icon_grid.start_index(edge_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2))

    horizontal_end = icon_grid.end_index(edge_domain(horizontal.Zone.INTERIOR))

    vertical_start = 0
    vertical_end = icon_grid.num_levels + 1

    mf.compute_ddxn_z_half_e.with_backend(backend)(
        z_ifc=z_ifc,
        inv_dual_edge_length=inv_dual_edge_length,
        ddxn_z_half_e=z_ddxn_z_half_e,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"E2C": icon_grid.get_connectivity("E2C")},
    )

    horizontal_start_edge = icon_grid.start_index(
        edge_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    horizontal_end_edge = icon_grid.end_index(edge_domain(horizontal.Zone.INTERIOR))

    mf.compute_ddxt_z_half_e.with_backend(backend)(
        cell_in=z_ifc,
        c_int=interpolation_savepoint.c_intp(),
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        ddxt_z_half_e=z_ddxt_z_half_e,
        horizontal_start=horizontal_start_edge,
        horizontal_end=horizontal_end_edge,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={
            "E2V": icon_grid.get_connectivity("E2V"),
            "V2C": icon_grid.get_connectivity("V2C"),
        },
    )

    horizontal_start_cell = icon_grid.start_index(
        cell_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    vwind_impl_wgt_ref = metrics_savepoint.vwind_impl_wgt()
    dual_edge_length = grid_savepoint.dual_edge_length()

    exner_w_implicit_weight_parameter = mf.compute_exner_w_implicit_weight_parameter(
        c2e=icon_grid.get_connectivity(dims.C2E).ndarray,
        vct_a=grid_savepoint.vct_a().ndarray,
        z_ifc=metrics_savepoint.z_ifc().ndarray,
        z_ddxn_z_half_e=z_ddxn_z_half_e.ndarray,
        z_ddxt_z_half_e=z_ddxt_z_half_e.ndarray,
        dual_edge_length=dual_edge_length.ndarray,
        vwind_offctr=experiment.config.metrics.vwind_offctr,
        nlev=icon_grid.num_levels,
        horizontal_start_cell=horizontal_start_cell,
    )
    assert testing_helpers.dallclose(
        vwind_impl_wgt_ref.asnumpy(), data_alloc.as_numpy(exner_w_implicit_weight_parameter)
    )


# TODO(halungge): add test in test_metric_factory.py?
@pytest.mark.datatest
def test_compute_wgtfac_e(
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: base_grid.Grid,
    backend: gtx_typing.Backend,
) -> None:
    wgtfac_e = data_alloc.zero_field(
        icon_grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )
    wgtfac_e_ref = metrics_savepoint.wgtfac_e()
    mf.compute_wgtfac_e.with_backend(backend)(
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        c_lin_e=interpolation_savepoint.c_lin_e(),
        wgtfac_e=wgtfac_e,
        horizontal_start=0,
        horizontal_end=icon_grid.num_edges,
        vertical_start=0,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={"E2C": icon_grid.get_connectivity("E2C")},
    )
    assert testing_helpers.dallclose(wgtfac_e.asnumpy(), wgtfac_e_ref.asnumpy())


@pytest.mark.level("unit")
@pytest.mark.embedded_remap_error
@pytest.mark.datatest
def test_compute_pressure_gradient_downward_extrapolation_mask_distance(
    metrics_savepoint: sb.MetricSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: base_grid.Grid,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    pg_exdist_ref = metrics_savepoint.pg_exdist_dsl()

    nlev = icon_grid.num_levels
    z_mc = metrics_savepoint.z_mc()
    z_ifc = metrics_savepoint.z_ifc()
    c_lin_e = interpolation_savepoint.c_lin_e()
    topography = gtx.as_field((dims.CellDim,), z_ifc.ndarray[:, nlev], allocator=backend)

    k = data_alloc.index_field(icon_grid, dim=dims.KDim, extend={dims.KDim: 1}, allocator=backend)
    edges = data_alloc.index_field(icon_grid, dim=dims.EdgeDim, allocator=backend)

    ex_distance = data_alloc.zero_field(icon_grid, dims.EdgeDim, dims.KDim, allocator=backend)

    start_edge_nudging = icon_grid.end_index(edge_domain(horizontal.Zone.NUDGING))
    start_edge_nudging_2 = icon_grid.start_index(edge_domain(horizontal.Zone.NUDGING_LEVEL_2))

    flat_idx_max = mf.compute_flat_max_idx(
        e2c=icon_grid.get_connectivity("E2C").ndarray,
        z_mc=z_mc.ndarray,
        c_lin_e=c_lin_e.ndarray,
        z_ifc=z_ifc.ndarray,
        k_lev=k.ndarray,
        exchange=decomposition.single_node_exchange,
    )
    # TODO (nfarabullini): fix type ignore
    flat_idx = gtx.as_field((dims.EdgeDim,), data=flat_idx_max, allocator=backend)  # type: ignore [arg-type]
    mf.compute_pressure_gradient_downward_extrapolation_mask_distance.with_backend(backend)(
        z_mc=z_mc,
        topography=topography,
        c_lin_e=c_lin_e,
        e_owner_mask=grid_savepoint.e_owner_mask(),
        flat_idx_max=flat_idx,
        e_lev=edges,
        k_lev=k,
        pg_exdist_dsl=ex_distance,
        horizontal_start_distance=start_edge_nudging,
        horizontal_end_distance=icon_grid.num_edges,
        horizontal_start=start_edge_nudging_2,
        horizontal_end=icon_grid.num_edges,
        vertical_start=0,
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "E2C": icon_grid.get_connectivity("E2C"),
        },
    )

    assert testing_helpers.dallclose(pg_exdist_ref.asnumpy(), ex_distance.asnumpy(), rtol=1.0e-9)


@pytest.mark.datatest
def test_compute_mask_prog_halo_c(
    metrics_savepoint: sb.MetricSavepoint,
    icon_grid: base_grid.Grid,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    mask_prog_halo_c = data_alloc.zero_field(icon_grid, dims.CellDim, dtype=bool, allocator=backend)
    c_refin_ctrl = grid_savepoint.refin_ctrl(dims.CellDim)
    mask_prog_halo_c_ref = metrics_savepoint.mask_prog_halo_c()
    horizontal_start = icon_grid.start_index(cell_domain(horizontal.Zone.HALO))
    horizontal_end = icon_grid.end_index(cell_domain(horizontal.Zone.LOCAL))
    mf.compute_mask_prog_halo_c.with_backend(backend)(
        c_refin_ctrl=c_refin_ctrl,
        mask_prog_halo_c=mask_prog_halo_c,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        offset_provider={},
    )
    assert testing_helpers.dallclose(mask_prog_halo_c.asnumpy(), mask_prog_halo_c_ref.asnumpy())


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_compute_horizontal_mask_for_3d_divdamp(
    metrics_savepoint: sb.MetricSavepoint,
    icon_grid: base_grid.Grid,
    grid_savepoint: sb.IconGridSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    horizontal_mask_for_3d_divdamp = data_alloc.zero_field(
        icon_grid, dims.EdgeDim, allocator=backend
    )
    e_refin_ctrl = grid_savepoint.refin_ctrl(dims.EdgeDim)
    horizontal_start = icon_grid.start_index(edge_domain(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2))
    hmask_dd3d_ref = metrics_savepoint.hmask_dd3d()
    mf.compute_horizontal_mask_for_3d_divdamp.with_backend(backend)(
        e_refin_ctrl=e_refin_ctrl,
        horizontal_mask_for_3d_divdamp=horizontal_mask_for_3d_divdamp,
        grf_nudge_start_e=gtx.int32(refinement.get_nudging_refinement_value(dims.EdgeDim)),
        grf_nudgezone_width=gtx.int32(refinement.DEFAULT_GRF_NUDGEZONE_WIDTH),
        horizontal_start=horizontal_start,
        horizontal_end=icon_grid.num_edges,
        offset_provider={},
    )

    assert testing_helpers.dallclose(
        horizontal_mask_for_3d_divdamp.asnumpy(), hmask_dd3d_ref.asnumpy()
    )
