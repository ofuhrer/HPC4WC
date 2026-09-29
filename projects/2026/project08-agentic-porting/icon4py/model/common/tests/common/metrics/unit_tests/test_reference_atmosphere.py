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

import icon4py.model.common.type_alias as ta
import icon4py.model.testing.test_utils as stencil_tests
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import horizontal
from icon4py.model.common.interpolation.stencils.cell_2_edge_interpolation import (
    cell_2_edge_interpolation,
)
from icon4py.model.common.metrics.reference_atmosphere import (
    compute_d2dexdz2_fac_mc,
    compute_reference_atmosphere_cell_fields,
    compute_reference_atmosphere_edge_fields,
    compute_theta_d_exner_dz_ref_ic,
)
from icon4py.model.common.utils import data_allocation as data_alloc
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


@pytest.mark.datatest
def test_compute_reference_atmosphere_fields_on_full_level_masspoints(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    exner_ref_mc_ref = metrics_savepoint.exner_ref_mc()
    rho_ref_mc_ref = metrics_savepoint.rho_ref_mc()
    theta_ref_mc_ref = metrics_savepoint.theta_ref_mc()

    z_mc = metrics_savepoint.z_mc()

    exner_ref_mc = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    rho_ref_mc = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    theta_ref_mc = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    compute_reference_atmosphere_cell_fields.with_backend(backend)(
        z_height=z_mc,
        p0ref=constants.P0REF,
        p0sl_bg=constants.SEA_LEVEL_PRESSURE,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        rd=constants.RD,
        t0sl_bg=constants.T0SL_BG,
        h_scal_bg=constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
        del_t_bg=constants.DELTA_TEMPERATURE,
        exner_ref_mc=exner_ref_mc,
        rho_ref_mc=rho_ref_mc,
        theta_ref_mc=theta_ref_mc,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(icon_grid.num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={},
    )

    assert stencil_tests.dallclose(rho_ref_mc.asnumpy(), rho_ref_mc_ref.asnumpy())
    assert stencil_tests.dallclose(theta_ref_mc.asnumpy(), theta_ref_mc_ref.asnumpy())
    assert stencil_tests.dallclose(exner_ref_mc.asnumpy(), exner_ref_mc_ref.asnumpy())


@pytest.mark.datatest
def test_compute_reference_atmosphere_on_half_level_mass_points(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    theta_ref_ic_ref = metrics_savepoint.theta_ref_ic()
    z_ifc = metrics_savepoint.z_ifc()

    exner_ref_ic = data_alloc.zero_field(
        icon_grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        dtype=ta.wpfloat,
        allocator=backend,
    )
    rho_ref_ic = data_alloc.zero_field(
        icon_grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        dtype=ta.wpfloat,
        allocator=backend,
    )
    theta_ref_ic = data_alloc.zero_field(
        icon_grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        dtype=ta.wpfloat,
        allocator=backend,
    )
    compute_reference_atmosphere_cell_fields.with_backend(backend=backend)(
        z_height=z_ifc,
        p0ref=constants.P0REF,
        p0sl_bg=constants.SEA_LEVEL_PRESSURE,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        rd=constants.RD,
        t0sl_bg=constants.T0SL_BG,
        h_scal_bg=constants._H_SCAL_BG,
        del_t_bg=constants.DELTA_TEMPERATURE,
        exner_ref_mc=exner_ref_ic,
        rho_ref_mc=rho_ref_ic,
        theta_ref_mc=theta_ref_ic,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(icon_grid.num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels + 1),
        offset_provider={},
    )

    assert stencil_tests.dallclose(theta_ref_ic.asnumpy(), theta_ref_ic_ref.asnumpy())


@pytest.mark.datatest
def test_compute_d_exner_dz_ref_ic(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    z_ifc = metrics_savepoint.z_ifc()
    theta_ref_ic = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )
    d_exner_dz_ref_ic = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, allocator=backend
    )
    compute_theta_d_exner_dz_ref_ic.with_backend(backend)(
        z_ifc=z_ifc,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        theta_ref_ic=theta_ref_ic,
        t0sl_bg=constants.SEA_LEVEL_TEMPERATURE,
        del_t_bg=constants.DELTA_TEMPERATURE,
        h_scal_bg=constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
        grav=constants.GRAV,
        rd=constants.RD,
        cpd=constants.CPD,
        p0sl_bg=constants.SEA_LEVEL_PRESSURE,
        rd_o_cpd=constants.RD_O_CPD,
        p0ref=constants.REFERENCE_PRESSURE,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(icon_grid.num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels + 1),
        offset_provider={},
    )

    d_exner_dz_ref_ic_ref = metrics_savepoint.d_exner_dz_ref_ic()
    assert stencil_tests.dallclose(d_exner_dz_ref_ic.asnumpy(), d_exner_dz_ref_ic_ref.asnumpy())


@pytest.mark.datatest
def test_compute_reference_atmosphere_on_full_level_edge_fields(
    icon_grid: base_grid.Grid,
    interpolation_savepoint: sb.InterpolationSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    rho_ref_me_ref = metrics_savepoint.rho_ref_me()
    theta_ref_me_ref = metrics_savepoint.theta_ref_me()

    c_lin_e = interpolation_savepoint.c_lin_e()
    z_mc = metrics_savepoint.z_mc()

    horizontal_start = icon_grid.start_index(
        horizontal.domain(dims.EdgeDim)(horizontal.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    rho_ref_me = data_alloc.zero_field(
        icon_grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )
    theta_ref_me = data_alloc.zero_field(
        icon_grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat, allocator=backend
    )

    compute_reference_atmosphere_edge_fields.with_backend(backend)(
        z_mc=z_mc,
        c_lin_e=c_lin_e,
        rho_ref_me=rho_ref_me,
        theta_ref_me=theta_ref_me,
        p0ref=constants.P0REF,
        p0sl_bg=constants.SEA_LEVEL_PRESSURE,
        grav=constants.GRAVITATIONAL_ACCELERATION,
        cpd=constants.CPD,
        rd=constants.RD,
        t0sl_bg=constants.T0SL_BG,
        h_scal_bg=constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
        del_t_bg=constants.DELTA_TEMPERATURE,
        horizontal_start=horizontal_start,
        horizontal_end=(gtx.int32(icon_grid.num_edges)),
        vertical_start=(gtx.int32(0)),
        vertical_end=(gtx.int32(icon_grid.num_levels)),
        offset_provider={"E2C": icon_grid.get_connectivity("E2C")},
    )
    assert stencil_tests.dallclose(rho_ref_me.asnumpy(), rho_ref_me_ref.asnumpy(), rtol=1e-10)
    assert stencil_tests.dallclose(theta_ref_me.asnumpy(), theta_ref_me_ref.asnumpy())


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_compute_d2dexdz2_fac_mc(
    icon_grid: base_grid.Grid,
    metrics_savepoint: sb.MetricSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    z_mc = metrics_savepoint.z_mc()
    d2dexdz2_fac1_mc_ref = metrics_savepoint.d2dexdz2_fac1_mc()
    d2dexdz2_fac2_mc_ref = metrics_savepoint.d2dexdz2_fac2_mc()

    d2dexdz2_fac1_mc = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)
    d2dexdz2_fac2_mc = data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, allocator=backend)

    compute_d2dexdz2_fac_mc.with_backend(backend=backend)(
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
        exner_ref_mc=metrics_savepoint.exner_ref_mc(),
        z_mc=z_mc,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        cpd=constants.CPD,
        grav=constants.GRAV,
        del_t_bg=constants.DEL_T_BG,
        h_scal_bg=constants.HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(icon_grid.num_cells),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={},
    )

    assert stencil_tests.dallclose(d2dexdz2_fac1_mc.asnumpy(), d2dexdz2_fac1_mc_ref.asnumpy())
    assert stencil_tests.dallclose(d2dexdz2_fac2_mc.asnumpy(), d2dexdz2_fac2_mc_ref.asnumpy())
