# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import EllipsisType
from typing import TYPE_CHECKING

import pytest

from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import base as base_grid, horizontal as h_grid
from icon4py.model.common.metrics import metrics_attributes as attrs, metrics_factory
from icon4py.model.testing import definitions as test_defs, parallel_helpers, test_utils

from ...fixtures import (
    backend,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    experiment_description,
    geometry_from_savepoint,
    grid_savepoint,
    icon_grid,
    interpolation_factory_from_savepoint,
    metrics_factory_from_savepoint,
    metrics_savepoint,
    process_props,
    topography_savepoint,
)


if TYPE_CHECKING:
    from gt4py import next as gtx
    from gt4py.next import typing as gtx_typing

    from icon4py.model.testing import serialbox as sb


def _get_slice_tuple_from_horizontal_range(
    grid: base_grid.Grid,
    horizontal_dim: gtx.Dimension,
    horizontal_range: tuple[h_grid.Zone | None, h_grid.Zone | None],
) -> tuple[slice | None | EllipsisType, ...]:
    # TODO(havogt): Ideally we refactor the factories to only construct fields on the domain where they matter,
    # then this function disappears as we get the verification range directly from the constructed field.
    start_zone, end_zone = horizontal_range
    horizontal_start = (
        grid.start_index(h_grid.domain(horizontal_dim)(start_zone))
        if start_zone is not None
        else None
    )
    horizontal_end = (
        grid.end_index(h_grid.domain(horizontal_dim)(end_zone)) if end_zone is not None else None
    )
    return (slice(horizontal_start, horizontal_end), ...)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, metrics_name, horizontal_range",
    [
        (attrs.CELL_HEIGHT_ON_HALF_LEVEL, "z_ifc", None),
        (attrs.DDQZ_Z_FULL_E, "ddqz_z_full_e", None),
        (
            attrs.ZDIFF_GRADP,
            "zdiff_gradp",
            (h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2, None),
        ),
        (
            attrs.VERTOFFSET_GRADP,
            "vertoffset_gradp",
            (h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2, None),
        ),
        (attrs.Z_MC, "z_mc", None),
        (attrs.DDQZ_Z_HALF, "ddqz_z_half", None),
        (attrs.SCALING_FACTOR_FOR_3D_DIVDAMP, "scalfac_dd3d", None),
        (attrs.RAYLEIGH_W, "rayleigh_w", None),
        (attrs.COEFF_GRADEKIN, "coeff_gradekin", None),
    ],
)
def test_distributed_metrics_attrs(  # noqa: PLR0917 [too-many-positional-arguments]
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    metrics_factory_from_savepoint: metrics_factory.MetricsFieldsFactory,
    attrs_name: str,
    metrics_name: str,
    horizontal_range: tuple[h_grid.Zone | None, h_grid.Zone | None] | None,
    experiment: test_defs.Experiment,
) -> None:
    if test_utils.is_embedded(backend) and metrics_name == "ddqz_z_half":
        pytest.xfail("Embedded backend does not support concat_where")

    parallel_helpers.check_comm_size(process_props)
    parallel_helpers.log_process_properties(process_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = metrics_factory_from_savepoint

    field = factory.get(attrs_name).asnumpy()
    field_ref = metrics_savepoint.__getattribute__(metrics_name)().asnumpy()
    if horizontal_range is not None:
        # We assume that the horizontal dimension exists and is the first one.
        slicer = _get_slice_tuple_from_horizontal_range(
            factory.grid, attrs.attrs[attrs_name]["dims"][0], horizontal_range
        )
        field = field[slicer]
        field_ref = field_ref[slicer]

    test_utils.assert_dallclose(field, field_ref, rtol=1e-8, atol=1.0e-8)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.uses_concat_where
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, metrics_name",
    [
        (attrs.DDQZ_Z_FULL, "ddqz_z_full"),
        (attrs.INV_DDQZ_Z_FULL, "inv_ddqz_z_full"),
        (attrs.COEFF1_DWDZ, "coeff1_dwdz"),
        (attrs.COEFF2_DWDZ, "coeff2_dwdz"),
        (attrs.THETA_REF_MC, "theta_ref_mc"),
        (attrs.EXNER_REF_MC, "exner_ref_mc"),
        (attrs.RHO_REF_ME, "rho_ref_me"),
        (attrs.THETA_REF_ME, "theta_ref_me"),
        (attrs.D2DEXDZ2_FAC1_MC, "d2dexdz2_fac1_mc"),
        (attrs.D2DEXDZ2_FAC2_MC, "d2dexdz2_fac2_mc"),
        (attrs.DDXN_Z_FULL, "ddxn_z_full"),
        (attrs.DDXT_Z_FULL, "ddxt_z_full"),
        (attrs.EXNER_W_IMPLICIT_WEIGHT_PARAMETER, "vwind_impl_wgt"),
        (attrs.EXNER_W_EXPLICIT_WEIGHT_PARAMETER, "vwind_expl_wgt"),
        (attrs.PG_EXDIST_DSL, "pg_exdist_dsl"),
        (attrs.MASK_PROG_HALO_C, "mask_prog_halo_c"),
        (attrs.HORIZONTAL_MASK_FOR_3D_DIVDAMP, "hmask_dd3d"),
        (attrs.WGTFAC_C, "wgtfac_c"),
        (attrs.EXNER_EXFAC, "exner_exfac"),
    ],
)
def test_distributed_metrics_attrs_no_halo(  # noqa: PLR0917 [too-many-positional-arguments]
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    metrics_factory_from_savepoint: metrics_factory.MetricsFieldsFactory,
    attrs_name: str,
    metrics_name: str,
    experiment: test_defs.Experiment,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    parallel_helpers.log_process_properties(process_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = metrics_factory_from_savepoint

    field = factory.get(attrs_name).asnumpy()
    field_ref = metrics_savepoint.__getattribute__(metrics_name)().asnumpy()
    assert test_utils.dallclose(field, field_ref, rtol=1e-7, atol=1.0e-8)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, metrics_name",
    [
        (attrs.ZD_DIFFCOEF, "zd_diffcoef"),
        (attrs.ZD_INTCOEF, "zd_intcoef"),
        (attrs.ZD_VERTOFFSET, "zd_vertoffset"),
    ],
)
def test_distributed_metrics_attrs_no_halo_regional(  # noqa: PLR0917 [too-many-positional-arguments]
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    metrics_factory_from_savepoint: metrics_factory.MetricsFieldsFactory,
    attrs_name: str,
    metrics_name: str,
    experiment: test_defs.Experiment,
) -> None:
    if test_utils.is_embedded(backend):
        # https://github.com/GridTools/gt4py/issues/1583
        pytest.xfail("ValueError: axes don't match array")
    if experiment.description == test_defs.Experiments.EXCLAIM_APE:
        pytest.skip(f"Fields not computed for {experiment}")
    parallel_helpers.check_comm_size(process_props)
    parallel_helpers.log_process_properties(process_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = metrics_factory_from_savepoint

    field = factory.get(attrs_name).asnumpy()
    field_ref = metrics_savepoint.__getattribute__(metrics_name)().asnumpy()
    assert test_utils.dallclose(field, field_ref, atol=1e-8)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_distributed_metrics_wgtfacq_e(  # noqa: PLR0917 [too-many-positional-arguments]
    backend: gtx_typing.Backend,
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    metrics_factory_from_savepoint: metrics_factory.MetricsFieldsFactory,
    experiment: test_defs.Experiment,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    parallel_helpers.log_process_properties(process_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = metrics_factory_from_savepoint

    field = factory.get(attrs.WGTFACQ_E).asnumpy()
    field_ref = metrics_savepoint.wgtfacq_e().asnumpy()
    assert test_utils.dallclose(field, field_ref)


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_distributed_metrics_nflat_gradp(  # noqa: PLR0917 [too-many-positional-arguments]
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    metrics_factory_from_savepoint: metrics_factory.MetricsFieldsFactory,
    experiment: test_defs.Experiment,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    parallel_helpers.log_process_properties(process_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = metrics_factory_from_savepoint

    value = factory.get(attrs.NFLAT_GRADP)
    value_ref = grid_savepoint.nflat_gradp()
    assert value == value_ref
