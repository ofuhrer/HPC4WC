# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging
import pathlib

import numpy as np
import pytest
from gt4py import next as gtx
from gt4py.next import common as gtx_common, typing as gtx_typing

from icon4py.model.common import dimension as dims, exceptions, model_backends
from icon4py.model.common.decomposition import (
    decomposer as decomp,
    definitions as decomp_defs,
    mpi_decomposition,
)
from icon4py.model.common.grid import (
    base,
    geometry,
    geometry_attributes,
    geometry_config,
    grid_manager as gm,
    gridfile,
    icon,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    parallel_helpers,
    test_utils,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    download_ser_data,
    experiment,
    experiment_description,
    grid_description,
    process_props,
    topography_savepoint,
)

from . import utils


_log = logging.getLogger(__file__)


@pytest.mark.datatest
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.mpi(min_size=2)
def test_grid_manager_validate_decomposer(
    process_props: decomp_defs.ProcessProperties,
    experiment: test_defs.Experiment,
) -> None:
    if experiment.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    file = dt_utils.get_grid_filepath(experiment.grid)
    manager = gm.GridManager(
        grid_file=file,
        config=v_grid.VerticalGridConfig(num_levels=utils.NUM_LEVELS),
        offset_transformation=gridfile.ToZeroBasedIndexTransformation(),
    )
    with pytest.raises(exceptions.InvalidConfigError) as e:
        manager(
            keep_skip_values=True,
            allocator=None,
            process_props=process_props,
            decomposer=decomp.SingleNodeDecomposer(),
        )

    assert "Need a Decomposer for multi" in e.value.args[0]


def _get_neighbor_tables(grid: base.Grid) -> dict:
    return {k: v.ndarray for k, v in grid.connectivities.items() if gtx_common.is_neighbor_table(v)}


# These fields can't be computed with the embedded backend for one reason or
# another, so we declare them here for xfailing.
embedded_broken_fields = {
    metrics_attributes.DDQZ_Z_HALF,
    metrics_attributes.DEEPATMO_DIVH,
    metrics_attributes.DEEPATMO_DIVZL,
    metrics_attributes.DEEPATMO_DIVZU,
    metrics_attributes.EXNER_EXFAC,
    metrics_attributes.MAXHGTD_AVG,
    metrics_attributes.MAXSLP_AVG,
    metrics_attributes.PG_EXDIST_DSL,
    metrics_attributes.WGTFAC_C,
    metrics_attributes.WGTFAC_E,
    metrics_attributes.ZD_DIFFCOEF,
    metrics_attributes.ZD_INTCOEF,
    metrics_attributes.ZD_VERTOFFSET,
}


def _make_single_rank_geometry(
    grid_file: pathlib.Path,
    backend: gtx_typing.Backend | None,
    allocator: gtx.typing.Allocator,
    num_levels: int = utils.NUM_LEVELS,
) -> tuple[gm.GridManager, geometry.GridGeometry]:
    grid_manager = utils.run_grid_manager_for_single_rank(
        grid_file, allocator=allocator, num_levels=num_levels
    )
    grid_geometry = geometry.GridGeometry(
        backend=backend,
        grid=grid_manager.grid,
        coordinates=grid_manager.coordinates,
        decomposition_info=grid_manager.decomposition_info,
        extra_fields=grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
        config=geometry_config.GeometryConfig(),
        process_props=decomp_defs.SingleNodeProcessProperties(),
        exchange=decomp_defs.single_node_exchange,
    )
    return grid_manager, grid_geometry


def _make_multi_rank_geometry(
    grid_file: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    allocator: gtx.typing.Allocator,
    num_levels: int = utils.NUM_LEVELS,
) -> tuple[gm.GridManager, geometry.GridGeometry]:
    grid_manager = utils.run_grid_manager_for_multi_rank(
        file=grid_file,
        process_props=process_props,
        decomposer=decomp.MetisDecomposer(),
        allocator=allocator,
        num_levels=num_levels,
    )
    grid_geometry = geometry.GridGeometry(
        backend=backend,
        grid=grid_manager.grid,
        coordinates=grid_manager.coordinates,
        decomposition_info=grid_manager.decomposition_info,
        extra_fields=grid_manager.geometry_fields,
        metadata=geometry_attributes.attrs,
        config=geometry_config.GeometryConfig(),
        process_props=process_props,
        exchange=decomp_defs.create_exchange(process_props, grid_manager.decomposition_info),
        global_reductions=decomp_defs.create_reduction(
            process_props, grid_manager.decomposition_info
        ),
    )
    return grid_manager, grid_geometry


def _compare_geometry_fields_single_multi_rank(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    grid_description: test_defs.GridDescription,
    attrs_name: str,
) -> None:
    if grid_description.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    if attrs_name in embedded_broken_fields and test_utils.is_embedded(backend):
        pytest.xfail(f"Field {attrs_name} can't be computed with the embedded backend")

    allocator = model_backends.get_allocator(backend)

    grid_file = grid_utils._download_grid_file(grid_description)
    _log.info(f"running on {process_props.comm} with {process_props.comm_size} ranks")
    single_rank_gm, single_rank_geometry = _make_single_rank_geometry(grid_file, backend, allocator)
    _log.info(
        f"rank = {process_props.rank} : single node grid has size "
        f"{single_rank_gm.decomposition_info.get_horizontal_size()!r}"
    )

    multi_rank_gm, multi_rank_geometry = _make_multi_rank_geometry(
        grid_file, process_props, backend, allocator
    )
    _log.info(
        f"rank = {process_props.rank} : {multi_rank_gm.decomposition_info.get_horizontal_size()!r}"
    )
    _log.info(
        f"rank = {process_props.rank}: halo size for 'CellDim' "
        f"(1: {multi_rank_gm.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {multi_rank_gm.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )

    field_ref = single_rank_geometry.get(attrs_name)
    field = multi_rank_geometry.get(attrs_name)
    dim = field_ref.domain.dims[0]

    parallel_helpers.check_local_global_field(
        decomposition_info=multi_rank_gm.decomposition_info,
        process_props=process_props,
        dim=dim,
        global_reference_field=field_ref.asnumpy(),
        local_field=field.asnumpy(),
        check_halos=True,
        atol=1e-15,
    )

    _log.info(f"rank = {process_props.rank} - DONE")


@pytest.mark.datatest
@pytest.mark.level("unit")
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name",
    [
        geometry_attributes.CELL_CENTER_Y,
        geometry_attributes.CELL_CENTER_Z,
        geometry_attributes.CELL_LON,
        geometry_attributes.DUAL_EDGE_LENGTH,
        geometry_attributes.EDGE_CENTER_Y,
        geometry_attributes.EDGE_CENTER_Z,
        geometry_attributes.EDGE_DUAL_V,
        geometry_attributes.EDGE_LENGTH,
        geometry_attributes.EDGE_LON,
        geometry_attributes.EDGE_NORMAL_CELL_V,
        geometry_attributes.EDGE_NORMAL_V,
        geometry_attributes.EDGE_NORMAL_VERTEX_V,
        geometry_attributes.EDGE_NORMAL_Y,
        geometry_attributes.EDGE_NORMAL_Z,
        geometry_attributes.EDGE_TANGENT_CELL_V,
        geometry_attributes.EDGE_TANGENT_VERTEX_V,
        geometry_attributes.EDGE_TANGENT_Y,
        geometry_attributes.EDGE_TANGENT_Z,
        geometry_attributes.VERTEX_LON,
        geometry_attributes.VERTEX_VERTEX_LENGTH,
        geometry_attributes.VERTEX_Y,
        geometry_attributes.VERTEX_Z,
    ],
)
def test_geometry_fields_compare_single_multi_rank_unit(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    grid_description: test_defs.GridDescription,
    attrs_name: str,
) -> None:
    _compare_geometry_fields_single_multi_rank(process_props, backend, grid_description, attrs_name)


@pytest.mark.datatest
@pytest.mark.level("integration")
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name",
    [
        geometry_attributes.CELL_AREA,
        geometry_attributes.CELL_CENTER_X,
        geometry_attributes.CELL_LAT,
        geometry_attributes.CELL_NORMAL_ORIENTATION,
        geometry_attributes.CORIOLIS_PARAMETER,
        geometry_attributes.DUAL_AREA,
        f"inverse_of_{geometry_attributes.DUAL_EDGE_LENGTH}",
        geometry_attributes.EDGE_AREA,
        geometry_attributes.EDGE_CELL_DISTANCE,
        geometry_attributes.EDGE_CENTER_X,
        geometry_attributes.EDGE_DUAL_U,
        geometry_attributes.EDGE_LAT,
        f"inverse_of_{geometry_attributes.EDGE_LENGTH}",
        geometry_attributes.EDGE_NORMAL_CELL_U,
        geometry_attributes.EDGE_NORMAL_U,
        geometry_attributes.EDGE_NORMAL_VERTEX_U,
        geometry_attributes.EDGE_NORMAL_X,
        geometry_attributes.EDGE_TANGENT_CELL_U,
        geometry_attributes.EDGE_TANGENT_VERTEX_U,
        geometry_attributes.EDGE_TANGENT_X,
        geometry_attributes.EDGE_VERTEX_DISTANCE,
        geometry_attributes.TANGENT_ORIENTATION,
        geometry_attributes.VERTEX_EDGE_ORIENTATION,
        geometry_attributes.VERTEX_LAT,
        f"inverse_of_{geometry_attributes.VERTEX_VERTEX_LENGTH}",
        geometry_attributes.VERTEX_X,
    ],
)
def test_geometry_fields_compare_single_multi_rank_integration(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    grid_description: test_defs.GridDescription,
    attrs_name: str,
) -> None:
    _compare_geometry_fields_single_multi_rank(process_props, backend, grid_description, attrs_name)


def _compare_interpolation_fields_single_multi_rank(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    experiment: test_defs.Experiment,
    attrs_name: str,
) -> None:
    if experiment.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    if attrs_name in embedded_broken_fields and test_utils.is_embedded(backend):
        pytest.xfail(f"Field {attrs_name} can't be computed with the embedded backend")

    allocator = model_backends.get_allocator(backend)

    grid_file = dt_utils.get_grid_filepath(experiment.grid)
    _log.info(f"running on {process_props.comm} with {process_props.comm_size} ranks")
    single_rank_gm, single_rank_geometry = _make_single_rank_geometry(grid_file, backend, allocator)
    single_rank_interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=single_rank_gm.grid,
        decomposition_info=single_rank_gm.decomposition_info,
        geometry_source=single_rank_geometry,
        config=experiment.config.interpolation,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=decomp_defs.SingleNodeExchange(),
    )
    _log.info(
        f"rank = {process_props.rank} : single node grid has size "
        f"{single_rank_gm.decomposition_info.get_horizontal_size()!r}"
    )

    multi_rank_gm, multi_rank_geometry = _make_multi_rank_geometry(
        grid_file, process_props, backend, allocator
    )
    _log.info(
        f"rank = {process_props.rank} : {multi_rank_gm.decomposition_info.get_horizontal_size()!r}"
    )
    _log.info(
        f"rank = {process_props.rank}: halo size for 'CellDim' "
        f"(1: {multi_rank_gm.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {multi_rank_gm.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    multi_rank_interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=multi_rank_gm.grid,
        decomposition_info=multi_rank_gm.decomposition_info,
        geometry_source=multi_rank_geometry,
        config=experiment.config.interpolation,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=decomp_defs.create_exchange(process_props, multi_rank_gm.decomposition_info),
    )

    field_ref = single_rank_interpolation.get(attrs_name)
    field = multi_rank_interpolation.get(attrs_name)
    dim = field_ref.domain.dims[0]

    parallel_helpers.check_local_global_field(
        decomposition_info=multi_rank_gm.decomposition_info,
        process_props=process_props,
        dim=dim,
        global_reference_field=field_ref.asnumpy(),
        local_field=field.asnumpy(),
        check_halos=True,
        atol=3e-9
        if attrs_name.startswith("rbf")
        else 1e-10
        if attrs_name.startswith("pos_on_tplane")
        else 1e-15,
    )

    _log.info(f"rank = {process_props.rank} - DONE")


@pytest.mark.datatest
@pytest.mark.level("unit")
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name",
    [
        interpolation_attributes.CELL_AW_VERTS,
        interpolation_attributes.C_BLN_AVG,
        interpolation_attributes.C_LIN_E,
        interpolation_attributes.E_BLN_C_S,
        interpolation_attributes.GEOFAC_DIV,
        interpolation_attributes.GEOFAC_GRG_Y,
        interpolation_attributes.GEOFAC_ROT,
        interpolation_attributes.LSQ_PSEUDOINV,
        interpolation_attributes.NUDGECOEFFS_E,
        interpolation_attributes.POS_ON_TPLANE_E_X,
        interpolation_attributes.POS_ON_TPLANE_E_Y,
        interpolation_attributes.RBF_VEC_COEFF_C2,
        interpolation_attributes.RBF_VEC_COEFF_V2,
    ],
)
def test_interpolation_fields_compare_single_multi_rank_unit(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    experiment: test_defs.Experiment,
    attrs_name: str,
) -> None:
    _compare_interpolation_fields_single_multi_rank(process_props, backend, experiment, attrs_name)


@pytest.mark.datatest
@pytest.mark.level("integration")
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name",
    [
        interpolation_attributes.E_FLX_AVG,
        interpolation_attributes.GEOFAC_GRDIV,
        interpolation_attributes.GEOFAC_GRG_X,
        interpolation_attributes.GEOFAC_N2S,
        interpolation_attributes.RBF_VEC_COEFF_C1,
        interpolation_attributes.RBF_VEC_COEFF_E,
        interpolation_attributes.RBF_VEC_COEFF_V1,
    ],
)
def test_interpolation_fields_compare_single_multi_rank_integration(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    experiment: test_defs.Experiment,
    attrs_name: str,
) -> None:
    _compare_interpolation_fields_single_multi_rank(process_props, backend, experiment, attrs_name)


def _compare_metrics_fields_single_multi_rank(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    experiment: test_defs.Experiment,
    attrs_name: str,
) -> None:
    if experiment.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    if attrs_name in embedded_broken_fields and test_utils.is_embedded(backend):
        pytest.xfail(f"Field {attrs_name} can't be computed with the embedded backend")

    file = dt_utils.get_grid_filepath(experiment.grid)

    vertical_config = experiment.config.vertical_grid
    xp = data_alloc.import_array_ns(backend)
    allocator = model_backends.get_allocator(backend)
    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field(
            (dims.KDim,),
            xp.linspace(12000.0, 0.0, experiment.config.vertical_grid.num_levels + 1),
            allocator=allocator,
        ),
        vct_b=gtx.as_field(
            (dims.KDim,),
            xp.linspace(12000.0, 0.0, experiment.config.vertical_grid.num_levels + 1),
            allocator=allocator,
        ),
    )

    _log.info(f"running on {process_props.comm} with {process_props.comm_size} ranks")
    single_rank_gm, single_rank_geometry = _make_single_rank_geometry(
        file, backend, allocator, num_levels=experiment.config.vertical_grid.num_levels
    )
    single_rank_interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=single_rank_gm.grid,
        decomposition_info=single_rank_gm.decomposition_info,
        geometry_source=single_rank_geometry,
        config=experiment.config.interpolation,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=decomp_defs.SingleNodeExchange(),
    )
    single_rank_metrics = metrics_factory.MetricsFieldsFactory(
        grid=single_rank_geometry.grid,
        vertical_grid=vertical_grid,
        decomposition_info=single_rank_gm.decomposition_info,
        geometry_source=single_rank_geometry,
        topography=(
            gtx.as_field(
                (dims.CellDim,),
                xp.zeros(single_rank_geometry.grid.num_cells),
                allocator=allocator,
            )
        ),
        interpolation_source=single_rank_interpolation,
        config=experiment.config.metrics,
        backend=backend,
        metadata=metrics_attributes.attrs,
        exchange=decomp_defs.SingleNodeExchange(),
    )
    _log.info(
        f"rank = {process_props.rank} : single node grid has size "
        f"{single_rank_gm.decomposition_info.get_horizontal_size()!r}"
    )

    multi_rank_gm, multi_rank_geometry = _make_multi_rank_geometry(
        file,
        process_props,
        backend,
        allocator,
        num_levels=experiment.config.vertical_grid.num_levels,
    )
    _log.info(
        f"rank = {process_props.rank} : {multi_rank_gm.decomposition_info.get_horizontal_size()!r}"
    )
    _log.info(
        f"rank = {process_props.rank}: halo size for 'CellDim' "
        f"(1: {multi_rank_gm.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {multi_rank_gm.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    multi_rank_interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=multi_rank_gm.grid,
        decomposition_info=multi_rank_gm.decomposition_info,
        geometry_source=multi_rank_geometry,
        config=experiment.config.interpolation,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=decomp_defs.create_exchange(process_props, multi_rank_gm.decomposition_info),
    )
    multi_rank_metrics = metrics_factory.MetricsFieldsFactory(
        grid=multi_rank_geometry.grid,
        vertical_grid=vertical_grid,
        decomposition_info=multi_rank_gm.decomposition_info,
        geometry_source=multi_rank_geometry,
        topography=(
            gtx.as_field(
                (dims.CellDim,),
                xp.zeros(multi_rank_geometry.grid.num_cells),
                allocator=allocator,
            )
        ),
        interpolation_source=multi_rank_interpolation,
        config=experiment.config.metrics,
        backend=backend,
        metadata=metrics_attributes.attrs,
        exchange=mpi_decomposition.GHexMultiNodeExchange(
            process_props, multi_rank_gm.decomposition_info
        ),
    )

    field_ref = single_rank_metrics.get(attrs_name)
    field = multi_rank_metrics.get(attrs_name)

    if isinstance(field_ref, state_utils.ScalarType):
        assert isinstance(field, state_utils.ScalarType)
        assert pytest.approx(field) == field_ref
    else:
        if test_utils.is_dace(backend) and (
            model_backends.is_cpu_backend(backend)
            or (
                model_backends.is_gpu_backend(backend)
                and attrs_name
                in {
                    metrics_attributes.DDQZ_Z_FULL_E,
                    metrics_attributes.RHO_REF_ME,
                    metrics_attributes.THETA_REF_ME,
                }
            )
        ):
            # TODO (jcanton,phimuell): figure out dace undeterministic behaviour
            atol = 2e-13
        else:
            atol = 0.0
        parallel_helpers.check_local_global_field(
            decomposition_info=multi_rank_gm.decomposition_info,
            process_props=process_props,
            dim=field_ref.domain.dims[0],
            global_reference_field=field_ref.asnumpy(),
            local_field=field.asnumpy(),
            check_halos=True,
            atol=atol,
        )

    _log.info(f"rank = {process_props.rank} - DONE")


@pytest.mark.datatest
@pytest.mark.level("unit")
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name",
    [
        metrics_attributes.CELL_HEIGHT_ON_HALF_LEVEL,
        metrics_attributes.COEFF2_DWDZ,
        metrics_attributes.COEFF_GRADEKIN,
        metrics_attributes.D2DEXDZ2_FAC2_MC,
        metrics_attributes.DDQZ_Z_FULL,
        metrics_attributes.DDXN_Z_HALF_E,
        metrics_attributes.DDXT_Z_FULL,
        metrics_attributes.DDXT_Z_HALF_E,
        metrics_attributes.DEEPATMO_DIVH,
        metrics_attributes.DEEPATMO_DIVZL,
        metrics_attributes.DEEPATMO_DIVZU,
        metrics_attributes.D_EXNER_DZ_REF_IC,
        metrics_attributes.EXNER_REF_MC,
        metrics_attributes.EXNER_W_IMPLICIT_WEIGHT_PARAMETER,
        metrics_attributes.FLAT_IDX_MAX,
        metrics_attributes.HORIZONTAL_MASK_FOR_3D_DIVDAMP,
        metrics_attributes.INV_DDQZ_Z_FULL,
        metrics_attributes.MAXHGTD,
        metrics_attributes.MAXSLP,
        metrics_attributes.MAXSLP_AVG,
        metrics_attributes.MAX_NBHGT,
        metrics_attributes.PG_EXDIST_DSL,
        metrics_attributes.RAYLEIGH_W,
        metrics_attributes.RHO_REF_MC,
        metrics_attributes.RHO_REF_ME,
        metrics_attributes.SCALING_FACTOR_FOR_3D_DIVDAMP,
        metrics_attributes.THETA_REF_IC,
        metrics_attributes.THETA_REF_MC,
        metrics_attributes.THETA_REF_ME,
        metrics_attributes.VERTOFFSET_GRADP,
        metrics_attributes.WGTFACQ_C,
        metrics_attributes.WGTFAC_C,
        metrics_attributes.ZDIFF_GRADP,
        metrics_attributes.ZD_DIFFCOEF,
        metrics_attributes.ZD_VERTOFFSET,
        metrics_attributes.Z_MC,
    ],
)
def test_metrics_fields_compare_single_multi_rank_unit(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    experiment: test_defs.Experiment,
    attrs_name: str,
) -> None:
    _compare_metrics_fields_single_multi_rank(process_props, backend, experiment, attrs_name)


@pytest.mark.datatest
@pytest.mark.level("integration")
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name",
    [
        metrics_attributes.COEFF1_DWDZ,
        metrics_attributes.D2DEXDZ2_FAC1_MC,
        metrics_attributes.DDQZ_Z_FULL_E,
        metrics_attributes.DDQZ_Z_HALF,
        metrics_attributes.DDXN_Z_FULL,
        metrics_attributes.EXNER_EXFAC,
        metrics_attributes.EXNER_W_EXPLICIT_WEIGHT_PARAMETER,
        metrics_attributes.MAXHGTD_AVG,
        metrics_attributes.NFLAT_GRADP,
        metrics_attributes.WGTFACQ_E,
        metrics_attributes.WGTFAC_E,
        metrics_attributes.ZD_INTCOEF,
    ],
)
def test_metrics_fields_compare_single_multi_rank_integration(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    experiment: test_defs.Experiment,
    attrs_name: str,
) -> None:
    _compare_metrics_fields_single_multi_rank(process_props, backend, experiment, attrs_name)


# MASK_PROG_HALO_C is defined specially only on halos, so we have a separate
# test for it. It doesn't make sense to compare to a single-rank reference since
# it has no halos.
@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_metrics_mask_prog_halo_c(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    experiment: test_defs.Experiment,
) -> None:
    if experiment.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    file = dt_utils.get_grid_filepath(experiment.grid)

    vertical_config = experiment.config.vertical_grid
    xp = data_alloc.import_array_ns(backend)
    allocator = model_backends.get_allocator(backend)
    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field(
            (dims.KDim,),
            xp.linspace(12000.0, 0.0, experiment.config.vertical_grid.num_levels + 1),
            allocator=allocator,
        ),
        vct_b=gtx.as_field(
            (dims.KDim,),
            xp.linspace(12000.0, 0.0, experiment.config.vertical_grid.num_levels + 1),
            allocator=allocator,
        ),
    )

    _log.info(f"running on {process_props.comm} with {process_props.comm_size} ranks")

    multi_rank_gm, multi_rank_geometry = _make_multi_rank_geometry(
        file,
        process_props,
        backend,
        allocator,
        num_levels=experiment.config.vertical_grid.num_levels,
    )
    _log.info(
        f"rank = {process_props.rank} : {multi_rank_gm.decomposition_info.get_horizontal_size()!r}"
    )
    _log.info(
        f"rank = {process_props.rank}: halo size for 'CellDim' "
        f"(1: {multi_rank_gm.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {multi_rank_gm.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )
    multi_rank_interpolation = interpolation_factory.InterpolationFieldsFactory(
        grid=multi_rank_gm.grid,
        decomposition_info=multi_rank_gm.decomposition_info,
        geometry_source=multi_rank_geometry,
        config=experiment.config.interpolation,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=decomp_defs.create_exchange(process_props, multi_rank_gm.decomposition_info),
    )
    multi_rank_metrics = metrics_factory.MetricsFieldsFactory(
        grid=multi_rank_geometry.grid,
        vertical_grid=vertical_grid,
        decomposition_info=multi_rank_gm.decomposition_info,
        geometry_source=multi_rank_geometry,
        topography=(
            gtx.as_field(
                (dims.CellDim,),
                xp.zeros(multi_rank_geometry.grid.num_cells),
                allocator=allocator,
            )
        ),
        interpolation_source=multi_rank_interpolation,
        config=experiment.config.metrics,
        backend=backend,
        metadata=metrics_attributes.attrs,
        exchange=mpi_decomposition.GHexMultiNodeExchange(
            process_props, multi_rank_gm.decomposition_info
        ),
    )

    attrs_name = metrics_attributes.MASK_PROG_HALO_C
    field = multi_rank_metrics.get(attrs_name).ndarray
    c_refin_ctrl = multi_rank_metrics.get("c_refin_ctrl").ndarray
    assert not (
        field[
            multi_rank_gm.decomposition_info.local_index(
                dims.CellDim, decomp_defs.DecompositionInfo.EntryType.OWNED
            )
        ]
    ).any(), f"rank={process_props.rank} - found nonzero in owned entries of {attrs_name}"
    halo_indices = multi_rank_gm.decomposition_info.local_index(
        dims.CellDim, decomp_defs.DecompositionInfo.EntryType.HALO
    )
    assert (
        field[halo_indices]
        == xp.invert((c_refin_ctrl[halo_indices] >= 1) & (c_refin_ctrl[halo_indices] <= 4))
    ).all(), f"rank={process_props.rank} - halo for MASK_PROG_HALO_C is incorrect"

    _log.info(f"rank = {process_props.rank} - DONE")


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_validate_skip_values_in_distributed_connectivities(
    process_props: decomp_defs.ProcessProperties,
    experiment: test_defs.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    if experiment.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    file = dt_utils.get_grid_filepath(experiment.grid)
    multi_rank_grid_manager = utils.run_grid_manager_for_multi_rank(
        file=file,
        process_props=process_props,
        decomposer=decomp.MetisDecomposer(),
        allocator=model_backends.get_allocator(backend),
    )
    distributed_grid = multi_rank_grid_manager.grid
    for k, c in distributed_grid.connectivities.items():
        if gtx_common.is_neighbor_table(c):
            skip_values_in_table = np.count_nonzero(c.asnumpy() == c.skip_value)
            found_skips = skip_values_in_table > 0
            assert found_skips == (c.skip_value is not None), (
                f"rank={process_props.rank} / {process_props.comm_size}: {k} - # of skip values found in table = {skip_values_in_table},  skip value is {c.skip_value}"
            )
            if skip_values_in_table > 0:
                dim = gtx.Dimension(k, gtx.DimensionKind.LOCAL)
                assert (
                    dim in icon.CONNECTIVITIES_ON_BOUNDARIES
                    or dim in icon.CONNECTIVITIES_ON_PENTAGONS
                ), (
                    f"rank={process_props.rank} / {process_props.comm_size}: {k} has skip found in table, expected none"
                )


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize("grid", [test_defs.Grids.MCH_CH_R04B09_DSL])
def test_limited_area_raises(
    process_props: decomp_defs.ProcessProperties,
    grid: test_defs.GridDescription,
    backend: gtx_typing.Backend | None,
) -> None:
    with pytest.raises(
        NotImplementedError, match="Limited-area grids are not supported in distributed runs"
    ):
        _ = utils.run_grid_manager_for_multi_rank(
            file=dt_utils.get_grid_filepath(grid),
            process_props=process_props,
            decomposer=decomp.MetisDecomposer(),
            allocator=model_backends.get_allocator(backend),
        )


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    ("field_name", "reduction"),
    # NOTE: these fields are selected as examples for cell, edge, vertex fields
    [
        (geometry_attributes.CELL_AREA, "min"),
        (geometry_attributes.CELL_AREA, "max"),
        (geometry_attributes.CELL_AREA, "sum"),
        (geometry_attributes.CELL_AREA, "mean"),
        (geometry_attributes.EDGE_LENGTH, "min"),
        (geometry_attributes.EDGE_LENGTH, "max"),
        (geometry_attributes.EDGE_LENGTH, "sum"),
        (geometry_attributes.EDGE_LENGTH, "mean"),
        (geometry_attributes.DUAL_AREA, "min"),
        (geometry_attributes.DUAL_AREA, "max"),
        (geometry_attributes.DUAL_AREA, "sum"),
        (geometry_attributes.DUAL_AREA, "mean"),
    ],
)
def test_global_reductions_single_vs_multi_rank(
    process_props: decomp_defs.ProcessProperties,
    experiment: test_defs.Experiment,
    backend: gtx_typing.Backend | None,
    field_name: str,
    reduction: str,
) -> None:
    """Compare global reductions from multi-rank (with halos) against single-rank (no halos).

    Uses real geometry fields from the grid file (cell_area on CellDim,
    edge_length on EdgeDim, dual_area on VertexDim) so that all three
    horizontal dimensions are exercised.
    """
    if experiment.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    allocator = model_backends.get_allocator(backend)
    grid_file = grid_utils._download_grid_file(experiment.grid)

    single_rank_gm, single_rank_geometry = _make_single_rank_geometry(grid_file, backend, allocator)
    single_rank_reductions = decomp_defs.create_reduction(
        decomp_defs.SingleNodeProcessProperties(), single_rank_gm.decomposition_info
    )
    single_rank_field = single_rank_geometry.get(field_name).ndarray

    multi_rank_gm, multi_rank_geometry = _make_multi_rank_geometry(
        grid_file, process_props, backend, allocator
    )
    multi_rank_reductions = decomp_defs.create_reduction(
        process_props, multi_rank_gm.decomposition_info
    )
    multi_rank_field = multi_rank_geometry.get(field_name).ndarray

    reduce_fn_single = getattr(single_rank_reductions, reduction)
    reduce_fn_multi = getattr(multi_rank_reductions, reduction)

    expected = reduce_fn_single(single_rank_field)
    result = reduce_fn_multi(multi_rank_field)

    # Also verify against plain NumPy as a sanity check.
    np_reference = getattr(np, reduction)(data_alloc.as_numpy(single_rank_field))

    assert result == pytest.approx(expected, rel=1e-15), (
        f"rank={process_props.rank}: multi-rank {reduction}({field_name}) = {result}, "
        f"single-rank = {expected}"
    )
    assert result == pytest.approx(np_reference, rel=1e-15), (
        f"rank={process_props.rank}: multi-rank {reduction}({field_name}) = {result}, "
        f"numpy reference = {np_reference}"
    )
