# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from icon4py.model.common import dimension as dims, utils as common_utils
from icon4py.model.common.decomposition.definitions import single_node_exchange
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_meta,
    horizontal as h_grid,
    states as grid_states,
)
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_factory,
    rbf_interpolation as rbf,
)
from icon4py.model.common.states import factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions,
    grid_utils as gridtest_utils,
    test_utils as test_helpers,
)
from icon4py.model.testing.fixtures import backend, data_provider, decomposition_info, experiment
from icon4py.model.testing.fixtures.datatest import (
    download_ser_data,
    experiment_description,
    interpolation_savepoint,
    process_props,
)

from .test_rbf_interpolation import RBF_TOLERANCES


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.decomposition import definitions as decomposition
    from icon4py.model.testing import serialbox


V2E_SIZE = 6

C2E_SIZE = 3
E2C_SIZE = 2


interpolation_factories: dict = {}

cell_domain = h_grid.domain(dims.CellDim)
edge_domain = h_grid.domain(dims.EdgeDim)
vertex_domain = h_grid.domain(dims.VertexDim)


def _get_interpolation_factory(
    backend: gtx_typing.Backend | None,
    experiment: definitions.Experiment,
) -> interpolation_factory.InterpolationFieldsFactory:
    registry_key = "_".join((experiment.name, data_alloc.backend_name(backend)))
    factory = interpolation_factories.get(registry_key)
    if not factory:
        geometry = gridtest_utils.get_grid_geometry(backend, experiment.grid, experiment.config)

        factory = interpolation_factory.InterpolationFieldsFactory(
            grid=geometry.grid,
            decomposition_info=geometry._decomposition_info,
            geometry_source=geometry,
            config=experiment.config.interpolation,
            backend=backend,
            metadata=attrs.attrs,
            exchange=single_node_exchange,
        )
        interpolation_factories[registry_key] = factory
    return factory


@pytest.mark.datatest
def test_factory_raises_error_on_unknown_field(
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
    decomposition_info: decomposition.DecompositionInfo,
) -> None:
    geometry = gridtest_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    interpolation_source = interpolation_factory.InterpolationFieldsFactory(
        grid=geometry.grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry,
        config=experiment.config.interpolation,
        backend=backend,
        metadata=attrs.attrs,
        exchange=single_node_exchange,
    )
    with pytest.raises(ValueError, match="Field 'foo' not provided by the source"):
        interpolation_source.get("foo", factory.RetrievalType.METADATA)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_get_c_lin_e(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref = interpolation_savepoint.c_lin_e()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field = factory.get(attrs.C_LIN_E)
    assert field.shape == (grid.num_edges, E2C_SIZE)
    assert test_helpers.dallclose(field.asnumpy(), field_ref.asnumpy())


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_get_geofac_div(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref = interpolation_savepoint.geofac_div()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_DIV)
    assert field.shape == (grid.num_cells, C2E_SIZE)
    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy())


@pytest.mark.datatest
def test_get_geofac_grdiv(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref = interpolation_savepoint.geofac_grdiv()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_GRDIV).asnumpy()
    assert field.shape == (grid.num_edges, 5)
    assert test_helpers.dallclose(field, field_ref.asnumpy())


@pytest.mark.datatest
def test_get_geofac_rot(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref = interpolation_savepoint.geofac_rot()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_ROT).asnumpy()
    horizontal_start = grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    assert field.shape == (grid.num_vertices, V2E_SIZE)
    assert test_helpers.dallclose(
        field_ref.asnumpy()[horizontal_start:, :], field[horizontal_start:, :]
    )


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_get_geofac_n2s(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref = interpolation_savepoint.geofac_n2s()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field = factory.get(attrs.GEOFAC_N2S).asnumpy()
    assert field.shape == (grid.num_cells, 4)
    assert test_helpers.dallclose(field_ref.asnumpy(), field)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_get_geofac_grg(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref = interpolation_savepoint.geofac_grg()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field_x = factory.get(attrs.GEOFAC_GRG_X).asnumpy()
    assert field_x.shape == (grid.num_cells, 4)
    field_y = factory.get(attrs.GEOFAC_GRG_Y).asnumpy()
    assert field_y.shape == (grid.num_cells, 4)
    # less than 1.1e-16 does not pass on mac for mch_ch_r04b09_dsl (but still passes on CI)
    assert test_helpers.dallclose(field_ref[0].asnumpy(), field_x, rtol=1e-11, atol=1.1e-16)
    assert test_helpers.dallclose(field_ref[1].asnumpy(), field_y, rtol=1e-11, atol=1e-16)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_get_mass_conserving_cell_average_weight(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref = interpolation_savepoint.c_bln_avg()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field = factory.get(attrs.C_BLN_AVG).asnumpy()

    assert field.shape == (grid.num_cells, 4)
    assert test_helpers.dallclose(field_ref.asnumpy(), field, rtol=1e-11)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_e_flx_avg(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref = interpolation_savepoint.e_flx_avg()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field = factory.get(attrs.E_FLX_AVG).asnumpy()
    assert field.shape == (grid.num_edges, grid.get_connectivity(dims.E2C2EO).shape[1])
    assert test_helpers.dallclose(field, field_ref.asnumpy(), atol=1e-12)


@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "experiment_description, rtol",
    [
        (definitions.Experiments.MCH_CH_R04B09, 1e-10),
        (definitions.Experiments.EXCLAIM_APE, 1e-11),
        (definitions.Experiments.GAUSS3D, 0),
    ],
)
@pytest.mark.datatest
def test_e_bln_c_s(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
    rtol: float,
) -> None:
    field_ref = interpolation_savepoint.e_bln_c_s()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field = factory.get(attrs.E_BLN_C_S).asnumpy()
    assert field.shape == (grid.num_cells, C2E_SIZE)
    assert test_helpers.dallclose(field_ref.asnumpy(), field, rtol=rtol)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_pos_on_tplane_e_x_y(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref_1 = interpolation_savepoint.pos_on_tplane_e_x()
    field_ref_2 = interpolation_savepoint.pos_on_tplane_e_y()
    factory = _get_interpolation_factory(backend, experiment)
    field_1 = factory.get(attrs.POS_ON_TPLANE_E_X)
    field_2 = factory.get(attrs.POS_ON_TPLANE_E_Y)
    assert test_helpers.dallclose(field_ref_1.asnumpy(), field_1.asnumpy(), atol=1e-8, rtol=1e-9)
    assert test_helpers.dallclose(field_ref_2.asnumpy(), field_2.asnumpy(), atol=1e-8, rtol=1e-9)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_cells_aw_verts(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref = interpolation_savepoint.c_intp()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field = factory.get(attrs.CELL_AW_VERTS).asnumpy()

    assert field.shape == (grid.num_vertices, 6)
    assert test_helpers.dallclose(field_ref.asnumpy(), field)


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_nudgecoeffs(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref = interpolation_savepoint.nudgecoeff_e()
    factory = _get_interpolation_factory(backend, experiment)
    field = factory.get(attrs.NUDGECOEFFS_E)

    assert test_helpers.dallclose(field_ref.asnumpy(), field.asnumpy())


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_rbf_interpolation_coeffs_cell(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref_c1 = interpolation_savepoint.rbf_vec_coeff_c1()
    field_ref_c2 = interpolation_savepoint.rbf_vec_coeff_c2()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field_c1 = factory.get(attrs.RBF_VEC_COEFF_C1).asnumpy()
    field_c2 = factory.get(attrs.RBF_VEC_COEFF_C2).asnumpy()
    horizontal_start = grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

    assert field_c1.shape == (grid.num_cells, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.CELL])
    assert field_c2.shape == (grid.num_cells, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.CELL])
    assert test_helpers.dallclose(
        field_ref_c1.asnumpy()[horizontal_start:],
        field_c1[horizontal_start:],
        atol=RBF_TOLERANCES[dims.CellDim][experiment.name],
    )
    assert test_helpers.dallclose(
        field_ref_c2.asnumpy()[horizontal_start:],
        field_c2[horizontal_start:],
        atol=RBF_TOLERANCES[dims.CellDim][experiment.name],
    )


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_rbf_interpolation_coeffs_edge(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref_e = interpolation_savepoint.rbf_vec_coeff_e()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field_e = factory.get(attrs.RBF_VEC_COEFF_E).asnumpy()
    horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

    assert field_e.shape == (grid.num_edges, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.EDGE])
    assert test_helpers.dallclose(
        field_ref_e.asnumpy()[horizontal_start:],
        field_e[horizontal_start:],
        atol=RBF_TOLERANCES[dims.EdgeDim][experiment.name],
    )


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_rbf_interpolation_coeffs_vertex(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref_v1 = interpolation_savepoint.rbf_vec_coeff_v1()
    field_ref_v2 = interpolation_savepoint.rbf_vec_coeff_v2()
    factory = _get_interpolation_factory(backend, experiment)
    grid = factory.grid
    field_v1 = factory.get(attrs.RBF_VEC_COEFF_V1).asnumpy()
    field_v2 = factory.get(attrs.RBF_VEC_COEFF_V2).asnumpy()
    horizontal_start = grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

    assert field_v1.shape == (grid.num_vertices, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX])
    assert field_v2.shape == (grid.num_vertices, rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX])
    assert test_helpers.dallclose(
        field_ref_v1.asnumpy()[horizontal_start:],
        field_v1[horizontal_start:],
        atol=RBF_TOLERANCES[dims.VertexDim][experiment.name],
    )
    assert test_helpers.dallclose(
        field_ref_v2.asnumpy()[horizontal_start:],
        field_v2[horizontal_start:],
        atol=RBF_TOLERANCES[dims.VertexDim][experiment.name],
    )


@pytest.mark.level("integration")
@pytest.mark.datatest
def test_lsq_pseudoinv(
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend | None,
) -> None:
    field_ref_1 = interpolation_savepoint.lsq_pseudoinv_1().asnumpy()
    field_ref_2 = interpolation_savepoint.lsq_pseudoinv_2().asnumpy()
    factory = _get_interpolation_factory(backend, experiment)
    field = factory.get(attrs.LSQ_PSEUDOINV).asnumpy()
    assert test_helpers.dallclose(field_ref_1, field[:, 0, :], atol=1e-15)
    assert test_helpers.dallclose(field_ref_2, field[:, 1, :], atol=1e-15)
