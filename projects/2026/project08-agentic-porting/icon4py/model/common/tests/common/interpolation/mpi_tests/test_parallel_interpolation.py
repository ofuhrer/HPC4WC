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

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_factory,
)
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
    interpolation_factory_from_savepoint,
    interpolation_savepoint,
    process_props,
)
from ..unit_tests.test_rbf_interpolation import RBF_TOLERANCES


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb


@pytest.mark.level("integration")
@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, intrp_name, rtol, atol",
    [
        (attrs.C_BLN_AVG, "c_bln_avg", 1e-11, 0.0),
        (
            attrs.E_FLX_AVG,
            "e_flx_avg",
            5e-9,
            1e-10,
        ),  # FIXME (halungge): should run with default tolerances
        (attrs.E_BLN_C_S, "e_bln_c_s", 1e-10, 0.0),
        (attrs.POS_ON_TPLANE_E_X, "pos_on_tplane_e_x", 1e-9, 1e-8),
        (attrs.POS_ON_TPLANE_E_Y, "pos_on_tplane_e_y", 1e-9, 1e-8),
    ],
)
def test_distributed_interpolation_with_custom_tolerance(  # noqa: PLR0917 [too-many-positional-arguments]
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: test_defs.Experiment,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    interpolation_factory_from_savepoint: interpolation_factory.InterpolationFieldsFactory,
    attrs_name: str,
    intrp_name: str,
    rtol: float,
    atol: float,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    intp_factory = interpolation_factory_from_savepoint
    field_ref = interpolation_savepoint.__getattribute__(intrp_name)()
    field_ref = field_ref.asnumpy()
    field = intp_factory.get(attrs_name).asnumpy()
    assert test_utils.dallclose(field, field_ref, atol=atol, rtol=rtol), (
        f"comparison of {attrs_name} failed"
    )


# attrs.E_FLX_AVG should work here
@pytest.mark.level("integration")
@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, intrp_name",
    [
        (attrs.C_LIN_E, "c_lin_e"),
        (attrs.NUDGECOEFFS_E, "nudgecoeff_e"),
        (attrs.GEOFAC_DIV, "geofac_div"),
        (attrs.GEOFAC_N2S, "geofac_n2s"),
        (attrs.GEOFAC_GRDIV, "geofac_grdiv"),
        (attrs.CELL_AW_VERTS, "c_intp"),
    ],
)
def test_distributed_interpolation_fields(  # noqa: PLR0917 [too-many-positional-arguments]
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: test_defs.Experiment,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    interpolation_factory_from_savepoint: interpolation_factory.InterpolationFieldsFactory,
    attrs_name: str,
    intrp_name: str,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    intp_factory = interpolation_factory_from_savepoint
    field_ref = interpolation_savepoint.__getattribute__(intrp_name)()
    field_ref = field_ref.asnumpy()
    field = intp_factory.get(attrs_name).asnumpy()
    assert test_utils.dallclose(field, field_ref), f"comparison of {attrs_name} failed"


@pytest.mark.level("integration")
@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_distributed_interpolation_grg(  # noqa: PLR0917 [too-many-positional-arguments]
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: test_defs.Experiment,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    interpolation_factory_from_savepoint: interpolation_factory.InterpolationFieldsFactory,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    intp_factory = interpolation_factory_from_savepoint
    field_ref = interpolation_savepoint.geofac_grg()
    ref_x = field_ref[0].asnumpy()
    ref_y = field_ref[1].asnumpy()
    field_x = intp_factory.get(attrs.GEOFAC_GRG_X).asnumpy()
    field_y = intp_factory.get(attrs.GEOFAC_GRG_Y).asnumpy()

    assert test_utils.dallclose(
        field_x,
        ref_x,
        rtol=1e-11,
        atol=1e-16,
    ), f"comparison of {attrs.GEOFAC_GRG_X} failed"
    assert test_utils.dallclose(
        field_y,
        ref_y,
        rtol=1e-11,
        atol=1e-16,
    ), f"comparison of {attrs.GEOFAC_GRG_Y} failed"


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_distributed_interpolation_geofac_rot(  # noqa: PLR0917 [too-many-positional-arguments]
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: test_defs.Experiment,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    interpolation_factory_from_savepoint: interpolation_factory.InterpolationFieldsFactory,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    parallel_helpers.log_process_properties(process_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = interpolation_factory_from_savepoint
    horizontal_start = factory.grid.start_index(
        h_grid.vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    field_ref = interpolation_savepoint.geofac_rot().asnumpy()
    field = factory.get(attrs.GEOFAC_ROT).asnumpy()
    assert test_utils.dallclose(field[horizontal_start:, :], field_ref[horizontal_start:, :]), (
        f"comparison of {attrs.GEOFAC_ROT} failed"
    )


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "attrs_name, intrp_name",
    [
        (attrs.RBF_VEC_COEFF_C1, "rbf_vec_coeff_c1"),
        (attrs.RBF_VEC_COEFF_C2, "rbf_vec_coeff_c2"),
        (attrs.RBF_VEC_COEFF_E, "rbf_vec_coeff_e"),
        (attrs.RBF_VEC_COEFF_V1, "rbf_vec_coeff_v1"),
        (attrs.RBF_VEC_COEFF_V2, "rbf_vec_coeff_v2"),
    ],
)
def test_distributed_interpolation_rbf(  # noqa: PLR0917 [too-many-positional-arguments]
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: test_defs.Experiment,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    interpolation_factory_from_savepoint: interpolation_factory.InterpolationFieldsFactory,
    attrs_name: str,
    intrp_name: str,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    parallel_helpers.log_process_properties(process_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = interpolation_factory_from_savepoint
    field_ref = interpolation_savepoint.__getattribute__(intrp_name)()
    field = factory.get(attrs_name)
    dim = field.domain.dims[0]
    assert test_utils.dallclose(
        field.asnumpy(), field_ref.asnumpy(), atol=RBF_TOLERANCES[dim][experiment.name]
    )


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_distributed_interpolation_lsq_pseudoinv(  # noqa: PLR0917 [too-many-positional-arguments]
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: test_defs.Experiment,
    process_props: decomp_defs.ProcessProperties,
    decomposition_info: decomp_defs.DecompositionInfo,
    interpolation_factory_from_savepoint: interpolation_factory.InterpolationFieldsFactory,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    parallel_helpers.log_process_properties(process_props)
    parallel_helpers.log_local_field_size(decomposition_info)
    factory = interpolation_factory_from_savepoint
    field_ref_1 = interpolation_savepoint.lsq_pseudoinv_1().asnumpy()
    field_ref_2 = interpolation_savepoint.lsq_pseudoinv_2().asnumpy()
    field = factory.get(attrs.LSQ_PSEUDOINV).asnumpy()
    assert test_utils.dallclose(field[:, 0, :], field_ref_1, atol=1e-15)
    assert test_utils.dallclose(field[:, 1, :], field_ref_2, atol=1e-15)
