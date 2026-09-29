# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import grid_refinement as refinement, horizontal as h_grid
from icon4py.model.testing import definitions as test_defs, grid_utils
from icon4py.model.testing.fixtures import backend, cpu_allocator, grid_description

from .. import utils


@pytest.mark.parametrize("dim", dims.horizontal_dims())
@pytest.mark.parametrize(
    "grid_file, expected",
    [
        (test_defs.Grids.R02B04_GLOBAL, False),
        (test_defs.Grids.MCH_OPR_R04B07_DOMAIN01, True),
    ],
)
def test_is_local_area_grid_for_grid_files(
    grid_file: test_defs.GridDescription,
    expected: bool,
    dim: gtx.Dimension,
    backend: gtx_typing.Backend | None,
) -> None:
    grid = grid_utils.get_grid_manager_from_identifier(
        grid_file, 1, True, model_backends.get_allocator(backend)
    ).grid
    refinement_field = grid.refinement_control[dim]
    limited_area = refinement.is_limited_area_grid(refinement_field.ndarray)
    assert isinstance(limited_area, bool)
    assert expected == limited_area


MCH_OPR_R04B07_CELL_BOUNDS: dict[h_grid.Zone, tuple[int, int]] = {
    h_grid.Zone.LATERAL_BOUNDARY: (0, 629),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2: (629, 1244),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3: (1244, 1843),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4: (1843, 2424),
    h_grid.Zone.NUDGING: (2424, 2989),
    h_grid.Zone.INTERIOR: (2989, 10700),
    h_grid.Zone.LOCAL: (0, 10700),
    h_grid.Zone.END: (10700, 10700),
    h_grid.Zone.HALO: (10700, 10700),
    h_grid.Zone.HALO_LEVEL_2: (10700, 10700),
}
MCH_OPR_R04B07_EDGE_BOUNDS: dict[h_grid.Zone, tuple[int, int]] = {
    h_grid.Zone.LATERAL_BOUNDARY: (0, 318),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2: (318, 947),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3: (947, 1258),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4: (1258, 1873),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5: (1873, 2177),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6: (2177, 2776),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7: (2776, 3071),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8: (3071, 3652),
    h_grid.Zone.NUDGING: (3652, 3938),
    h_grid.Zone.NUDGING_LEVEL_2: (3938, 4503),
    h_grid.Zone.INTERIOR: (4503, 16209),
    h_grid.Zone.LOCAL: (0, 16209),
    h_grid.Zone.END: (16209, 16209),
    h_grid.Zone.HALO: (16209, 16209),
    h_grid.Zone.HALO_LEVEL_2: (16209, 16209),
}
MCH_OPR_R04B07_VERTEX_BOUNDS: dict[h_grid.Zone, tuple[int, int]] = {
    h_grid.Zone.LATERAL_BOUNDARY: (0, 318),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2: (318, 629),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3: (629, 933),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4: (933, 1228),
    h_grid.Zone.NUDGING: (1228, 1514),
    h_grid.Zone.INTERIOR: (1514, 5510),
    h_grid.Zone.LOCAL: (0, 5510),
    h_grid.Zone.END: (5510, 5510),
    h_grid.Zone.HALO: (5510, 5510),
    h_grid.Zone.HALO_LEVEL_2: (5510, 5510),
}


MCH_CH_R04B09_CELL_BOUNDS: dict[h_grid.Zone, tuple[int, int]] = {
    h_grid.Zone.LATERAL_BOUNDARY: (0, 850),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2: (850, 1688),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3: (1688, 2511),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4: (2511, 3316),
    h_grid.Zone.NUDGING: (3316, 4104),
    h_grid.Zone.INTERIOR: (4104, 20896),
    h_grid.Zone.LOCAL: (0, 20896),
    h_grid.Zone.END: (20896, 20896),
    h_grid.Zone.HALO: (20896, 20896),
    h_grid.Zone.HALO_LEVEL_2: (20896, 20896),
}
MCH_CH_R04B09_EDGE_BOUNDS: dict[h_grid.Zone, tuple[int, int]] = {
    h_grid.Zone.LATERAL_BOUNDARY: (0, 428),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2: (428, 1278),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3: (1278, 1700),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4: (1700, 2538),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5: (2538, 2954),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6: (2954, 3777),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7: (3777, 4184),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8: (4184, 4989),
    h_grid.Zone.NUDGING: (4989, 5387),
    h_grid.Zone.NUDGING_LEVEL_2: (5387, 6176),
    h_grid.Zone.INTERIOR: (6176, 31558),
    h_grid.Zone.LOCAL: (0, 31558),
    h_grid.Zone.END: (31558, 31558),
    h_grid.Zone.HALO: (31558, 31558),
    h_grid.Zone.HALO_LEVEL_2: (31558, 31558),
}
MCH_CH_R04B09_VERTEX_BOUNDS: dict[h_grid.Zone, tuple[int, int]] = {
    h_grid.Zone.LATERAL_BOUNDARY: (0, 428),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2: (428, 850),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3: (850, 1266),
    h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4: (1266, 1673),
    h_grid.Zone.NUDGING: (1673, 2071),
    h_grid.Zone.INTERIOR: (2071, 10663),
    h_grid.Zone.LOCAL: (0, 10663),
    h_grid.Zone.END: (10663, 10663),
    h_grid.Zone.HALO: (10663, 10663),
    h_grid.Zone.HALO_LEVEL_2: (10663, 10663),
}
#: random invalid values to mark failure for start_/end_index tuple
_FALLBACK_FAIL = (-10, -10)


@pytest.mark.parametrize(
    "grid_description, dim, expected",
    [
        (test_defs.Grids.MCH_OPR_R04B07_DOMAIN01, dims.CellDim, MCH_OPR_R04B07_CELL_BOUNDS),
        (test_defs.Grids.MCH_OPR_R04B07_DOMAIN01, dims.EdgeDim, MCH_OPR_R04B07_EDGE_BOUNDS),
        (test_defs.Grids.MCH_OPR_R04B07_DOMAIN01, dims.VertexDim, MCH_OPR_R04B07_VERTEX_BOUNDS),
        (test_defs.Grids.MCH_CH_R04B09_DSL, dims.CellDim, MCH_CH_R04B09_CELL_BOUNDS),
        (test_defs.Grids.MCH_CH_R04B09_DSL, dims.EdgeDim, MCH_CH_R04B09_EDGE_BOUNDS),
        (test_defs.Grids.MCH_CH_R04B09_DSL, dims.VertexDim, MCH_CH_R04B09_VERTEX_BOUNDS),
    ],
)
def test_compute_domain_bounds_for_limited_area_grid(
    grid_description: test_defs.GridDescription,
    dim: gtx.Dimension,
    expected: dict[h_grid.Zone, tuple[int, int]],
    cpu_allocator: gtx_typing.Allocator,
) -> None:
    grid_manager = grid_utils.get_grid_manager_from_identifier(
        grid_description, 1, True, cpu_allocator
    )

    grid = grid_manager.grid
    assert grid.limited_area, "Test expects limited area grid"
    refinement_field = grid.refinement_control
    decomposition_info = grid_manager.decomposition_info

    start_index, end_index = refinement.compute_domain_bounds(
        dim, refinement_field, decomposition_info=decomposition_info
    )

    for d, v in start_index.items():
        expected_value = expected.get(d.zone, _FALLBACK_FAIL)[0]
        assert v == expected_value, (
            f"Expected start index {expected_value} for domain = {d} , but got {v}"
        )

    for d, v in end_index.items():
        expected_value = expected.get(d.zone, _FALLBACK_FAIL)[1]
        assert v == expected_value, (
            f"Expected end index {expected_value} for domain = {d} , but got {v}"
        )


@pytest.mark.parametrize("dim", dims.horizontal_dims())
def test_compute_domain_bounds_for_global_grid(
    grid_description: test_defs.GridDescription,
    dim: gtx.Dimension,
    cpu_allocator: gtx_typing.Allocator,
) -> None:
    if grid_description.limited_area:
        pytest.skip("This test is not for limited area grids")

    grid_manager = grid_utils.get_grid_manager_from_identifier(
        grid_description, 1, True, cpu_allocator
    )
    grid = grid_manager.grid
    refinement_fields = grid.refinement_control
    decomposition_info = grid_manager.decomposition_info
    start_index, end_index = refinement.compute_domain_bounds(
        dim, refinement_fields, decomposition_info
    )
    for k, v in start_index.items():
        assert isinstance(v, gtx.int32)
        if k.zone.is_halo() or k.zone is h_grid.Zone.END:
            assert v == grid.size[dim], (
                f"Expected start index '{grid.size[dim]}' for '{dim}' in zone '{k.zone}', but got '{v}'"
            )
        else:
            assert v == 0, f"Expected start index '0' for {dim} in {k.zone}, but got '{v}'"

    for k, v in end_index.items():
        assert isinstance(v, gtx.int32)
        if k.zone.is_lateral_boundary() or k.zone.is_nudging():
            assert v == 0, f"Expected end index '0' for '{dim}' in zone '{k.zone}', but got '{v}'"
        else:
            assert v == grid.size[k.dim], (
                f"Expected end index '{grid.size[k.dim]}' for {dim} in {k.zone}, but got '{v}'"
            )
