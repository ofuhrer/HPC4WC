# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.dimension import KDim
from icon4py.model.common.math.vertical_operations import with_boundaries_on_half_levels_on_cells
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc


@gtx.field_operator
def _compute_wgtfac_c(
    z_ifc: fa.CellKField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    return with_boundaries_on_half_levels_on_cells(
        top=(z_ifc(KDim + 1) - z_ifc) / (z_ifc(KDim + 2) - z_ifc),
        interior=(z_ifc(KDim - 1) - z_ifc) / (z_ifc(KDim - 1) - z_ifc(KDim + 1)),
        bottom=(z_ifc(KDim - 1) - z_ifc) / (z_ifc(KDim - 2) - z_ifc),
        nlev=nlev,
    )


# TODO(halungge): missing test?
@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_wgtfac_c(  # noqa: PLR0917 [too-many-positional-arguments]
    wgtfac_c: fa.CellKField[wpfloat],
    z_ifc: fa.CellKField[wpfloat],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_wgtfac_c(
        z_ifc=z_ifc,
        nlev=nlev,
        out=wgtfac_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


def _compute_z1_z2_z3(
    z_ifc: data_alloc.NDArray, i1: int, i2: int, i3: int, i4: int
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    z1 = 0.5 * (z_ifc[:, i2] - z_ifc[:, i1])
    z2 = 0.5 * (z_ifc[:, i2] + z_ifc[:, i3]) - z_ifc[:, i1]
    z3 = 0.5 * (z_ifc[:, i3] + z_ifc[:, i4]) - z_ifc[:, i1]
    return z1, z2, z3


def compute_wgtfacq_c_dsl(
    z_ifc: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    """
    Compute weighting factor for quadratic interpolation to surface.

    Args:
        z_ifc: Field[CellDim, KDim] (half levels), geometric height at the vertical interface of cells.
        nlev: int, last k level
    Returns:
    Field[CellDim, KDim] (full levels)
    """
    array_ns = data_alloc.array_namespace(z_ifc)
    wgtfacq_c = array_ns.zeros((z_ifc.shape[0], nlev + 1))
    wgtfacq_c_dsl = array_ns.zeros((z_ifc.shape[0], nlev))
    z1, z2, z3 = _compute_z1_z2_z3(z_ifc, nlev, nlev - 1, nlev - 2, nlev - 3)

    wgtfacq_c[:, 2] = z1 * z2 / (z2 - z3) / (z1 - z3)
    wgtfacq_c[:, 1] = (z1 - wgtfacq_c[:, 2] * (z1 - z3)) / (z1 - z2)
    wgtfacq_c[:, 0] = 1.0 - (wgtfacq_c[:, 1] + wgtfacq_c[:, 2])

    wgtfacq_c_dsl[:, nlev - 1] = wgtfacq_c[:, 0]
    wgtfacq_c_dsl[:, nlev - 2] = wgtfacq_c[:, 1]
    wgtfacq_c_dsl[:, nlev - 3] = wgtfacq_c[:, 2]

    return wgtfacq_c_dsl[:, -3:]


def compute_wgtfacq_e_dsl(
    *,
    e2c: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    c_lin_e: data_alloc.NDArray,
    wgtfacq_c_dsl: data_alloc.NDArray,
    n_edges: int,
    nlev: int,
    exchange: decomposition.ExchangeRuntime,
) -> data_alloc.NDArray:
    """
    Compute weighting factor for quadratic interpolation to surface.

    Args:
        e2c: Edge to Cell offset
        z_ifc: geometric height at the vertical interface of cells.
        wgtfacq_c_dsl: weighting factor for quadratic interpolation to surface
        c_lin_e: interpolation field
        n_edges: number of edges
        nlev: int, last k level
    Returns:
    Field[EdgeDim, KDim] (full levels)
    """
    array_ns = data_alloc.array_namespace(e2c)
    wgtfacq_e_dsl = array_ns.zeros(shape=(n_edges, nlev + 1))
    z_aux_c = array_ns.zeros((z_ifc.shape[0], 6))
    z1, z2, z3 = _compute_z1_z2_z3(z_ifc, nlev, nlev - 1, nlev - 2, nlev - 3)
    z_aux_c[:, 2] = z1 * z2 / (z2 - z3) / (z1 - z3)
    z_aux_c[:, 1] = (z1 - wgtfacq_c_dsl[:, 0] * (z1 - z3)) / (z1 - z2)
    z_aux_c[:, 0] = 1.0 - (wgtfacq_c_dsl[:, 1] + wgtfacq_c_dsl[:, 0])

    z1, z2, z3 = _compute_z1_z2_z3(z_ifc, 0, 1, 2, 3)
    z_aux_c[:, 5] = z1 * z2 / (z2 - z3) / (z1 - z3)
    z_aux_c[:, 4] = (z1 - z_aux_c[:, 5] * (z1 - z3)) / (z1 - z2)
    z_aux_c[:, 3] = 1.0 - (z_aux_c[:, 4] + z_aux_c[:, 5])

    c_lin_e = c_lin_e[:, :, array_ns.newaxis]
    z_aux_e = array_ns.sum(c_lin_e * z_aux_c[e2c], axis=1)
    exchange.exchange(dims.EdgeDim, z_aux_e, stream=decomposition.BLOCK)

    wgtfacq_e_dsl[:, nlev] = z_aux_e[:, 0]
    wgtfacq_e_dsl[:, nlev - 1] = z_aux_e[:, 1]
    wgtfacq_e_dsl[:, nlev - 2] = z_aux_e[:, 2]

    return wgtfacq_e_dsl[:, -3:]
