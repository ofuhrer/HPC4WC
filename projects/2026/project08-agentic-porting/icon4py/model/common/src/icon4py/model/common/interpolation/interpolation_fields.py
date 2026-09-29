# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
import math
from typing import Final

from gt4py import next as gtx
from gt4py.next import where

import icon4py.model.common.field_type_aliases as fa
import icon4py.model.common.type_alias as ta
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.dimension import C2E, V2E
from icon4py.model.common.grid import gridfile, icon as icon_grid
from icon4py.model.common.grid.geometry_stencils import compute_primal_cart_normal
from icon4py.model.common.math import projection
from icon4py.model.common.utils import data_allocation as data_alloc


MISSING: Final[int] = gridfile.GridFile.INVALID_INDEX

logger = logging.Logger(__file__)


def compute_c_lin_e(
    edge_cell_length: data_alloc.NDArray,
    inv_dual_edge_length: data_alloc.NDArray,
    edge_owner_mask: data_alloc.NDArray,
    horizontal_start: gtx.int32,
) -> data_alloc.NDArray:
    """
    Compute E2C average inverse distance.

    Args:
        edge_cell_length: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]
        inv_dual_edge_length: ndarray, inverse dual edge length, numpy array representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        edge_owner_mask: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim], bool]boolean field, True for all edges owned by this compute node
        horizontal_start: start index from the field is computed: c_lin_e is not calculated for the first boundary layer
    Returns: c_lin_e: numpy array, representing gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]

    """
    array_ns = data_alloc.array_namespace(edge_cell_length)
    c_lin_e_ = edge_cell_length[:, 1] * inv_dual_edge_length
    c_lin_e = array_ns.transpose(array_ns.vstack((c_lin_e_, (1.0 - c_lin_e_))))
    c_lin_e[0:horizontal_start, :] = 0.0
    mask = array_ns.transpose(array_ns.tile(edge_owner_mask, (2, 1)))
    res = array_ns.where(mask, c_lin_e, 0.0)
    return res


@gtx.field_operator
def compute_geofac_div(
    primal_edge_length: fa.EdgeField[ta.wpfloat],
    edge_orientation: gtx.Field[[dims.CellDim, dims.C2EDim], ta.wpfloat],
    area: fa.CellField[ta.wpfloat],
) -> gtx.Field[[dims.CellDim, dims.C2EDim], ta.wpfloat]:
    """
    Compute geometrical factor for divergence.

    Args:
        primal_edge_length:
        edge_orientation:
        area:

    Returns:
    """
    geofac_div = primal_edge_length(C2E) * edge_orientation / area
    return geofac_div


@gtx.field_operator
def compute_geofac_rot(
    dual_edge_length: fa.EdgeField[ta.wpfloat],
    edge_orientation: gtx.Field[[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    dual_area: fa.VertexField[ta.wpfloat],
    owner_mask: fa.VertexField[bool],
) -> gtx.Field[[dims.VertexDim, dims.V2EDim], ta.wpfloat]:
    """
    Compute geometrical factor for curl.

    Args:
        dual_edge_length:
        edge_orientation:
        dual_area:
        owner_mask:

    Returns:
    """
    geofac_rot = where(owner_mask, dual_edge_length(V2E) * edge_orientation / dual_area, 0.0)
    return geofac_rot


def compute_geofac_n2s(
    *,
    dual_edge_length: data_alloc.NDArray,
    geofac_div: data_alloc.NDArray,
    c2e: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    c2e2c: data_alloc.NDArray,
    horizontal_start: gtx.int32,
) -> data_alloc.NDArray:
    """
    Compute geometric factor for nabla2-scalar.

    Args:
        dual_edge_length: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        geofac_div: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        c2e: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        e2c: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        c2e2c: ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2E2CDim], gtx.int32]
        horizontal_start: start index from where the field is computed

    Returns:
        geometric factor for nabla2-scalar, Field[CellDim, C2E2CODim]
    """
    array_ns = data_alloc.array_namespace(dual_edge_length)
    num_cells = c2e.shape[0]
    geofac_n2s = array_ns.zeros([num_cells, 4])
    index = array_ns.transpose(
        array_ns.vstack(
            (
                array_ns.arange(num_cells),
                array_ns.arange(num_cells),
                array_ns.arange(num_cells),
            )
        )
    )
    mask = e2c[c2e, 0] == index
    geofac_n2s[horizontal_start:, 0] = geofac_n2s[horizontal_start:, 0] - array_ns.sum(
        mask[horizontal_start:] * (geofac_div / dual_edge_length[c2e])[horizontal_start:], axis=1
    )
    mask = e2c[c2e, 1] == index
    geofac_n2s[horizontal_start:, 0] = geofac_n2s[horizontal_start:, 0] + array_ns.sum(
        mask[horizontal_start:] * (geofac_div / dual_edge_length[c2e])[horizontal_start:], axis=1
    )
    mask = e2c[c2e, 0] == c2e2c
    geofac_n2s[horizontal_start:, 1:] = (
        geofac_n2s[horizontal_start:, 1:]
        - mask[horizontal_start:, :] * (geofac_div / dual_edge_length[c2e])[horizontal_start:, :]
    )
    mask = e2c[c2e, 1] == c2e2c
    geofac_n2s[horizontal_start:, 1:] = (
        geofac_n2s[horizontal_start:, 1:]
        + mask[horizontal_start:, :] * (geofac_div / dual_edge_length[c2e])[horizontal_start:, :]
    )
    return geofac_n2s


def compute_geofac_grg(
    *,
    primal_normal_cell_x: data_alloc.NDArray,
    primal_normal_cell_y: data_alloc.NDArray,
    owner_mask: data_alloc.NDArray,
    geofac_div: data_alloc.NDArray,
    c_lin_e: data_alloc.NDArray,
    c2e: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    c2e2c: data_alloc.NDArray,
    horizontal_start: gtx.int32,
    exchange: decomposition.ExchangeRuntime,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(primal_normal_cell_x)
    owned = array_ns.stack((owner_mask, owner_mask, owner_mask)).T
    inv_neighbor_index = _create_inverse_neighbor_index(e2c, c2e)
    primal_normal_ec_u = array_ns.where(owned, primal_normal_cell_x[c2e, inv_neighbor_index], 0.0)
    primal_normal_ec_v = array_ns.where(owned, primal_normal_cell_y[c2e, inv_neighbor_index], 0.0)

    exchange.exchange(
        dims.CellDim, primal_normal_ec_u, primal_normal_ec_v, stream=decomposition.BLOCK
    )

    num_cells = c2e.shape[0]
    targ_local_size = c2e.shape[1] + 1
    target_shape = (num_cells, targ_local_size)
    geofac_grg_x = array_ns.zeros(target_shape)
    geofac_grg_y = array_ns.zeros(target_shape)

    tmp = geofac_div * c_lin_e[c2e, inv_neighbor_index]
    geofac_grg_x[horizontal_start:, 0] = array_ns.sum(primal_normal_ec_u * tmp, axis=1)[
        horizontal_start:
    ]
    geofac_grg_y[horizontal_start:, 0] = array_ns.sum(primal_normal_ec_v * tmp, axis=1)[
        horizontal_start:
    ]

    for k in range(e2c.shape[1]):
        mask = (e2c[c2e, k] == c2e2c)[horizontal_start:, :]
        geofac_grg_x[horizontal_start:, 1:] = (
            geofac_grg_x[horizontal_start:, 1:]
            + mask * (primal_normal_ec_u * geofac_div * c_lin_e[c2e, k])[horizontal_start:, :]
        )
        geofac_grg_y[horizontal_start:, 1:] = (
            geofac_grg_y[horizontal_start:, 1:]
            + mask * (primal_normal_ec_v * geofac_div * c_lin_e[c2e, k])[horizontal_start:, :]
        )
    exchange.exchange(dims.CellDim, geofac_grg_x, geofac_grg_y, stream=decomposition.BLOCK)

    return geofac_grg_x, geofac_grg_y


def compute_geofac_grdiv(
    *,
    geofac_div: data_alloc.NDArray,
    inv_dual_edge_length: data_alloc.NDArray,
    owner_mask: data_alloc.NDArray,
    c2e: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    e2c2e: data_alloc.NDArray,
    horizontal_start: gtx.int32,
) -> data_alloc.NDArray:
    """
    Compute geometrical factor for gradient of divergence (triangles only).

    Args:
        geofac_div:  ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        inv_dual_edge_length: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        owner_mask:  ndarray, representing a gtx.Field[gtx.Dims[EdgeDim], bool]
        c2e:  ndarray, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        e2c: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        e2c2e: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2C2EDim], gtx.int32]
        horizontal_start:

    Returns:
        geofac_grdiv:  ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2C2EODim], ta.wpfloat]
    """
    array_ns = data_alloc.array_namespace(geofac_div)
    num_edges = e2c.shape[0]
    geofac_grdiv = array_ns.zeros([num_edges, 1 + 2 * e2c.shape[1]])
    index = array_ns.arange(horizontal_start, num_edges)
    for j in range(c2e.shape[1]):
        mask = array_ns.where(
            c2e[e2c[horizontal_start:, 1], j] == index, owner_mask[horizontal_start:], False
        )
        geofac_grdiv[horizontal_start:, 0] = array_ns.where(
            mask, geofac_div[e2c[horizontal_start:, 1], j], geofac_grdiv[horizontal_start:, 0]
        )
    for j in range(c2e.shape[1]):
        mask = array_ns.where(
            c2e[e2c[horizontal_start:, 0], j] == index, owner_mask[horizontal_start:], False
        )
        geofac_grdiv[horizontal_start:, 0] = array_ns.where(
            mask,
            (geofac_grdiv[horizontal_start:, 0] - geofac_div[e2c[horizontal_start:, 0], j])
            * inv_dual_edge_length[horizontal_start:],
            geofac_grdiv[horizontal_start:, 0],
        )
    for j in range(e2c.shape[1]):
        for k in range(c2e.shape[1]):
            mask = c2e[e2c[horizontal_start:, 0], k] == e2c2e[horizontal_start:, j]
            geofac_grdiv[horizontal_start:, e2c.shape[1] - 1 + j] = array_ns.where(
                mask,
                -geofac_div[e2c[horizontal_start:, 0], k] * inv_dual_edge_length[horizontal_start:],
                geofac_grdiv[horizontal_start:, e2c.shape[1] - 1 + j],
            )
            mask = c2e[e2c[horizontal_start:, 1], k] == e2c2e[horizontal_start:, e2c.shape[1] + j]
            geofac_grdiv[horizontal_start:, 2 * e2c.shape[1] - 1 + j] = array_ns.where(
                mask,
                geofac_div[e2c[horizontal_start:, 1], k] * inv_dual_edge_length[horizontal_start:],
                geofac_grdiv[horizontal_start:, 2 * e2c.shape[1] - 1 + j],
            )
    return geofac_grdiv


def _rotate_latlon(
    lat: data_alloc.NDArray,
    lon: data_alloc.NDArray,
    pollat: data_alloc.NDArray,
    pollon: data_alloc.NDArray,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """
    (Compute rotation of lattitude and longitude.)

    Rotates latitude and longitude for more accurate computation
    of bilinear interpolation

    Args:
        lat: scalar or numpy array
        lon: scalar or numpy array
        pollat: scalar or numpy array
        pollon: scalar or numpy array

    Returns:
        rotlat:
        rotlon:
    """
    array_ns = data_alloc.array_namespace(lat)
    rotlat = array_ns.arcsin(
        array_ns.sin(lat) * array_ns.sin(pollat)
        + array_ns.cos(lat) * array_ns.cos(pollat) * array_ns.cos(lon - pollon)
    )
    rotlon = array_ns.arctan2(
        array_ns.cos(lat) * array_ns.sin(lon - pollon),
        (
            array_ns.cos(lat) * array_ns.sin(pollat) * array_ns.cos(lon - pollon)
            - array_ns.sin(lat) * array_ns.cos(pollat)
        ),
    )

    return (rotlat, rotlon)


def _weighting_factors(
    ytemp: data_alloc.NDArray,
    xtemp: data_alloc.NDArray,
    yloc: data_alloc.NDArray,
    xloc: data_alloc.NDArray,
    wgt_loc: ta.wpfloat,
) -> data_alloc.NDArray:
    """
        Compute weighting factors.
        The weighting factors are based on the requirement that sum(w(i)*x(i)) = 0
        and sum(w(i)*y(i)) = 0, which ensures that linear horizontal gradients
        are not aliased into a checkerboard pattern between upward- and downward
        directed cells. The third condition is sum(w(i)) = 1., and the weight
        of the local point is 0.5 (see above). Analytical elimination yields...

    # TODO(ajocksch): computation different for Torus grids see mo_intp_coeffs.f90
    # The function weighting_factors does not exist in the Fortran code, the
    # Fortran is organised differently with code duplication

        Args:
            ytemp:  \\   numpy array of size [[3, flexible], ta.wpfloat]
            xtemp:  //
            yloc:   \\   numpy array of size [[flexible], ta.wpfloat]
            xloc:   //
            wgt_loc:

        Returns:
            wgt: numpy array of size [[3, flexible], ta.wpfloat]
    """
    array_ns = data_alloc.array_namespace(ytemp)
    rotate = functools.partial(_rotate_latlon)

    pollat = array_ns.where(yloc >= 0.0, yloc - math.pi * 0.5, yloc + math.pi * 0.5)
    pollon = xloc
    (yloc, xloc) = rotate(yloc, xloc, pollat, pollon)
    x = array_ns.zeros([ytemp.shape[0], ytemp.shape[1]])
    y = array_ns.zeros([ytemp.shape[0], ytemp.shape[1]])
    wgt = array_ns.zeros([ytemp.shape[0], ytemp.shape[1]])

    for i in range(ytemp.shape[0]):
        (ytemp[i], xtemp[i]) = rotate(ytemp[i], xtemp[i], pollat, pollon)
        y[i] = ytemp[i] - yloc
        x[i] = xtemp[i] - xloc
        # This is needed when the date line is crossed
        x[i] = array_ns.where(x[i] > 3.5, x[i] - math.pi * 2, x[i])
        x[i] = array_ns.where(x[i] < -3.5, x[i] + math.pi * 2, x[i])

    mask = array_ns.logical_and(abs(x[1] - x[0]) > 1.0e-11, abs(y[2] - y[0]) > 1.0e-11)
    wgt_1_no_mask = (
        1.0
        / ((y[1] - y[0]) - (x[1] - x[0]) * (y[2] - y[0]) / (x[2] - x[0]))
        * (1.0 - wgt_loc)
        * (-y[0] + x[0] * (y[2] - y[0]) / (x[2] - x[0]))
    )
    wgt[2] = array_ns.where(
        mask,
        1.0
        / ((y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]) / (x[1] - x[0]))
        * (1.0 - wgt_loc)
        * (-y[0] + x[0] * (y[1] - y[0]) / (x[1] - x[0])),
        (-(1.0 - wgt_loc) * x[0] - wgt_1_no_mask * (x[1] - x[0])) / (x[2] - x[0]),
    )
    wgt[1] = array_ns.where(
        mask,
        (-(1.0 - wgt_loc) * x[0] - wgt[2] * (x[2] - x[0])) / (x[1] - x[0]),
        wgt_1_no_mask,
    )
    wgt[0] = 1.0 - wgt[1] - wgt[2] if wgt_loc == 0.0 else 1.0 - wgt_loc - wgt[1] - wgt[2]
    return wgt


def _compute_c_bln_avg(
    c2e2c: data_alloc.NDArray,
    lat: data_alloc.NDArray,
    lon: data_alloc.NDArray,
    divergence_averaging_central_cell_weight: ta.wpfloat,
    horizontal_start: gtx.int32,
) -> data_alloc.NDArray:
    """
    Compute bilinear cell average weight.

    Args:
        divergence_averaging_central_cell_weight:
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[CellDim], bool]
        c2e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, C2E2CDim], gtx.int32]
        lat: \\ numpy array, representing a gtx.Field[gtx.Dims[CellDim], ta.wpfloat]
        lon: //
        horizontal_start:

    Returns:
        c_bln_avg: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
    """
    array_ns = data_alloc.array_namespace(c2e2c)
    num_cells = c2e2c.shape[0]
    ytemp = array_ns.zeros([c2e2c.shape[1], num_cells - horizontal_start])
    xtemp = array_ns.zeros([c2e2c.shape[1], num_cells - horizontal_start])

    for i in range(ytemp.shape[0]):
        ytemp[i] = lat[c2e2c[horizontal_start:, i]]
        xtemp[i] = lon[c2e2c[horizontal_start:, i]]

    wgt = _weighting_factors(
        ytemp,
        xtemp,
        lat[horizontal_start:],
        lon[horizontal_start:],
        divergence_averaging_central_cell_weight,
    )
    c_bln_avg = array_ns.zeros((c2e2c.shape[0], c2e2c.shape[1] + 1))
    c_bln_avg[horizontal_start:, 0] = divergence_averaging_central_cell_weight
    c_bln_avg[horizontal_start:, 1] = wgt[0]
    c_bln_avg[horizontal_start:, 2] = wgt[1]
    c_bln_avg[horizontal_start:, 3] = wgt[2]
    return c_bln_avg


def _force_mass_conservation_to_c_bln_avg(
    *,
    c2e2c0: data_alloc.NDArray,
    c_bln_avg: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    cell_owner_mask: data_alloc.NDArray,
    divergence_averaging_central_cell_weight: ta.wpfloat,
    horizontal_start: gtx.int32,
    exchange: decomposition.ExchangeRuntime,
    niter: int = 1000,
) -> data_alloc.NDArray:
    """
    Iteratively enforce mass conservation to the input field c_bln_avg.

    Mass conservation is enforced by the following condition:
    The three point divergence calculated on any given grid point is used with a total factor of 1.

    Practically, the sum of the  bilinear cell weights  applied to a cell from all neighbors times its area should be exactly one.

    The weights are adjusted iteratively by the condition up to a max of niter iterations

    Args:
        c2e2c0: cell to cell connectivity
        c_bln_avg: input field: bilinear cell weight average
        cell_areas: area of cells
        cell_owner_mask:
        divergence_averaging_central_cell_weight: configured central weight
        horizontal_start:
        niter: max number of iterations

    Returns:

    """
    array_ns = data_alloc.array_namespace(c2e2c0)

    def _compute_local_weights(
        c_bln_avg, cell_areas, c2e2c0, inverse_neighbor_idx
    ) -> data_alloc.NDArray:
        """
        Compute the total weight which each local point contributes to the sum.

        Args:
            c_bln_avg: ndarray representing a weight field of (CellDim, C2E2C0Dim)
            inverse_neighbor_index: Sequence of to access all weights of a local cell in a field of shape (CellDim, C2E2C0Dim)

        Returns: ndarray of CellDim, containing the sum of weigh contributions for each local cell index

        """
        weights = array_ns.sum(c_bln_avg[c2e2c0, inverse_neighbor_idx] * cell_areas[c2e2c0], axis=1)
        return weights

    def _compute_residual_to_mass_conservation(
        owner_mask: data_alloc.NDArray,
        local_weight: data_alloc.NDArray,
        cell_area: data_alloc.NDArray,
    ) -> data_alloc.NDArray:
        """The local_weight weighted by the area should be 1. We compute how far we are off that weight."""
        horizontal_size = local_weight.shape[0]
        assert horizontal_size == owner_mask.shape[0], "Fields do not have the same shape"
        assert horizontal_size == cell_area.shape[0], "Fields do not have the same shape"
        residual = array_ns.where(owner_mask, local_weight / cell_area - 1.0, 0.0)
        return residual

    def _apply_correction(
        c_bln_avg: data_alloc.NDArray,
        residual: data_alloc.NDArray,
        c2e2c0: data_alloc.NDArray,
        divergence_averaging_central_cell_weight: float,
        horizontal_start: gtx.int32,
    ) -> data_alloc.NDArray:
        """Apply correction to local weigths based on the computed residuals."""
        maxwgt_loc = divergence_averaging_central_cell_weight + 0.003
        minwgt_loc = divergence_averaging_central_cell_weight - 0.003
        relax_coeff = 0.46
        c_bln_avg[horizontal_start:, :] = (
            c_bln_avg[horizontal_start:, :] - relax_coeff * residual[c2e2c0][horizontal_start:, :]
        )
        local_weight = array_ns.sum(c_bln_avg, axis=1) - 1.0

        c_bln_avg[horizontal_start:, :] = c_bln_avg[horizontal_start:, :] - (
            0.25 * local_weight[horizontal_start:, array_ns.newaxis]
        )

        # avoid runaway condition:
        c_bln_avg[horizontal_start:, 0] = array_ns.maximum(
            c_bln_avg[horizontal_start:, 0], minwgt_loc
        )
        c_bln_avg[horizontal_start:, 0] = array_ns.minimum(
            c_bln_avg[horizontal_start:, 0], maxwgt_loc
        )
        return c_bln_avg

    def _enforce_mass_conservation(
        c_bln_avg: data_alloc.NDArray,
        residual: data_alloc.NDArray,
        owner_mask: data_alloc.NDArray,
        horizontal_start: gtx.int32,
    ) -> data_alloc.NDArray:
        """Enforce the mass conservation condition on the local cells by forcefully subtracting the
        residual from the central field contribution."""
        c_bln_avg[horizontal_start:, 0] = array_ns.where(
            owner_mask[horizontal_start:],
            c_bln_avg[horizontal_start:, 0] - residual[horizontal_start:],
            c_bln_avg[horizontal_start:, 0],
        )
        return c_bln_avg

    local_summed_weights = array_ns.zeros(c_bln_avg.shape[0])
    residual = array_ns.zeros(c_bln_avg.shape[0])
    inverse_neighbor_idx = _create_inverse_neighbor_index(c2e2c0, c2e2c0)

    for iteration in range(niter):
        local_summed_weights[horizontal_start:] = _compute_local_weights(
            c_bln_avg, cell_areas, c2e2c0, inverse_neighbor_idx
        )[horizontal_start:]

        residual[horizontal_start:] = _compute_residual_to_mass_conservation(
            cell_owner_mask, local_summed_weights, cell_areas
        )[horizontal_start:]

        exchange.exchange(dims.CellDim, residual, stream=decomposition.BLOCK)

        # in practice the convergence criteria is never reached before the niter is reached for (niter <= 1000)
        # so when parallelizing we opt for disableing the criteria instead of doing an inefficient
        # global reduction. (We assume that there is no convergence criteria for the very same reason in the
        # original icon code.
        # for max_ = array_ns.max(residual)
        max_ = 1.0
        if iteration >= (niter - 1) or max_ < 1e-9:
            logger.debug(f"number of iterations: {iteration} - max residual disabled")
            c_bln_avg = _enforce_mass_conservation(
                c_bln_avg, residual, cell_owner_mask, horizontal_start
            )
        else:
            c_bln_avg = _apply_correction(
                c_bln_avg=c_bln_avg,
                residual=residual,
                c2e2c0=c2e2c0,
                divergence_averaging_central_cell_weight=divergence_averaging_central_cell_weight,
                horizontal_start=horizontal_start,
            )

        exchange.exchange(dims.CellDim, c_bln_avg, stream=decomposition.BLOCK)

    return c_bln_avg


def _compute_uniform_c_bln_avg(
    c2e2c: data_alloc.NDArray,
    divergence_averaging_central_cell_weight: ta.wpfloat,
    horizontal_start: gtx.int32,
) -> data_alloc.NDArray:
    """
    Compute bilinear cell average weight for a torus grid.

    Args:
        c2e2c
        divergence_averaging_central_cell_weight: weight for local / center contribution
        horizontal_start: start index of the horizontal domain

    Returns:
        c_bln_avg
    """
    array_ns = data_alloc.array_namespace(c2e2c)
    local_weight = divergence_averaging_central_cell_weight
    neighbor_weight = (1.0 - divergence_averaging_central_cell_weight) / 3.0

    weights = array_ns.asarray([local_weight, neighbor_weight, neighbor_weight, neighbor_weight])

    c_bln_avg = array_ns.tile(
        weights,
        (c2e2c.shape[0], 1),
    )

    return c_bln_avg


def compute_mass_conserving_bilinear_cell_average_weight(
    *,
    c2e2c0: data_alloc.NDArray,
    lat: data_alloc.NDArray,
    lon: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    cell_owner_mask: data_alloc.NDArray,
    divergence_averaging_central_cell_weight: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_start_level_3: gtx.int32,
    exchange: decomposition.ExchangeRuntime,
) -> data_alloc.NDArray:
    c_bln_avg = _compute_c_bln_avg(
        c2e2c0[:, 1:],
        lat,
        lon,
        divergence_averaging_central_cell_weight,
        horizontal_start,
    )

    exchange.exchange(dims.CellDim, c_bln_avg, stream=decomposition.BLOCK)

    return _force_mass_conservation_to_c_bln_avg(
        c2e2c0=c2e2c0,
        c_bln_avg=c_bln_avg,
        cell_areas=cell_areas,
        cell_owner_mask=cell_owner_mask,
        divergence_averaging_central_cell_weight=divergence_averaging_central_cell_weight,
        horizontal_start=horizontal_start_level_3,
        exchange=exchange,
    )


def compute_mass_conserving_bilinear_cell_average_weight_torus(
    *,
    c2e2c0: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    cell_owner_mask: data_alloc.NDArray,
    divergence_averaging_central_cell_weight: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_start_level_3: gtx.int32,
    exchange: decomposition.ExchangeRuntime,
) -> data_alloc.NDArray:
    c_bln_avg = _compute_uniform_c_bln_avg(
        c2e2c0[:, 1:],
        divergence_averaging_central_cell_weight,
        horizontal_start,
    )
    exchange.exchange(dims.CellDim, c_bln_avg, stream=decomposition.BLOCK)
    # TODO(msimberg): Exact result for torus without the following. 1e-16 error
    # with the the following. Is it needed?
    return _force_mass_conservation_to_c_bln_avg(
        c2e2c0=c2e2c0,
        c_bln_avg=c_bln_avg,
        cell_areas=cell_areas,
        cell_owner_mask=cell_owner_mask,
        divergence_averaging_central_cell_weight=divergence_averaging_central_cell_weight,
        horizontal_start=horizontal_start_level_3,
        exchange=exchange,
    )


def _create_inverse_neighbor_index(
    source_offset: data_alloc.NDArray, inverse_offset: data_alloc.NDArray
) -> data_alloc.NDArray:
    """
    The inverse neighbor index determines the position of a central element c_1
    in the neighbor table of its neighbors:

    For example: for let e_1, e_2, e_3 be the neighboring edges of a cell: c2e(c_1) will
    map  c_1 -> (e_1, e_2,e_3) then in the inverse lookup table e2c the
    neighborhoods of e_1, e_2, e_3 will all contain c_1 in some position.
    The inverse neighbor index tells what position that is. It essentially says
    "I am neighbor number x \\in (0,1) of my neighboring edges"

    In mathematical notation:
        given a and a2b, find b2a such that [b, b2a] = a for all a, a2b where a2b[a, b] != MISSING
        assume a and b are dimensions
        source_offset = [a, a2b]
        inverse_offset = [b, b2a]
        inv_neighbor_idx = [a, b2a], this is the output

    Args:
        source_offset: [a, a2b]
        inverse_offset: [b, b2a]

    Returns:
        ndarray of the same shape as target_offset: [a, b2a]

    """
    array_ns = data_alloc.array_namespace(source_offset)
    inv_neighbor_idx = array_ns.full(inverse_offset.shape, fill_value=MISSING, dtype=gtx.int32)
    n_inv_elem, n_inv_neighbors = inverse_offset.shape
    # loop over all neighbors, denoted by i, of inverse_offset. For each neighbor i, find which neighbor of the source_offset goes back to the inverse_offset itself
    for i in range(n_inv_neighbors):
        inverse_neighbor = inverse_offset[:, i]
        valid_mask = inverse_neighbor >= 0
        valid_source_offset = source_offset[
            inverse_neighbor[valid_mask], :
        ]  # (n_valid, n_source_neighbors)
        # find for each valid neighbor of an inverse_offset element, which neighbor of the source_offset matches the index of the inverse_offset
        inv_elem_indices = array_ns.arange(n_inv_elem)[valid_mask]
        matches = array_ns.equal(
            valid_source_offset, inv_elem_indices[:, array_ns.newaxis]
        )  # (n_valid, n_source_neighbors)
        # we need has_match here before argmax, because argmax will return 0 if there is no match, which is a valid index, but we want to ignore it in that case
        has_match = array_ns.any(matches, axis=1)
        first_match = array_ns.argmax(matches, axis=1)
        inv_neighbor_idx[inv_elem_indices[has_match], i] = first_match[has_match].astype(gtx.int32)

    return inv_neighbor_idx


def compute_e_flx_avg(
    *,
    c_bln_avg: data_alloc.NDArray,
    geofac_div: data_alloc.NDArray,
    owner_mask: data_alloc.NDArray,
    primal_cart_normal_x: data_alloc.NDArray,
    primal_cart_normal_y: data_alloc.NDArray,
    primal_cart_normal_z: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    c2e: data_alloc.NDArray,
    c2e2c: data_alloc.NDArray,
    e2c2e: data_alloc.NDArray,
    horizontal_start_p3: gtx.int32,
    horizontal_start_p4: gtx.int32,
    exchange: decomposition.ExchangeRuntime,
) -> data_alloc.NDArray:
    """
    Compute edge flux average.

    FIXME (@halungge) the correctness of this function depends on the local order of the e2c2e connectivity fields

    Args:
        c_bln_avg: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        geofac_div: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], bool]
        primal_cart_normal: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        c2e: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        c2e2c: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2E2CDim], gtx.int32]
        e2c2e: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2C2EDim], gtx.int32]
        horizontal_start_p3:
        horizontal_start_p4:

    Returns:
        e_flx_avg: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2C2EODim], ta.wpfloat]
    """
    array_ns = data_alloc.array_namespace(c_bln_avg)
    primal_cart_normal = compute_primal_cart_normal(
        primal_cart_normal_x,
        primal_cart_normal_y,
        primal_cart_normal_z,
    )
    diamond_shape = e2c2e.shape[1]
    num_edges = e2c.shape[0]
    num_cells = c2e2c.shape[0]
    llb = 0
    e_flx_avg = array_ns.zeros((num_edges, diamond_shape + 1))
    index = array_ns.arange(llb, num_cells)
    inv_neighbor_id = array_ns.full((num_cells - llb, c2e2c.shape[1]), MISSING, dtype=int)
    for i in range(c2e2c.shape[1]):
        for j in range(c2e2c.shape[1]):
            inv_neighbor_id[:, j] = array_ns.where(
                array_ns.logical_and(c2e2c[c2e2c[llb:, j], i] == index, c2e2c[llb:, j] >= 0),
                i,
                inv_neighbor_id[:, j],
            )

    llb = horizontal_start_p3
    index = array_ns.arange(llb, num_edges)
    for j in range(c2e.shape[1]):
        for i in range(2):
            e_flx_avg[llb:, i + 1] = array_ns.where(
                owner_mask[llb:],
                array_ns.where(
                    c2e[e2c[llb:, 0], j] == index,
                    c_bln_avg[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], j] + 1]
                    * geofac_div[e2c[llb:, 0], array_ns.mod(i + j + 1, 3)]
                    / geofac_div[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], j]],
                    e_flx_avg[llb:, i + 1],
                ),
                e_flx_avg[llb:, i + 1],
            )
            e_flx_avg[llb:, i + 3] = array_ns.where(
                owner_mask[llb:],
                array_ns.where(
                    c2e[e2c[llb:, 0], j] == index,
                    c_bln_avg[e2c[llb:, 0], 1 + j]
                    * geofac_div[
                        e2c[llb:, 1], array_ns.mod(inv_neighbor_id[e2c[llb:, 0], j] + i + 1, 3)
                    ]
                    / geofac_div[e2c[llb:, 0], j],
                    e_flx_avg[llb:, i + 3],
                ),
                e_flx_avg[llb:, i + 3],
            )

    exchange.exchange(dims.EdgeDim, e_flx_avg, stream=decomposition.BLOCK)

    # the icon prescribed order dependency is probably due to these magic numbers...
    iie = array_ns.full(e2c2e.shape, MISSING, dtype=int)
    iie[:, 0] = array_ns.where(
        e2c[e2c2e[:, 0], 0] == e2c[:, 0],
        2,
        array_ns.where(e2c[e2c2e[:, 0], 1] == e2c[:, 0], 4, MISSING),
    )

    iie[:, 1] = array_ns.where(
        e2c[e2c2e[:, 1], 0] == e2c[:, 0],
        1,
        array_ns.where(e2c[e2c2e[:, 1], 1] == e2c[:, 0], 3, MISSING),
    )

    iie[:, 2] = array_ns.where(
        e2c[e2c2e[:, 2], 0] == e2c[:, 1],
        2,
        array_ns.where(e2c[e2c2e[:, 2], 1] == e2c[:, 1], 4, MISSING),
    )

    iie[:, 3] = array_ns.where(
        e2c[e2c2e[:, 3], 0] == e2c[:, 1],
        1,
        array_ns.where(e2c[e2c2e[:, 3], 1] == e2c[:, 1], 3, MISSING),
    )

    llb = horizontal_start_p4
    index = array_ns.arange(llb, num_edges)
    for i in range(c2e.shape[1]):
        # INVALID_INDEX
        if i <= MISSING:
            continue
        e_flx_avg[llb:, 0] = array_ns.where(
            owner_mask[llb:],
            array_ns.where(
                c2e[e2c[llb:, 0], i] == index,
                0.5
                * (
                    (
                        geofac_div[e2c[llb:, 0], i] * c_bln_avg[e2c[llb:, 0], 0]
                        + geofac_div[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], i]]
                        * c_bln_avg[e2c[llb:, 0], i + 1]
                        - e_flx_avg[e2c2e[llb:, 0], iie[llb:, 0]]
                        * geofac_div[e2c[llb:, 0], array_ns.mod(i + 1, 3)]
                        - e_flx_avg[e2c2e[llb:, 1], iie[llb:, 1]]
                        * geofac_div[e2c[llb:, 0], array_ns.mod(i + 2, 3)]
                    )
                    / geofac_div[e2c[llb:, 0], i]
                    + (
                        geofac_div[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], i]]
                        * c_bln_avg[e2c[llb:, 1], 0]
                        + geofac_div[e2c[llb:, 0], i]
                        * c_bln_avg[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], i] + 1]
                        - e_flx_avg[e2c2e[llb:, 2], iie[llb:, 2]]
                        * geofac_div[
                            e2c[llb:, 1], array_ns.mod(inv_neighbor_id[e2c[llb:, 0], i] + 1, 3)
                        ]
                        - e_flx_avg[e2c2e[llb:, 3], iie[llb:, 3]]
                        * geofac_div[
                            e2c[llb:, 1], array_ns.mod(inv_neighbor_id[e2c[llb:, 0], i] + 2, 3)
                        ]
                    )
                    / geofac_div[e2c[llb:, 1], inv_neighbor_id[e2c[llb:, 0], i]]
                ),
                e_flx_avg[llb:, 0],
            ),
            e_flx_avg[llb:, 0],
        )

    d2 = array_ns.sum(primal_cart_normal[:, None, :] * primal_cart_normal[e2c2e], axis=2)
    checksum = e_flx_avg[:, 0] + array_ns.sum(d2 * e_flx_avg[:, 1:], axis=1)

    e_flx_avg[llb:, :] = array_ns.where(
        owner_mask[llb:, None], e_flx_avg[llb:, :] / checksum[llb:, None], e_flx_avg[llb:, :]
    )

    exchange.exchange(dims.EdgeDim, e_flx_avg, stream=decomposition.BLOCK)

    return e_flx_avg


def compute_cells_aw_verts(
    *,
    dual_area: data_alloc.NDArray,
    edge_vert_length: data_alloc.NDArray,
    edge_cell_length: data_alloc.NDArray,
    v2e: data_alloc.NDArray,
    e2v: data_alloc.NDArray,
    v2c: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    horizontal_start: gtx.int32,
) -> data_alloc.NDArray:
    """
    Compute cells_aw_verts for interpolating fields from cells to vertices.

    In mathematical notation:
        let cells_aw_verts = x, x has dimensions (number of vertices, 6),
        x(i,j) = 0.5 sum_k l(j,k) d(i,k) / dual_area(i), where summation is over neighboring edges of vertex i,
        dual_area is the area of hexagon around the vertex, l(j,k) is the shortest distance between the center of cell j and edge k,
        d(i,k) is the distance between the vertex i and center of edge k.

    Args:
        dual_area: ndarray, representing a gtx.Field[gtx.Dims[VertexDim], ta.wpfloat]
        edge_vert_length: \\  ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2VDim], ta.wpfloat]
        edge_cell_length: //
        owner_mask: ndarray, representing a gtx.Field[gtx.Dims[VertexDim], bool]
        v2e: ndarray, representing a gtx.Field[gtx.Dims[VertexDim, V2EDim], gtx.int32]
        e2v: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2VDim], gtx.int32]
        v2c: ndarray, representing a gtx.Field[gtx.Dims[VertexDim, V2CDim], gtx.int32]
        e2c: ndarray, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        horizontal_start: int32, representing the start index of the horizontal dimension

    Returns:
        aw_verts: ndarray, representing a gtx.Field[gtx.Dims[VertexDim, 6], ta.wpfloat]
    """
    array_ns = data_alloc.array_namespace(dual_area)
    cells_aw_verts = array_ns.zeros(v2e.shape)
    num_verts = cells_aw_verts.shape[0]
    num_cells_per_vert = v2c.shape[1]
    num_edges_per_vert = v2e.shape[1]
    vertex_range = array_ns.arange(horizontal_start, num_verts)

    # Precompute valid v2c mask: skip MISSING and consecutive duplicates that may come from the pentagon points or the domain boundary.
    # Keep this outside the loop over edges to avoid unnecessarily recomputing: it only depends on v2c and not on the edge je.
    valid_v2c = array_ns.zeros(v2c.shape, dtype=bool)
    for jc in range(num_cells_per_vert):
        valid_v2c[:, jc] = (
            v2c[:, jc] != MISSING
            if jc == 0
            else (v2c[:, jc] != MISSING) & (v2c[:, jc] != v2c[:, jc - 1])
        )

    # loop over all edges, je, adjacent to vertices and compute the weighting coefficients of the cells adjacent to je
    for je in range(num_edges_per_vert):
        v2e_edge = v2e[horizontal_start:, je]

        # skip invalid or duplicate edges that may come from the pentagon points or the domain boundary
        valid_v2e = (
            v2e_edge != MISSING
            if je == 0
            else (v2e_edge != MISSING) & (v2e_edge != v2e[horizontal_start:, je - 1])
        )

        valid_v2e_edge = v2e_edge[valid_v2e]
        valid_vertices = vertex_range[valid_v2e]

        # determine which vertex of the edge je this is (0 or 1)
        idx_ve = array_ns.where(e2v[valid_v2e_edge, 0] == valid_vertices, 0, 1)

        # cells adjacent to each edge and the corresponding coefficients at those cells
        cell_0 = e2c[valid_v2e_edge, 0]
        cell_1 = e2c[valid_v2e_edge, 1]
        coefficient_at_cell_0 = (
            edge_vert_length[valid_v2e_edge, idx_ve] * edge_cell_length[valid_v2e_edge, 0]
        )
        coefficient_at_cell_1 = (
            edge_vert_length[valid_v2e_edge, idx_ve] * edge_cell_length[valid_v2e_edge, 1]
        )

        # loop over cells, jc, adjacent to the vertex adjacent to the edge je (v2e),
        # check if jc belongs to one of the neighboring cells of the edge and add the corresponding coeff l(j,k) * d(i,k)
        for jc in range(num_cells_per_vert):
            current_cell = v2c[valid_vertices, jc]
            valid_cell = valid_v2c[valid_vertices, jc]
            match0 = valid_cell & (cell_0 == current_cell)
            match1 = valid_cell & (cell_1 == current_cell)
            cells_aw_verts[valid_vertices[match0], jc] += coefficient_at_cell_0[match0]
            cells_aw_verts[valid_vertices[match1], jc] += coefficient_at_cell_1[match1]
    cells_aw_verts = 0.5 * cells_aw_verts / dual_area[:, array_ns.newaxis]
    return cells_aw_verts


def compute_e_bln_c_s(
    *,
    c2e: data_alloc.NDArray,
    cells_lat: data_alloc.NDArray,
    cells_lon: data_alloc.NDArray,
    edges_lat: data_alloc.NDArray,
    edges_lon: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """
    Compute e_bln_c_s.

    Args:
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[CellDim], bool]
        c2e: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], gtx.int32]
        cells_lat: \\ numpy array, representing a gtx.Field[gtx.Dims[CellDim], ta.wpfloat]
        cells_lon: //
        edges_lat: \\ numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        edges_lon: //

    Returns:
        e_bln_c_s: numpy array, representing a gtx.Field[gtx.Dims[CellDim, C2EDim], ta.wpfloat]
    """
    array_ns = data_alloc.array_namespace(c2e)
    llb = 0
    num_cells = c2e.shape[0]
    e_bln_c_s = array_ns.zeros([num_cells, c2e.shape[1]])
    yloc = cells_lat[llb:]
    xloc = cells_lon[llb:]
    ytemp = array_ns.zeros([c2e.shape[1], num_cells])
    xtemp = array_ns.zeros([c2e.shape[1], num_cells])

    for i in range(ytemp.shape[0]):
        ytemp[i] = edges_lat[c2e[llb:, i]]
        xtemp[i] = edges_lon[c2e[llb:, i]]

    # wgt_loc is hardcoded to 0.0 (actually completely missing) in the Fortran
    # e_bln_c_s code path (mo_intp_coeffs_lsq_bln.f90:2453-2462); contrast with
    # c_bln_avg which uses wgt_loc = divavg_cntrwgt (mo_intp_coeffs.f90:181).
    wgt = _weighting_factors(
        ytemp,
        xtemp,
        yloc,
        xloc,
        0.0,
    )

    e_bln_c_s[:, 0] = wgt[0]
    e_bln_c_s[:, 1] = wgt[1]
    e_bln_c_s[:, 2] = wgt[2]
    return e_bln_c_s


def compute_e_bln_c_s_torus(
    c2e: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """
    Compute e_bln_c_s.

    Args:
        c2e: connectivity from cell to its neighboring edges

    Returns:
        e_bln_c_s
    """
    array_ns = data_alloc.array_namespace(c2e)
    return array_ns.full_like(c2e, 1.0 / 3.0, dtype=ta.wpfloat)


def compute_pos_on_tplane_e_x_y(
    *,
    grid_sphere_radius: ta.wpfloat,
    primal_normal_v1: data_alloc.NDArray,
    primal_normal_v2: data_alloc.NDArray,
    dual_normal_v1: data_alloc.NDArray,
    dual_normal_v2: data_alloc.NDArray,
    cells_lon: data_alloc.NDArray,
    cells_lat: data_alloc.NDArray,
    edges_lon: data_alloc.NDArray,
    edges_lat: data_alloc.NDArray,
    owner_mask: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
    horizontal_start: gtx.int32,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """
    Compute pos_on_tplane_e_x_y.
    get geographical coordinates of edge midpoint
    get line and block indices of neighbour cells
    get geographical coordinates of first cell center
    projection first cell center into local \\lambda-\\Phi-system
    get geographical coordinates of second cell center
    projection second cell center into local \\lambda-\\Phi-system

    Args:
        grid_sphere_radius:
        primal_normal_v1: \\
        primal_normal_v2:  \\ numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        dual_normal_v1:    //
        dual_normal_v2:   //
        cells_lon: \\ numpy array, representing a gtx.Field[gtx.Dims[CellDim], ta.wpfloat]
        cells_lat: //
        edges_lon: \\ numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], ta.wpfloat]
        edges_lat: //
        owner_mask: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim], bool]
        e2c: numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], gtx.int32]
        horizontal_start:

    Returns:
        pos_on_tplane_e_x: \\ numpy array, representing a gtx.Field[gtx.Dims[EdgeDim, E2CDim], ta.wpfloat]
        pos_on_tplane_e_y: //
    """
    array_ns = data_alloc.array_namespace(primal_normal_v1)
    llb = horizontal_start
    pos_on_tplane_e_x = array_ns.zeros(e2c.shape)
    pos_on_tplane_e_y = array_ns.zeros(e2c.shape)
    xyloc_plane_n1 = array_ns.zeros([2, e2c.shape[0]])
    xyloc_plane_n2 = array_ns.zeros([2, e2c.shape[0]])
    xyloc_plane_n1[0, llb:], xyloc_plane_n1[1, llb:] = projection.gnomonic_proj(
        edges_lon[llb:], edges_lat[llb:], cells_lon[e2c[llb:, 0]], cells_lat[e2c[llb:, 0]]
    )
    xyloc_plane_n2[0, llb:], xyloc_plane_n2[1, llb:] = projection.gnomonic_proj(
        edges_lon[llb:], edges_lat[llb:], cells_lon[e2c[llb:, 1]], cells_lat[e2c[llb:, 1]]
    )

    pos_on_tplane_e_x[llb:, 0] = array_ns.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (
            xyloc_plane_n1[0, llb:] * primal_normal_v1[llb:]
            + xyloc_plane_n1[1, llb:] * primal_normal_v2[llb:]
        ),
        pos_on_tplane_e_x[llb:, 0],
    )
    pos_on_tplane_e_y[llb:, 0] = array_ns.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (
            xyloc_plane_n1[0, llb:] * dual_normal_v1[llb:]
            + xyloc_plane_n1[1, llb:] * dual_normal_v2[llb:]
        ),
        pos_on_tplane_e_y[llb:, 0],
    )
    pos_on_tplane_e_x[llb:, 1] = array_ns.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (
            xyloc_plane_n2[0, llb:] * primal_normal_v1[llb:]
            + xyloc_plane_n2[1, llb:] * primal_normal_v2[llb:]
        ),
        pos_on_tplane_e_x[llb:, 1],
    )
    pos_on_tplane_e_y[llb:, 1] = array_ns.where(
        owner_mask[llb:],
        grid_sphere_radius
        * (
            xyloc_plane_n2[0, llb:] * dual_normal_v1[llb:]
            + xyloc_plane_n2[1, llb:] * dual_normal_v2[llb:]
        ),
        pos_on_tplane_e_y[llb:, 1],
    )

    return pos_on_tplane_e_x, pos_on_tplane_e_y


def compute_pos_on_tplane_e_x_y_torus(
    dual_edge_length: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """
    Compute pos_on_tplane_e_x_y.
    get geographical coordinates of edge midpoint
    get line and block indices of neighbour cells
    get geographical coordinates of first cell center
    projection first cell center into local \\lambda-\\Phi-system
    get geographical coordinates of second cell center
    projection second cell center into local \\lambda-\\Phi-system

    Args:
        dual_edge_length
        e2c

    Returns:
        pos_on_tplane_e_x
        pos_on_tplane_e_y
    """
    array_ns = data_alloc.array_namespace(dual_edge_length)
    # The implementation makes the simplifying assumptions that:
    # - The torus grid consists of equilateral triangles, which means that the
    #   neighboring cell centers must always be at 0.5 * dual_edge_length from
    #   the edge center (the edge lies symmetrically perpendicular to the primal
    #   edge).
    # - The neighboring cell centers are exactly along the primal normal/dual
    #   tangent direction, which means the x component in the local coordinate
    #   system is always zero, and the y component is always 0.5 *
    #   dual_edge_length.
    # - The first neighbor cell is in the opposite direction of the primal
    #   normal and the second neighbor is in the direction of the primal normal.
    half_dual_edge_length = 0.5 * dual_edge_length[0]
    num_edges = e2c.shape[0]

    pos_on_tplane_e_x = array_ns.empty((num_edges, 2), dtype=dual_edge_length.dtype)
    pos_on_tplane_e_x[:, 0] = -half_dual_edge_length
    pos_on_tplane_e_x[:, 1] = half_dual_edge_length

    pos_on_tplane_e_y = array_ns.zeros((num_edges, 2), dtype=dual_edge_length.dtype)

    return pos_on_tplane_e_x, pos_on_tplane_e_y


def compute_lsq_pseudoinv(
    *,
    cell_owner_mask: data_alloc.NDArray,
    z_lsq_mat_c: data_alloc.NDArray,
    lsq_weights_c: data_alloc.NDArray,
    start_idx: int,
    min_rlcell_int: int,
    lsq_dim_unk: int,
    lsq_dim_c: int,
) -> data_alloc.NDArray:
    """
    Compute least-squares pseudoinverse.

    let lsq_dim_c = c, lsq_dim_unk = k, where c is the number of neighboring cells for the least squares fit,
    and k is the number of unknowns we want to solve for (e.g. k=2 for a linear fit in 2D: f(x) = a + b*x)

    z_lsq_mat_c is a non-square matrix, so we need to compute pseudo inverse for the inverse matrix of z_lsq_mat_c.
    let lsq_pseudoinv be the inverse matrix of z_lsq_mat_c.
    z_lsq_mat_c has dimensions (c, k), lsq_pseudoinv has (k, c), and lsq_weights_c has (c)
    singular value decomposition of the matrix z_lsq_mat_c gives u_matrix (c, k), s_matrix (k), v_t_matrix (k, k)
    lsq_pseudoinv = v_t_matrix^T u_matrix^T / s * lsq_weights_c
    """
    array_ns = data_alloc.array_namespace(cell_owner_mask)
    cell_size = cell_owner_mask.shape[0]
    cell_sequence = array_ns.arange(cell_size)
    valid_cell_mask = (
        cell_owner_mask & (cell_sequence >= start_idx) & (cell_sequence < min_rlcell_int)
    )
    lsq_pseudoinv = array_ns.zeros((cell_size, lsq_dim_unk, lsq_dim_c), dtype=ta.wpfloat)
    u_matrix, s_matrix, v_t_matrix = array_ns.linalg.svd(z_lsq_mat_c[valid_cell_mask, :, :])
    v_t_over_s = (
        v_t_matrix[:, :lsq_dim_unk, :lsq_dim_unk] / s_matrix[:, :lsq_dim_unk, array_ns.newaxis]
    )  # (n_valid_cells, lsq_dim_unk, lsq_dim_unk)
    pinv = array_ns.matmul(
        array_ns.transpose(v_t_over_s, (0, 2, 1)),
        array_ns.transpose(u_matrix, (0, 2, 1))[:, :lsq_dim_unk, :lsq_dim_c],
    )
    pinv *= lsq_weights_c[valid_cell_mask, array_ns.newaxis, :lsq_dim_c]
    lsq_pseudoinv[valid_cell_mask, :, :] = pinv

    return lsq_pseudoinv


def compute_lsq_weights_c(
    z_dist_g: data_alloc.NDArray,
    lsq_dim_stencil: int,
    lsq_wgt_exp: int,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(z_dist_g)
    z_norm = array_ns.sqrt(array_ns.sum(z_dist_g[:, :lsq_dim_stencil, :] ** 2, axis=2))
    lsq_weights_c = 1.0 / (z_norm**lsq_wgt_exp)
    lsq_weights_c = lsq_weights_c / array_ns.max(lsq_weights_c, axis=1)[:, array_ns.newaxis]
    return lsq_weights_c


def compute_z_lsq_mat_c(
    *,
    cell_owner_mask: data_alloc.NDArray,
    lsq_weights_c: data_alloc.NDArray,
    z_dist_g: data_alloc.NDArray,
    start_idx: int,
    min_rlcell_int: int,
    lsq_dim_unk: int,
    lsq_dim_c: int,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(cell_owner_mask)
    cell_size = cell_owner_mask.shape[0]
    cell_sequence = array_ns.arange(cell_size)
    min_lsq_bound = min(lsq_dim_unk, lsq_dim_c)
    z_lsq_mat_c = array_ns.zeros((cell_size, lsq_dim_c, lsq_dim_c), dtype=ta.wpfloat)

    valid_cell_mask = (
        cell_owner_mask & (cell_sequence >= start_idx) & (cell_sequence < min_rlcell_int)
    )
    z_lsq_mat_c[valid_cell_mask, :min_lsq_bound, :min_lsq_bound] = 1.0
    valid_cell_mask_with_halo = (cell_sequence >= start_idx) & (cell_sequence < min_rlcell_int)

    z_lsq_mat_c[valid_cell_mask_with_halo, :lsq_dim_c, :lsq_dim_unk] = (
        lsq_weights_c[valid_cell_mask_with_halo, :lsq_dim_c, array_ns.newaxis]
        * z_dist_g[valid_cell_mask_with_halo, :lsq_dim_c, :]
    )
    return z_lsq_mat_c


def compute_lsq_coeffs(
    *,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    cell_lat: data_alloc.NDArray,
    cell_lon: data_alloc.NDArray,
    c2e2c: data_alloc.NDArray,
    cell_owner_mask: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
    grid_sphere_radius: float,
    lsq_dim_unk: int,
    lsq_dim_c: int,
    lsq_wgt_exp: int,
    lsq_dim_stencil: int,
    start_idx: int,
    min_rlcell_int: int,
    geometry_type: int,
    exchange: decomposition.ExchangeRuntime,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(cell_center_x)
    z_dist_g = array_ns.zeros((cell_owner_mask.shape[0], lsq_dim_c, 2))
    match icon_grid.GeometryType(geometry_type):
        case icon_grid.GeometryType.ICOSAHEDRON:
            for js in range(lsq_dim_stencil):
                z_dist_g[:, js, :] = array_ns.asarray(
                    projection.gnomonic_proj(
                        cell_lon,
                        cell_lat,
                        cell_lon[c2e2c[:, js]],
                        cell_lat[c2e2c[:, js]],
                    )
                ).T

            z_dist_g *= grid_sphere_radius

        case icon_grid.GeometryType.TORUS:
            for jc in range(start_idx, min_rlcell_int):
                ilc_s = c2e2c[jc, :lsq_dim_stencil]
                cc_cell = array_ns.zeros((lsq_dim_stencil, 2))
                cc_cv = array_ns.asarray((cell_center_x[jc], cell_center_y[jc]))
                for js in range(lsq_dim_stencil):
                    cc_cell[js, :] = array_ns.asarray(
                        projection.diff_on_edges_torus_numpy(
                            cc_cv_x=cell_center_x[jc],
                            cc_cv_y=cell_center_y[jc],
                            cc_cell_x=cell_center_x[ilc_s][js],
                            cc_cell_y=cell_center_y[ilc_s][js],
                            domain_length=domain_length,
                            domain_height=domain_height,
                        )
                    )
                z_dist_g[jc, :, :] = cc_cell - cc_cv

    lsq_weights_c = compute_lsq_weights_c(z_dist_g, lsq_dim_stencil, lsq_wgt_exp)

    z_lsq_mat_c = compute_z_lsq_mat_c(
        cell_owner_mask=cell_owner_mask,
        lsq_weights_c=lsq_weights_c,
        z_dist_g=z_dist_g,
        start_idx=start_idx,
        min_rlcell_int=min_rlcell_int,
        lsq_dim_unk=lsq_dim_unk,
        lsq_dim_c=lsq_dim_c,
    )

    exchange.exchange(dims.CellDim, lsq_weights_c, stream=decomposition.BLOCK)

    lsq_pseudoinv = compute_lsq_pseudoinv(
        cell_owner_mask=cell_owner_mask,
        z_lsq_mat_c=z_lsq_mat_c,
        lsq_weights_c=lsq_weights_c,
        start_idx=start_idx,
        min_rlcell_int=min_rlcell_int,
        lsq_dim_unk=lsq_dim_unk,
        lsq_dim_c=lsq_dim_c,
    )
    exchange.exchange(dims.CellDim, lsq_pseudoinv[:, 0, :], stream=decomposition.BLOCK)
    exchange.exchange(dims.CellDim, lsq_pseudoinv[:, 1, :], stream=decomposition.BLOCK)

    return lsq_pseudoinv
