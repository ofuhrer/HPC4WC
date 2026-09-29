# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next as gtx

from icon4py.model.common.utils import data_allocation as data_alloc


def compute_max_nbhgt_array_ns(
    c2e2c: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    nlev: int,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(c2e2c)
    z_mc_nlev = z_mc[:, nlev - 1]
    max_nbhgt_0_1 = array_ns.maximum(z_mc_nlev[c2e2c[:, 0]], z_mc_nlev[c2e2c[:, 1]])
    max_nbhgt = array_ns.maximum(max_nbhgt_0_1, z_mc_nlev[c2e2c[:, 2]])
    return max_nbhgt


def _compute_k_start_end(
    *,
    z_mc: data_alloc.NDArray,
    max_nbhgt: data_alloc.NDArray,
    maxslp_avg: data_alloc.NDArray,
    maxhgtd_avg: data_alloc.NDArray,
    c_owner_mask: data_alloc.NDArray,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    nlev: int,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(z_mc)
    condition1 = array_ns.logical_or(maxslp_avg >= thslp_zdiffu, maxhgtd_avg >= thhgtd_zdiffu)
    cell_mask = array_ns.tile(
        array_ns.where(condition1[:, nlev - 1], c_owner_mask, False), (nlev, 1)
    ).T
    threshold = array_ns.tile(max_nbhgt, (nlev, 1)).T
    owned_cell_above_threshold = array_ns.logical_and(cell_mask, z_mc >= threshold)
    last_true_indices = nlev - 1 - array_ns.argmax(owned_cell_above_threshold[:, ::-1], axis=1)
    kend = array_ns.where(
        array_ns.any(owned_cell_above_threshold, axis=1), last_true_indices + 1, 0
    )
    kstart = array_ns.argmax(condition1, axis=1)
    # reset the values where start > end to be an empty range(start, end)
    kstart = array_ns.where(kstart > kend, nlev, kstart)
    cell_index_mask = array_ns.where(kend > kstart, True, False)

    return kstart, kend, cell_index_mask


def compute_diffusion_mask_and_coef(
    *,
    c2e2c: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    max_nbhgt: data_alloc.NDArray,
    c_owner_mask: data_alloc.NDArray,
    maxslp_avg: data_alloc.NDArray,
    maxhgtd_avg: data_alloc.NDArray,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    cell_nudging: int,
    nlev: int,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """
    Compute the diffusion coefficient.

    let x = diffusion coefficient
    x = max(
        sqrt( max( 0, maxslp_avg - thslp_zdiffu ) ) / 250,
        2e-4 * sqrt( max( 0, maxhgtd_avg - thhgtd_zdiffu ) )
    )
    """
    array_ns = data_alloc.array_namespace(c2e2c)
    n_cells = c2e2c.shape[0]
    zd_diffcoef = array_ns.zeros(shape=(n_cells, nlev))
    k_start, k_end, cell_index_mask = _compute_k_start_end(
        z_mc=z_mc,
        max_nbhgt=max_nbhgt,
        maxslp_avg=maxslp_avg,
        maxhgtd_avg=maxhgtd_avg,
        c_owner_mask=c_owner_mask,
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        nlev=nlev,
    )
    valid_cell_mask = cell_index_mask & (array_ns.arange(n_cells) >= cell_nudging)
    default_level_idx = array_ns.arange(nlev)[array_ns.newaxis, :]
    krange_mask = (default_level_idx >= k_start[valid_cell_mask][:, array_ns.newaxis]) & (
        default_level_idx < k_end[valid_cell_mask][:, array_ns.newaxis]
    )  # (n_valid_cells, nlev)

    zd_diffcoef_var = array_ns.maximum(
        array_ns.sqrt(array_ns.maximum(0.0, maxslp_avg[valid_cell_mask, :] - thslp_zdiffu)) / 250.0,
        2.0e-4
        * array_ns.sqrt(array_ns.maximum(0.0, maxhgtd_avg[valid_cell_mask, :] - thhgtd_zdiffu)),
    )
    zd_diffcoef[valid_cell_mask, :] = array_ns.where(
        krange_mask, array_ns.minimum(0.002, zd_diffcoef_var), 0.0
    )

    return zd_diffcoef


def compute_diffusion_intcoef_and_vertoffset(
    *,
    c2e2c: data_alloc.NDArray,
    z_mc: data_alloc.NDArray,
    max_nbhgt: data_alloc.NDArray,
    c_owner_mask: data_alloc.NDArray,
    maxslp_avg: data_alloc.NDArray,
    maxhgtd_avg: data_alloc.NDArray,
    thslp_zdiffu: float,
    thhgtd_zdiffu: float,
    cell_nudging: int,
    nlev: int,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(c2e2c)
    n_cells = c2e2c.shape[0]
    n_c2e2c = c2e2c.shape[1]
    z_mc_off = z_mc[c2e2c]
    zd_vertoffset = array_ns.zeros(shape=(n_cells, n_c2e2c, nlev), dtype=gtx.int32)
    zd_intcoef = array_ns.zeros(shape=(n_cells, n_c2e2c, nlev))

    k_start, k_end, cell_index_mask = _compute_k_start_end(
        z_mc=z_mc,
        max_nbhgt=max_nbhgt,
        maxslp_avg=maxslp_avg,
        maxhgtd_avg=maxhgtd_avg,
        c_owner_mask=c_owner_mask,
        thslp_zdiffu=thslp_zdiffu,
        thhgtd_zdiffu=thhgtd_zdiffu,
        nlev=nlev,
    )

    # Identify valid cells (those with non-empty k_range above nudging boundary)
    valid_cell_mask = array_ns.where(cell_index_mask & (array_ns.arange(n_cells) >= cell_nudging))[
        0
    ]
    n_valid_cells = len(valid_cell_mask)

    # Compute max vertical z-range for all cells, this is used later to construct a one-to-one mapping when searching for neighboring cell's level index that lies within the vertical range of the current cell
    max_vertical_zrange = array_ns.maximum(z_mc.max() - z_mc.min(), 0)  # (n_valid_cells)

    if n_valid_cells > 0:
        masked_z_mc = z_mc[valid_cell_mask, :]  # (n_valid_cells, nlev)
        # Level mask: True where level is in [kstart, kend) for each valid cell
        default_level_idx = array_ns.arange(nlev)[array_ns.newaxis, :]
        krange_mask = (default_level_idx >= k_start[valid_cell_mask][:, array_ns.newaxis]) & (
            default_level_idx < k_end[valid_cell_mask][:, array_ns.newaxis]
        )  # (n_valid_cells, nlev)

        # loop over neighboring cells, c2e2c_idx, of cells to obtain the k level index of the neighboring cell at the same altitude and the corresponding vertical interpolation coefficient
        for c2e2c_idx in range(n_c2e2c):
            masked_z_mc_off = z_mc_off[valid_cell_mask, c2e2c_idx, :]

            # prepare the cell offsets to construct the one-to-one mapping for searching the neighboring cell's level index that lies within the vertical range of the current cell
            cell_offsets = (
                array_ns.arange(n_valid_cells, dtype=array_ns.float64) * max_vertical_zrange
            )
            # create a 1-D array for searching the neighboring cell's level index that lies within the vertical range of the current cell, a negative sign is added to make the array strictly ascending
            flattened_neighbor_z_mc = array_ns.ravel(
                -masked_z_mc_off + cell_offsets[:, array_ns.newaxis]
            )  # (n_valid_cells * nlev)
            flattened_current_z_mc = array_ns.ravel(
                -masked_z_mc + cell_offsets[:, array_ns.newaxis]
            )  # (n_valid_cells * nlev)
            # search for the neighboring cell's level index that lies within the vertical range of the current cell using the flattened arrays, this gives us a flat index in the flattened_neighbor_z_mc array
            flat_neighbor_k_idx = array_ns.searchsorted(
                flattened_neighbor_z_mc, flattened_current_z_mc, side="right"
            )
            # unravel the flat_neighbor_k_idx back to (n_valid_cells, nlev) and adjust the indices to get the correct neighboring cell's level index
            neighbor_k_idx = (
                flat_neighbor_k_idx.reshape(n_valid_cells, nlev)
                - array_ns.arange(n_valid_cells)[:, array_ns.newaxis] * nlev
                - 1
            )  # -1 is added because of searchsorted with side=right starts with 1 when the value is >= minimum (n_valid_cells, nlev)
            vertoffset = array_ns.clip(neighbor_k_idx, 0, nlev - 2)  # (n_valid_cells, nlev)

            # Vertical interpolation coefficients
            neighboring_cell_upper_height = array_ns.take_along_axis(
                masked_z_mc_off, vertoffset, axis=1
            )
            neighboring_cell_lower_height = array_ns.take_along_axis(
                masked_z_mc_off, vertoffset + 1, axis=1
            )
            denom = (
                neighboring_cell_upper_height - neighboring_cell_lower_height
            )  # (n_valid_cells, nlev)
            intcoeff = (
                masked_z_mc - neighboring_cell_lower_height
            ) / denom  # (n_valid_cells, nlev)

            zd_intcoef[valid_cell_mask, c2e2c_idx, :] = array_ns.where(krange_mask, intcoeff, 0.0)
            zd_vertoffset[valid_cell_mask, c2e2c_idx, :] = array_ns.where(
                krange_mask, (vertoffset - default_level_idx).astype(gtx.int32), gtx.int32(0)
            )

    return zd_intcoef, zd_vertoffset
