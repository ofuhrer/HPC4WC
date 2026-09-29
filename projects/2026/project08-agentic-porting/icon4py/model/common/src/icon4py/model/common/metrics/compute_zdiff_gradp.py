# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import data_allocation as data_alloc


def compute_zdiff_gradp(  # noqa: PLR0912 [too-many-branches]
    *,
    e2c,
    z_mc: data_alloc.NDArray,
    c_lin_e: data_alloc.NDArray,
    z_ifc: data_alloc.NDArray,
    flat_idx: data_alloc.NDArray,
    topography: data_alloc.NDArray,
    nlev: int,
    horizontal_start: gtx.int32,
    horizontal_start_1: gtx.int32,
    exchange: decomposition.ExchangeRuntime,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(z_mc)
    nedges = e2c.shape[0]
    z_me = array_ns.sum(z_mc[e2c] * array_ns.expand_dims(c_lin_e, axis=-1), axis=1)

    exchange.exchange(dims.EdgeDim, z_me, stream=decomposition.BLOCK)

    z_aux1 = array_ns.maximum(topography[e2c[:, 0]], topography[e2c[:, 1]])
    z_aux2 = z_aux1 - 5.0  # extrapol_dist
    zdiff_gradp = array_ns.zeros_like(z_mc[e2c])
    jk_field = array_ns.arange(nlev, dtype=gtx.int32)
    zdiff_gradp[horizontal_start:, :, :] = (
        array_ns.expand_dims(z_me, axis=1)[horizontal_start:, :, :]
        - z_mc[e2c][horizontal_start:, :, :]
    )
    vertidx_gradp = array_ns.expand_dims(
        array_ns.expand_dims(jk_field, axis=0).repeat(2, axis=0), axis=0
    ).repeat(nedges, axis=0)
    vertoffset_gradp = array_ns.expand_dims(
        array_ns.expand_dims(jk_field, axis=0).repeat(2, axis=0), axis=0
    ).repeat(nedges, axis=0)
    """
    First part for loop implementation with gt4py code

    >>> z_ifc_off_koff = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    >>> z_ifc_off = as_field((EdgeDim, KDim,), z_ifc[e2c[:, 0], :])
    >>> _compute_z_ifc_off_koff(
    >>>     z_ifc_off=z_ifc_off,
    >>>     domain={EdgeDim: (horizontal_start, nedges), KDim: (0, nlev)},
    >>>     out=z_ifc_off_koff,
    >>>     offset_provider={"Koff": icon_grid.get_offset_provider("Koff")}
    >>> )
    """

    for je in range(horizontal_start, nedges):
        for jk in range(int(flat_idx[je]) + 1, nlev):
            """
            Second part for loop implementation with gt4py code
            >>> param_2 = as_field((KDim,), array_ns.asarray([False] * nlev))
            >>> param_3 = as_field((KDim,), array_ns.arange(nlev))
            >>> z_ifc_off_e = as_field((KDim,), z_ifc[e2c[je, 0], :])
            >>> _compute_param.with_backend(backend)(
            >>>     z_me_jk=z_me[je, jk],
            >>>     z_ifc_off=z_ifc_off_e,
            >>>     z_ifc_off_koff=as_field((KDim,), z_ifc_off_koff.ndarray[je, :]),
            >>>     lower=int(flat_idx[je]),
            >>>     nlev=nlev - 1,
            >>>     out=(param_3, param_2),
            >>>     offset_provider={}
            >>> )
            >>> zdiff_gradp[je, 0, jk] = z_me[je, jk] - z_mc[e2c[je, 0], array_ns.where(param_2.ndarray)[0][0]]
            """

            param = array_ns.zeros((nlev,), dtype=bool)
            for jk1 in range(int(flat_idx[je]), nlev):
                if jk1 == nlev - 1 or (
                    z_me[je, jk] <= z_ifc[e2c[je, 0], jk1]
                    and z_me[je, jk] >= z_ifc[e2c[je, 0], jk1 + 1]
                ):
                    param[jk1] = True
            vertidx_gradp[je, 0, jk] = array_ns.where(param)[0][0]
            zdiff_gradp[je, 0, jk] = z_me[je, jk] - z_mc[e2c[je, 0], array_ns.where(param)[0][0]]

        jk_start = int(flat_idx[je])
        for jk in range(int(flat_idx[je]) + 1, nlev):
            for jk1 in range(jk_start, nlev):
                if jk1 == nlev - 1 or (
                    z_me[je, jk] <= z_ifc[e2c[je, 1], jk1]
                    and z_me[je, jk] >= z_ifc[e2c[je, 1], jk1 + 1]
                ):
                    vertidx_gradp[je, 1, jk] = jk1
                    zdiff_gradp[je, 1, jk] = z_me[je, jk] - z_mc[e2c[je, 1], jk1]
                    jk_start = jk1
                    break

    for je in range(horizontal_start_1, nedges):
        jk_start = int(flat_idx[je])
        for jk in range(int(flat_idx[je]) + 1, nlev):
            if z_me[je, jk] < z_aux2[je]:
                for jk1 in range(jk_start, nlev):
                    if jk1 == nlev - 1 or (
                        z_aux2[je] <= z_ifc[e2c[je, 0], jk1]
                        and z_aux2[je] >= z_ifc[e2c[je, 0], jk1 + 1]
                    ):
                        vertidx_gradp[je, 0, jk] = jk1
                        zdiff_gradp[je, 0, jk] = z_aux2[je] - z_mc[e2c[je, 0], jk1]
                        jk_start = jk1
                        break

        jk_start = int(flat_idx[je])
        for jk in range(int(flat_idx[je]) + 1, nlev):
            if z_me[je, jk] < z_aux2[je]:
                for jk1 in range(jk_start, nlev):
                    if jk1 == nlev - 1 or (
                        z_aux2[je] <= z_ifc[e2c[je, 1], jk1]
                        and z_aux2[je] >= z_ifc[e2c[je, 1], jk1 + 1]
                    ):
                        vertidx_gradp[je, 1, jk] = jk1
                        zdiff_gradp[je, 1, jk] = z_aux2[je] - z_mc[e2c[je, 1], jk1]
                        jk_start = jk1
                        break

    vertoffset_gradp = vertidx_gradp - vertoffset_gradp

    exchange.exchange(dims.EdgeDim, zdiff_gradp[:, 0, :], stream=decomposition.BLOCK)
    exchange.exchange(dims.EdgeDim, zdiff_gradp[:, 1, :], stream=decomposition.BLOCK)

    return zdiff_gradp, vertoffset_gradp
