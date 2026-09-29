# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims


def enhanced_smagorinski_factor_numpy(
    factor_in: tuple[float, ...], heigths_in: tuple[float, ...], a_vec: np.ndarray
) -> float:
    alin = (factor_in[1] - factor_in[0]) / (heigths_in[1] - heigths_in[0])
    df32 = factor_in[2] - factor_in[1]
    df42 = factor_in[3] - factor_in[1]
    dz32 = heigths_in[2] - heigths_in[1]
    dz42 = heigths_in[3] - heigths_in[1]
    bqdr = (df42 * dz32 - df32 * dz42) / (dz32 * dz42 * (dz42 - dz32))
    aqdr = df32 / dz32 - bqdr * dz32
    zf = 0.5 * (a_vec[:-1] + a_vec[1:])
    max0 = np.maximum(0.0, zf - heigths_in[0])
    dzlin = np.minimum(heigths_in[1] - heigths_in[0], max0)
    max1 = np.maximum(0.0, zf - heigths_in[1])
    dzqdr = np.minimum(heigths_in[3] - heigths_in[1], max1)
    return factor_in[0] + dzlin * alin + dzqdr * (aqdr + dzqdr * bqdr)


def nabla2_on_cell_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray], psi_c: np.ndarray, geofac_n2s: np.ndarray
) -> np.ndarray:
    c2e2cO = connectivities[dims.C2E2CODim]
    nabla2_psi_c = np.sum(np.where((c2e2cO != -1), psi_c[c2e2cO] * geofac_n2s, 0), axis=1)
    return nabla2_psi_c


def nabla2_on_cell_k_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray], psi_c: np.ndarray, geofac_n2s: np.ndarray
) -> np.ndarray:
    c2e2cO = connectivities[dims.C2E2CODim]
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
    nabla2_psi_c = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], psi_c[c2e2cO] * geofac_n2s, 0), axis=1
    )
    return nabla2_psi_c
