# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import broadcast, maximum, minimum

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import KDim


@gtx.field_operator
def _en_smag_fac_for_zero_nshift(  # noqa: PLR0917 [too-many-positional-arguments]
    vect_a: fa.KField[float],
    hdiff_smag_fac: float,
    hdiff_smag_fac2: float,
    hdiff_smag_fac3: float,
    hdiff_smag_fac4: float,
    hdiff_smag_z: float,
    hdiff_smag_z2: float,
    hdiff_smag_z3: float,
    hdiff_smag_z4: float,
) -> fa.KField[float]:
    dz21 = hdiff_smag_z2 - hdiff_smag_z
    alin = (hdiff_smag_fac2 - hdiff_smag_fac) / dz21
    df32 = hdiff_smag_fac3 - hdiff_smag_fac2
    df42 = hdiff_smag_fac4 - hdiff_smag_fac2
    dz32 = hdiff_smag_z3 - hdiff_smag_z2
    dz42 = hdiff_smag_z4 - hdiff_smag_z2

    bqdr = (df42 * dz32 - df32 * dz42) / (dz32 * dz42 * (dz42 - dz32))
    aqdr = df32 / dz32 - bqdr * dz32
    zf = 0.5 * (vect_a + vect_a(KDim + 1))
    zero = broadcast(0.0, (KDim,))

    dzlin = minimum(broadcast(dz21, (KDim,)), maximum(zero, zf - hdiff_smag_z))
    dzqdr = minimum(broadcast(dz42, (KDim,)), maximum(zero, zf - hdiff_smag_z2))
    enh_smag_fac = hdiff_smag_fac + (dzlin * alin) + dzqdr * (aqdr + dzqdr * bqdr)
    return enh_smag_fac


@gtx.program
def en_smag_fac_for_zero_nshift(  # noqa: PLR0917 [too-many-positional-arguments]
    vect_a: fa.KField[float],
    hdiff_smag_fac: float,
    hdiff_smag_fac2: float,
    hdiff_smag_fac3: float,
    hdiff_smag_fac4: float,
    hdiff_smag_z: float,
    hdiff_smag_z2: float,
    hdiff_smag_z3: float,
    hdiff_smag_z4: float,
    enh_smag_fac: fa.KField[float],
) -> None:
    _en_smag_fac_for_zero_nshift(
        vect_a=vect_a,
        hdiff_smag_fac=hdiff_smag_fac,
        hdiff_smag_fac2=hdiff_smag_fac2,
        hdiff_smag_fac3=hdiff_smag_fac3,
        hdiff_smag_fac4=hdiff_smag_fac4,
        hdiff_smag_z=hdiff_smag_z,
        hdiff_smag_z2=hdiff_smag_z2,
        hdiff_smag_z3=hdiff_smag_z3,
        hdiff_smag_z4=hdiff_smag_z4,
        out=enh_smag_fac,
    )
