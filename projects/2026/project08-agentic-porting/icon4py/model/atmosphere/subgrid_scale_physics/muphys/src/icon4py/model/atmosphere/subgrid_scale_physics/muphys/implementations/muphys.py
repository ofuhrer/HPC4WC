# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.saturation_adjustment import (
    _saturation_adjustment,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.implementations.graupel import graupel
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _muphys(  # noqa: PLR0917 [too-many-positional-arguments]
    last_level: gtx.int32,
    dz: fa.CellKField[ta.wpfloat],
    te: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Pressure
    rho: fa.CellKField[ta.wpfloat],  # Density containing dry air and water constituents
    q_in: Q,
    dt: ta.wpfloat,
    qnc: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    Q,
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    te, qve, qce = _saturation_adjustment(
        te=te,
        q_in=q_in,
        rho=rho,
    )

    t, q, pflx, pr, ps, pi, pg, pre = graupel(
        last_level=last_level,
        dz=dz,
        te=te,
        p=p,
        rho=rho,
        q=Q(v=qve, c=qce, r=q_in.r, s=q_in.s, i=q_in.i, g=q_in.g),
        dt=dt,
        qnc=qnc,
        enable_masking=True,  # TODO(havogt): expose this option when optimizing full muphys
    )

    te, qve, qce = _saturation_adjustment(
        te=t,
        q_in=q,
        rho=rho,
    )

    return t, Q(v=qve, c=qce, r=q.r, s=q.s, i=q.i, g=q.g), pflx, pr, ps, pi, pg, pre


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def muphys_run(  # noqa: PLR0917 [too-many-positional-arguments]
    dz: fa.CellKField[ta.wpfloat],
    te: fa.CellKField[ta.wpfloat],  # Temperature
    p: fa.CellKField[ta.wpfloat],  # Pressure
    rho: fa.CellKField[ta.wpfloat],  # Density containing dry air and water constituents
    q_in: Q,
    dt: ta.wpfloat,  # Time step
    qnc: ta.wpfloat,
    q_out: Q,
    t_out: fa.CellKField[ta.wpfloat],  # Revised temperature
    pflx: fa.CellKField[ta.wpfloat],  # Total precipitation flux
    pr: fa.CellKField[ta.wpfloat],  # Precipitation of rain
    ps: fa.CellKField[ta.wpfloat],  # Precipitation of snow
    pi: fa.CellKField[ta.wpfloat],  # Precipitation of ice
    pg: fa.CellKField[ta.wpfloat],  # Precipitation of graupel
    pre: fa.CellKField[ta.wpfloat],  # Precipitation of graupel
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _muphys(
        last_level=vertical_end - 1,
        dz=dz,
        te=te,
        p=p,
        rho=rho,
        q_in=q_in,
        dt=dt,
        qnc=qnc,
        out=(t_out, q_out, pflx, pr, ps, pi, pg, pre),
        domain=(
            # t_out
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            # q_out
            (
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
                {
                    dims.CellDim: (horizontal_start, horizontal_end),
                    dims.KDim: (vertical_start, vertical_end),
                },
            ),
            # pflx
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
            # pr
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_end - 1, vertical_end),
            },
            # ps
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_end - 1, vertical_end),
            },
            # pi
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_end - 1, vertical_end),
            },
            # pg
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_end - 1, vertical_end),
            },
            # pre
            {
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_end - 1, vertical_end),
            },
        ),
    )
