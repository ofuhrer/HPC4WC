# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import NamedTuple

from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.states.data import COMMON_TRACER_CF_ATTRIBUTES


class TracerField(NamedTuple):
    name: str
    field: fa.CellKField[ta.wpfloat]


#: Tracer names sorted by their Fortran index (QV=0, QC=1, QI=2, QR=3, QS=4, QG=5).
#: Every tracer must define ``icon_var_list_index``; a missing key raises ``KeyError``.
_TRACER_FIELDS: tuple[str, ...] = tuple(
    sorted(
        COMMON_TRACER_CF_ATTRIBUTES,
        key=lambda k: COMMON_TRACER_CF_ATTRIBUTES[k]["icon_var_list_index"],  # type: ignore[index]
    )
)


@dataclasses.dataclass(frozen=True)
class TracerConfig:
    """Which tracers are active in the model configuration.

    Each boolean field indicates whether the corresponding tracer is active.
    Used instead of a raw ``ntracer: int`` to provide type-safe tracer selection.
    """

    qv: bool = False
    qc: bool = False
    qi: bool = False
    qr: bool = False
    qs: bool = False
    qg: bool = False

    @classmethod
    def all(cls) -> TracerConfig:
        return cls(qv=True, qc=True, qi=True, qr=True, qs=True, qg=True)

    @classmethod
    def none(cls) -> TracerConfig:
        return cls()

    @classmethod
    def from_ntracer(cls, ntracer: int) -> TracerConfig:
        """Build a ``TracerConfig`` from a Fortran ``ntracer`` count.

        Fortran ICON uses a fixed tracer ordering: QV=0, QC=1, QI=2, QR=3, QS=4, QG=5.
        The first *ntracer* entries in this order are considered active.

        Raises:
            ValueError: if ntracer is outside the valid range.
        """
        n = len(_TRACER_FIELDS)
        if not 0 <= ntracer <= n:
            raise ValueError(f"ntracer must be between 0 and {n}, got {ntracer}")
        return cls(**{name: i < ntracer for i, name in enumerate(_TRACER_FIELDS)})

    @property
    def nactive(self) -> int:
        return sum(dataclasses.asdict(self).values())

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(name for name in _TRACER_FIELDS if getattr(self, name))

    def __iter__(self) -> Iterator[str]:
        return iter(self.active_names)

    def __len__(self) -> int:
        return self.nactive

    def __contains__(self, name: str) -> bool:
        return name in _TRACER_FIELDS and getattr(self, name)

    def __bool__(self) -> bool:
        return self.nactive > 0

    def __str__(self) -> str:
        names = ", ".join(self.active_names)
        return names if names else "none"


@dataclasses.dataclass
class TracerState:
    """
    Class that contains the tracer state which includes hydrometeors and aerosols.
    Corresponds to tracer pointers in ICON t_nh_prog
    """

    #: specific humidity [kg/kg] at cell center
    qv: fa.CellKField[ta.wpfloat] | None = None
    #: specific cloud water content [kg/kg] at cell center
    qc: fa.CellKField[ta.wpfloat] | None = None
    #: specific cloud ice content [kg/kg] at cell center
    qi: fa.CellKField[ta.wpfloat] | None = None
    #: specific rain content [kg/kg] at cell center
    qr: fa.CellKField[ta.wpfloat] | None = None
    #: specific snow content [kg/kg] at cell center
    qs: fa.CellKField[ta.wpfloat] | None = None
    #: specific graupel content [kg/kg] at cell center
    qg: fa.CellKField[ta.wpfloat] | None = None

    def active_fields(self) -> Iterator[TracerField]:
        """Yield a ``TracerField`` for each non-``None`` tracer field."""
        for name in _TRACER_FIELDS:
            field = getattr(self, name)
            if field is not None:
                yield TracerField(name=name, field=field)
