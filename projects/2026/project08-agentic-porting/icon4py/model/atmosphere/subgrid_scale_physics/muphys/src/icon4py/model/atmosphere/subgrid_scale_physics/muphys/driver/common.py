# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import ClassVar

import netCDF4
import numpy as np
from gt4py import next as gtx
from gt4py.next import typing as gtx_typing

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.common import dimension as dims


def _calc_dz(z: np.ndarray) -> np.ndarray:
    ksize = z.shape[0]
    dz = np.zeros(z.shape, np.float64)
    zh = 1.5 * z[ksize - 1, :] - 0.5 * z[ksize - 2, :]
    for k in range(ksize - 1, -1, -1):
        zh_new = 2.0 * z[k, :] - zh
        dz[k, :] = -zh + zh_new
        zh = zh_new
    return dz


def _as_field_from_nc(
    dataset: netCDF4.Dataset,
    allocator: gtx_typing.Allocator,
    varname: str,
    optional: bool = False,
    dtype: np.dtype | None = None,
) -> gtx.Field[dims.CellDim, dims.KDim] | None:
    if optional and varname not in dataset.variables:
        return None

    var = dataset.variables[varname]
    if var.dimensions[0] == "time":
        var = var[0, :, :]
    data = np.transpose(var)
    if dtype is not None:
        data = data.astype(dtype)
    return gtx.as_field(
        (dims.CellDim, dims.KDim),
        data,
        allocator=allocator,
    )


def _field_to_nc(
    dataset: netCDF4.Dataset,
    dims: tuple[str, str],
    varname: str,
    field: gtx.Field[dims.CellDim, dims.KDim],
    dtype: np.dtype = np.float64,
) -> None:
    var = dataset.createVariable(varname, dtype, dims)
    var[...] = field.asnumpy().transpose()


@dataclasses.dataclass
class GraupelInput:
    ncells: int
    nlev: int
    dz: gtx.Field[dims.CellDim, dims.KDim]
    p: gtx.Field[dims.CellDim, dims.KDim]
    rho: gtx.Field[dims.CellDim, dims.KDim]
    t: gtx.Field[dims.CellDim, dims.KDim]
    qv: gtx.Field[dims.CellDim, dims.KDim]
    qc: gtx.Field[dims.CellDim, dims.KDim]
    qi: gtx.Field[dims.CellDim, dims.KDim]
    qr: gtx.Field[dims.CellDim, dims.KDim]
    qs: gtx.Field[dims.CellDim, dims.KDim]
    qg: gtx.Field[dims.CellDim, dims.KDim]

    @property
    def q(self) -> Q:
        return Q(
            v=self.qv,
            c=self.qc,
            r=self.qr,
            s=self.qs,
            i=self.qi,
            g=self.qg,
        )

    @classmethod
    def load(
        cls,
        filename: pathlib.Path | str,
        allocator: gtx_typing.Allocator,
        dtype=np.float64,
    ) -> GraupelInput:
        with netCDF4.Dataset(filename, mode="r") as ncfile:
            try:
                ncells = len(ncfile.dimensions["cell"])
            except KeyError:
                ncells = len(ncfile.dimensions["ncells"])

            nlev = len(ncfile.dimensions["height"])

            dz = _calc_dz(np.asarray(ncfile.variables["zg"]).astype(dtype))

            field_from_nc = functools.partial(_as_field_from_nc, ncfile, allocator, dtype=dtype)
            return cls(
                ncells=ncells,
                nlev=nlev,
                dz=gtx.as_field(
                    (dims.CellDim, dims.KDim), np.transpose(dz), allocator=allocator, dtype=dtype
                ),
                t=field_from_nc("ta"),
                p=field_from_nc("pfull"),
                qs=field_from_nc("qs"),
                qi=field_from_nc("cli"),
                qg=field_from_nc("qg"),
                qv=field_from_nc("hus"),
                qc=field_from_nc("clw"),
                qr=field_from_nc("qr"),
                rho=field_from_nc("rho"),
            )


@dataclasses.dataclass
class GraupelOutput:
    t: gtx.Field[dims.CellDim, dims.KDim]
    qv: gtx.Field[dims.CellDim, dims.KDim]
    qc: gtx.Field[dims.CellDim, dims.KDim]
    qi: gtx.Field[dims.CellDim, dims.KDim]
    qr: gtx.Field[dims.CellDim, dims.KDim]
    qs: gtx.Field[dims.CellDim, dims.KDim]
    qg: gtx.Field[dims.CellDim, dims.KDim]

    pflx: gtx.Field[dims.CellDim, dims.KDim] | None
    pr: gtx.Field[dims.CellDim, dims.KDim] | None
    ps: gtx.Field[dims.CellDim, dims.KDim] | None
    pi: gtx.Field[dims.CellDim, dims.KDim] | None
    pg: gtx.Field[dims.CellDim, dims.KDim] | None
    pre: gtx.Field[dims.CellDim, dims.KDim] | None

    _surface_fields: ClassVar[list[str]] = ["pr", "ps", "pi", "pg", "pre"]

    @classmethod
    def allocate(
        cls,
        allocator: gtx_typing.Allocator,
        domain: gtx.Domain,
        dtype: gtx.float32 | gtx.float64 = gtx.float64,
        references: dict[str, gtx.Field] | None = None,
    ):
        """
        Returns a GraupelOutput with allocated fields.

        :param domain: Full domain of the Muphys fields.
        :param references: Dictionary of fields that should be re-used instead of allocated.
        """
        # TODO(havogt): maybe this function should become an __init__ with defaults
        if references is None:
            references = {}

        zeros_full = functools.partial(gtx.zeros, domain=domain, allocator=allocator, dtype=dtype)
        surface_domain = gtx.Domain(
            dims=domain.dims,
            ranges=(
                domain.ranges[0],
                gtx.unit_range((domain.ranges[1].stop - 1, domain.ranges[1].stop)),
            ),
        )
        zeros_surface = functools.partial(
            gtx.zeros, domain=surface_domain, allocator=allocator, dtype=dtype
        )
        return cls(
            **{
                field.name: references[field.name]
                if field.name in references
                else zeros_surface()
                if field.name in cls._surface_fields
                else zeros_full()
                for field in dataclasses.fields(cls)
            }
        )

    @classmethod
    def load(
        cls,
        filename: pathlib.Path | str,
        allocator: gtx_typing.Allocator,
        dtype: gtx.float32 | gtx.float64 = gtx.float64,
    ):
        with netCDF4.Dataset(filename, mode="r") as ncfile:
            field_from_nc = functools.partial(_as_field_from_nc, ncfile, allocator, dtype=dtype)
            return cls(
                t=field_from_nc("ta"),
                qv=field_from_nc("hus"),
                qc=field_from_nc("clw"),
                qi=field_from_nc("cli"),
                qr=field_from_nc("qr"),
                qs=field_from_nc("qs"),
                qg=field_from_nc("qg"),
                pflx=field_from_nc("pflx", optional=True),
                pr=field_from_nc("prr_gsp", optional=True),
                ps=field_from_nc("prs_gsp", optional=True),
                pi=field_from_nc("pri_gsp", optional=True),
                pg=field_from_nc("prg_gsp", optional=True),
                pre=field_from_nc("pre_gsp", optional=True),
            )

    @property
    def q(self) -> Q:
        return Q(
            v=self.qv,
            c=self.qc,
            r=self.qr,
            s=self.qs,
            i=self.qi,
            g=self.qg,
        )

    def write(self, filename: pathlib.Path | str):
        ncells = self.t.shape[0]
        nlev = self.t.shape[1]

        with netCDF4.Dataset(filename, mode="w") as ncfile:
            ncfile.createDimension("ncells", ncells)
            ncfile.createDimension("height", nlev)
            ncfile.createDimension("surface", 1)

            write_height_field = functools.partial(
                _field_to_nc, ncfile, ("height", "ncells"), dtype=np.float64
            )
            write_surface_field = functools.partial(
                _field_to_nc, ncfile, ("surface", "ncells"), dtype=np.float64
            )

            write_height_field("ta", self.t)
            write_height_field("hus", self.qv)
            write_height_field("clw", self.qc)
            write_height_field("cli", self.qi)
            write_height_field("qr", self.qr)
            write_height_field("qs", self.qs)
            write_height_field("qg", self.qg)
            if self.pflx is not None:
                write_height_field("pflx", self.pflx)
            if self.pr is not None:
                write_surface_field("prr_gsp", self.pr)
            if self.ps is not None:
                write_surface_field("prs_gsp", self.ps)
            if self.pi is not None:
                write_surface_field("pri_gsp", self.pi)
            if self.pg is not None:
                write_surface_field("prg_gsp", self.pg)
            if self.pre is not None:
                write_surface_field("pre_gsp", self.pre)
