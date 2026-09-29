# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import contextlib
import logging
import pathlib
import typing
from typing import Final

import gt4py.next as gtx
import uxarray  # type: ignore[import-untyped] # mypy cannot find the corresponding uxarray type hints
import xarray as xa

import icon4py.model.common.dimension as dim
from icon4py.model.common.grid import gridfile


log = logging.getLogger(__name__)

FILL_VALUE = gridfile.GridFile.INVALID_INDEX
MESH = "mesh"

HORIZONTAL_DIMENSION_MAPPING: Final[dict[gtx.Dimension, str]] = {
    dim.CellDim: "cell",
    dim.EdgeDim: "edge",
    dim.VertexDim: "vertex",
}

COORDINATES_MAPPING: Final[dict[gtx.Dimension, str]] = {
    dim.CellDim: "clon clat",
    dim.VertexDim: "vlon vlat",
    dim.EdgeDim: "elon elat",
}

LOCATION_MAPPING: Final[dict[gtx.Dimension, str]] = {
    dim.CellDim: "face",
    dim.VertexDim: "node",
    dim.EdgeDim: "edge",
}


def extract_horizontal_coordinates(
    ds: xa.Dataset,
) -> dict[str, tuple[xa.DataArray, xa.DataArray]]:
    """
    Extract the coordinates from the ICON grid file.

    TODO(halungge) does it  work for decomposed grids?
    """
    return dict(
        cell=(ds["clat"], ds["clon"]),
        vertex=(ds["vlat"], ds["vlon"]),
        edge=(ds["elat"], ds["elon"]),
    )


def dimension_mapping(dim: gtx.Dimension, is_on_half_levels: bool) -> str:
    assert dim.kind in (
        gtx.DimensionKind.HORIZONTAL,
        gtx.DimensionKind.VERTICAL,
    ), "only horizontal and vertical dimensions are supported."
    if dim.kind == gtx.DimensionKind.VERTICAL:
        return "half_level" if is_on_half_levels else "level"
    else:
        return HORIZONTAL_DIMENSION_MAPPING[dim]


def ugrid_attributes(dim: gtx.Dimension) -> dict:
    if dim.kind == gtx.DimensionKind.HORIZONTAL:
        return dict(
            location=LOCATION_MAPPING[dim],
            coordinates=COORDINATES_MAPPING[dim],
            mesh=MESH,
        )
    else:
        return {}


def extract_bounds(ds: xa.Dataset) -> dict[str, tuple[xa.DataArray, xa.DataArray]]:
    """
    Extract the bounds from the ICON grid file.
    TODO(halungge) does it  work for decomposed grids?
    """
    return dict(
        cell=(ds["clat_vertices"], ds["clon_vertices"]),
        vertex=(ds["vlat_vertices"], ds["vlon_vertices"]),
        edge=(ds["elat_vertices"], ds["elon_vertices"]),
    )


class IconUGridPatcher:
    """
    Patch an ICON grid file with necessary information for UGRID.

    TODO: (magdalena) should all the unnecessary data fields be removed?
    """

    def __init__(self) -> None:
        self.connectivities = (
            "edge_of_cell",  # E2C connectivity
            "vertex_of_cell",  # V2C connectivity
            "adjacent_cell_of_edge",  # C2E connectivity
            "edge_vertices",  # E2V connectivity
            "cells_of_vertex",  # C2V connectivity
            "edges_of_vertex",  # E2V connectivity
            "vertices_of_vertex",  # V2E2V connectivity
            "neighbor_cell_index",  # C2E2C connectivity
        )
        self.horizontal_domain_borders = (
            "start_idx_c",  # start and end indices for refin_ctl_levels
            "end_idx_c",
            "start_idx_e",
            "end_idx_e",
            "start_idx_v",
            "end_idx_v",
        )
        self.index_lists = self.connectivities + self.horizontal_domain_borders
        # TODO(halungge): do not exist on local grid (grid.nc of mch_ch_r04b09_dsl)
        #  what do they contain?
        # "edge_index",
        # "vertex_index",
        # "cell_index",
        # only for nested grids, ignored for now
        # "child_cell_index",
        # "child_edge_index",

    def _add_mesh_var(self, ds: xa.Dataset) -> None:
        """Add the `mesh` variable and mappings for coordinates and connectivities to the ICON grid file."""
        ds["mesh"] = xa.DataArray(
            -1,  # A dummy value for creating the DataArray with the actual attributes
            attrs=dict(
                cf_role="mesh_topology",
                topology_dimension=2,
                node_coordinates="vlon vlat",
                face_node_connectivity="vertex_of_cell",
                face_dimension="cell",
                edge_node_connectivity="edge_vertices",
                edge_dimension="edge",
                edge_coordinates="elon elat",
                face_coordinates="clon clat",
                face_edge_connectivity="edge_of_cell",
                face_face_connectivity="neighbor_cell_index",
                edge_face_connectivity="adjacent_cell_of_edge",
                node_dimension="vertex",
                # TODO(halungge): do we need the boundary_node_connectivity ?
            ),
        )

    def _patch_start_index(self, ds: xa.Dataset, with_zero_start_index: bool = False) -> None:
        """Patch the start index of the index lists in the ICON grid file.

        Adds the 'start_index' attribute to index lists in the grid file.

        TODO: (@halungge) According to UGRID conventions 1 based index arrays should be converted
                on the fly by setting the 'start_index' attribute to 1. We do it manually here until it is implemented
                in UXarray.
        Args:
            ds: ICON grid file as xarray dataset
            with_zero_start_index: If True, the 'start_index' is set to 0 and the indices shifted, otherwise 'start_index' = 1 and no further manipulation done on the indices.

        """
        for var in self.index_lists:
            if var in ds:
                if with_zero_start_index:
                    # work around for uxarray not supporting [start_index] = 1 properly
                    ds[var].data = xa.where(ds[var].data > 0, ds[var].data - 1, FILL_VALUE)
                    ds[var].attrs["start_index"] = 0
                else:
                    ds[var].attrs["start_index"] = 1

    def _set_fill_value(self, ds: xa.Dataset) -> None:
        """Set the '_FillValue' attribute for the connectivity arrays in the ICON grid file."""
        for var in self.connectivities:
            if var in ds:
                ds[var].attrs["_FillValue"] = FILL_VALUE

    def _transpose_index_lists(self, ds: xa.Dataset) -> None:
        """Unify the dimension order of fields in ICON grid file.

        The ICON grid file contains some fields of order (sparse_dimension, horizontal_dimension)
        and others the other way around. We transpose them to have all the same ordering.

        TODO(halungge) should eventually be supported by UXarray.
        """
        for name in self.connectivities:
            shp = ds[name].shape
            if len(shp) == 2 and (shp[0] < shp[1]):
                ds[name] = xa.DataArray(
                    data=ds[name].data.T,
                    dims=ds[name].dims[::-1],
                    coords=ds[name].coords,
                    attrs=ds[name].attrs,
                )

    def _validate(self, ds: xa.Dataset) -> None:
        grid = uxarray.open_grid(ds)
        try:
            grid.validate()
        except RuntimeError as error:
            log.error(f"Validation of the ugrid failed with {error}>")
            raise UGridValidationError("Validation of the ugrid failed") from error

    def __call__(self, ds: xa.Dataset, validate: bool = False) -> xa.Dataset:
        self._patch_start_index(ds, with_zero_start_index=True)
        self._set_fill_value(ds)
        self._transpose_index_lists(ds)
        self._add_mesh_var(ds)
        if validate:
            self._validate(ds)
        return ds


class IconUGridWriter:
    """
    Patch an ICON grid file with necessary information to make it compliant with UGRID conventions.
    """

    def __init__(
        self,
        original_filename: pathlib.Path | str,
        output_path: pathlib.Path | str,
    ):
        self.original_filename = pathlib.Path(original_filename)
        self.output_path = pathlib.Path(output_path)

    def __call__(self, validate: bool = False) -> None:
        patch = IconUGridPatcher()
        with load_data_file(self.original_filename) as ds:
            patched_ds = patch(ds, validate)
            dump_ugrid_file(patched_ds, self.original_filename, self.output_path)


def dump_ugrid_file(
    ds: xa.Dataset, original_filename: pathlib.Path, output_path: pathlib.Path
) -> None:
    stem = original_filename.stem
    filename = output_path.joinpath(stem + "_ugrid.nc")
    ds.to_netcdf(filename, format="NETCDF4", engine="netcdf4")


@contextlib.contextmanager
def load_data_file(filename: pathlib.Path | str) -> typing.Generator[xa.Dataset, None, None]:
    ds = xa.open_dataset(filename)
    try:
        yield ds
    finally:
        ds.close()


class UGridValidationError(Exception):
    pass
