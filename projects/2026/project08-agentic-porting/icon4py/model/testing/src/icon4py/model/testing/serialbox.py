# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
from collections.abc import Sequence
from typing import Literal

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import serialbox

import icon4py.model.common.decomposition.definitions as decomposition
import icon4py.model.common.field_type_aliases as fa
import icon4py.model.common.grid.states as grid_states
from icon4py.model.common import dimension as dims, model_backends, type_alias
from icon4py.model.common.grid import base, horizontal as h_grid, icon, utils as grid_utils
from icon4py.model.common.states import prognostic_state
from icon4py.model.common.states.data import QC, QG, QI, QR, QS, QV
from icon4py.model.common.utils import data_allocation as data_alloc, field_utils


log = logging.getLogger(__name__)

type TimeIndex = Literal[0, 1]
type FourIndex = Literal[0, 1, 2, 3]
type TwoIndex = Literal[0, 1]


type TracerIndex = Literal[QV, QC, QI, QR, QS, QG]


class IconSavepoint:
    def __init__(
        self,
        sp: serialbox.Savepoint,
        ser: serialbox.Serializer,
        size: dict,
        backend: gtx_typing.Backend | None,
    ):
        self.savepoint = sp
        self.serializer = ser
        self.sizes = size
        self.log = logging.getLogger(__name__)
        self.backend = backend
        self.xp = data_alloc.import_array_ns(self.backend)

    def optionally_registered(*dims, dtype=type_alias.wpfloat):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                try:
                    name = func.__name__
                    return func(self, *args, **kwargs)
                except serialbox.SerialboxError:
                    log.warning(
                        f"{name}: field not registered in savepoint {self.savepoint.metainfo}"
                    )
                    if dims:
                        # We allocate a dummy field with size 1 in each dimension
                        # as a workaround for the lack of support for optional fields in gt4py.
                        shp = (1,) * len(dims)
                        return gtx.as_field(
                            dims, self.xp.zeros(shp, dtype=dtype), allocator=self.backend
                        )
                    else:
                        return None

            return wrapper

        return decorator

    def log_meta_info(self):
        self.log.info(self.savepoint.metainfo)

    def _get_field(
        self,
        name,
        *dimensions,
        dtype=float,
        slice_: int | slice | tuple[int | slice, ...] | None = None,
        transpose: None | Sequence[int] = None,
    ):
        # Note: slice is applied before transpose!
        buffer = self.xp.squeeze(self.serializer.read(name, self.savepoint).astype(dtype))
        if slice_ is not None:
            buffer = buffer[slice_]
        if transpose is not None:
            buffer = self.xp.transpose(buffer, axes=transpose)
        buffer = self._reduce_to_dim_size(buffer, dimensions)

        self.log.debug(f"{name} {buffer.shape}")
        return gtx.as_field(dimensions, buffer, allocator=self.backend)

    def _get_field_component(self, name: str, level: int, dims: tuple[gtx.Dimension, gtx]):
        buffer = self.serializer.read(name, self.savepoint).astype(float)
        buffer = self.xp.squeeze(buffer)[:, :, level]
        buffer = self._reduce_to_dim_size(buffer, dims)
        self.log.debug(f"{name} {buffer.shape}")
        return gtx.as_field(dims, buffer, allocator=self.backend)

    def _reduce_to_dim_size(self, buffer, dimensions):
        buffer_size = (
            self.sizes[d] if d.kind is gtx.DimensionKind.HORIZONTAL else s
            for s, d in zip(buffer.shape, dimensions, strict=False)
        )
        return buffer[tuple(map(slice, buffer_size))]

    def _get_field_from_ndarray(self, ar, *dimensions, dtype=float):
        ar = self._reduce_to_dim_size(ar, dimensions)
        return gtx.as_field(dimensions, ar, allocator=self.backend, dtype=dtype)

    def get_metadata(self, *names):
        metadata = self.savepoint.metainfo.to_dict()
        return {n: metadata[n] for n in names if n in metadata}

    def _read_int32_shift1(self, name: str):
        """
        Read a start indices field.

        use for start indices: the shift accounts for the zero based python
        values are converted to gtx.int32
        """
        return self._read_int32(name, offset=1)

    def _read_int32(self, name: str, offset=0):
        """
        Read an end indices field.

        use this for end indices: because FORTRAN slices  are inclusive [from:to] _and_ one based
        this accounts for being exclusive python exclusive bounds: [from:to)
        field values are convert to gtx.int32
        """
        return self._read(name, offset, dtype=gtx.int32)

    def _read_bool(self, name: str):
        return self._read(name, offset=0, dtype=bool)

    def _read(self, name: str, offset=0, dtype=int):
        return self.xp.asarray(
            self.xp.squeeze(self.serializer.read(name, self.savepoint) - offset).astype(dtype)
        )


class IconGridSavepoint(IconSavepoint):
    def __init__(
        self,
        *,
        sp: serialbox.Savepoint,
        ser: serialbox.Serializer,
        grid_id: str,
        size: dict,
        grid_params: icon.GridParams,
        backend: gtx_typing.Backend | None,
    ):
        super().__init__(sp, ser, size, backend)
        self._grid_id = grid_id
        self.grid_params = grid_params

    def verts_vertex_lat(self):
        """vertex latituted"""
        return self._get_field("verts_vertex_lat", dims.VertexDim)

    def verts_vertex_lon(self):
        """vertex longitude"""
        return self._get_field("verts_vertex_lon", dims.VertexDim)

    def verts_vertex_cart_x(self):
        """vertex cartesian x coordinate"""
        return self._get_field("verts_vertex_cart_x", dims.VertexDim)

    def verts_vertex_cart_y(self):
        """vertex cartesian y coordinate"""
        return self._get_field("verts_vertex_cart_y", dims.VertexDim)

    def verts_vertex_cart_z(self):
        """vertex cartesian z coordinate"""
        return self._get_field("verts_vertex_cart_z", dims.VertexDim)

    def primal_normal_v1(self):
        return self._get_field("primal_normal_v1", dims.EdgeDim)

    def primal_normal_v2(self):
        return self._get_field("primal_normal_v2", dims.EdgeDim)

    def dual_normal_v1(self):
        return self._get_field("dual_normal_v1", dims.EdgeDim)

    def dual_normal_v2(self):
        return self._get_field("dual_normal_v2", dims.EdgeDim)

    def edges_center_lat(self):
        """edge center latitude"""
        return self._get_field("edges_center_lat", dims.EdgeDim)

    def edges_center_lon(self):
        """edge center longitude"""
        return self._get_field("edges_center_lon", dims.EdgeDim)

    def edges_center_cart_x(self):
        """edge center cartesian x coordinate"""
        return self._get_field("edges_center_cart_x", dims.EdgeDim)

    def edges_center_cart_y(self):
        """edge center cartesian y coordinate"""
        return self._get_field("edges_center_cart_y", dims.EdgeDim)

    def edges_center_cart_z(self):
        """edge center cartesian z coordinate"""
        return self._get_field("edges_center_cart_z", dims.EdgeDim)

    def edge_vert_length(self):
        """length of edge midpoint to vertex"""
        return self._get_field("edge_vert_length", dims.EdgeDim, dims.E2C2VDim)

    def vct_a(self):
        return self._get_field("vct_a", dims.KDim)

    def vct_b(self):
        return self._get_field("vct_b", dims.KDim)

    def tangent_orientation(self):
        return self._get_field("tangent_orientation", dims.EdgeDim)

    def edge_orientation(self):
        return self._get_field("cells_edge_orientation", dims.CellDim, dims.C2EDim)

    def vertex_edge_orientation(self):
        return self._get_field("v_edge_orientation", dims.VertexDim, dims.V2EDim)

    def vertex_dual_area(self):
        return self._get_field("v_dual_area", dims.VertexDim)

    def inverse_primal_edge_lengths(self):
        return self._get_field("inv_primal_edge_length", dims.EdgeDim)

    def primal_edge_length(self):
        return self._get_field("primal_edge_length", dims.EdgeDim)

    def primal_cart_normal_x(self):
        return self._get_field("primal_cart_normal_x", dims.EdgeDim)

    def primal_cart_normal_y(self):
        return self._get_field("primal_cart_normal_y", dims.EdgeDim)

    def primal_cart_normal_z(self):
        return self._get_field("primal_cart_normal_z", dims.EdgeDim)

    def dual_cart_normal_x(self):
        return self._get_field("dual_cart_normal_x", dims.EdgeDim)

    def dual_cart_normal_y(self):
        return self._get_field("dual_cart_normal_y", dims.EdgeDim)

    def dual_cart_normal_z(self):
        return self._get_field("dual_cart_normal_z", dims.EdgeDim)

    def inv_vert_vert_length(self):
        return self._get_field("inv_vert_vert_length", dims.EdgeDim)

    def primal_normal_vert_x(self):
        return self._get_field("primal_normal_vert_x", dims.EdgeDim, dims.E2C2VDim)

    def primal_normal_vert_y(self):
        return self._get_field("primal_normal_vert_y", dims.EdgeDim, dims.E2C2VDim)

    def dual_normal_vert_y(self):
        return self._get_field("dual_normal_vert_y", dims.EdgeDim, dims.E2C2VDim)

    def dual_normal_vert_x(self):
        return self._get_field("dual_normal_vert_x", dims.EdgeDim, dims.E2C2VDim)

    def primal_normal_cell_x(self):
        return self._get_field("primal_normal_cell_x", dims.EdgeDim, dims.E2CDim)

    def primal_normal_cell_y(self):
        return self._get_field("primal_normal_cell_y", dims.EdgeDim, dims.E2CDim)

    def dual_normal_cell_x(self):
        return self._get_field("dual_normal_cell_x", dims.EdgeDim, dims.E2CDim)

    def dual_normal_cell_y(self):
        return self._get_field("dual_normal_cell_y", dims.EdgeDim, dims.E2CDim)

    def cell_areas(self):
        return self._get_field("cell_areas", dims.CellDim)

    def lat(self, dim: gtx.Dimension):
        match dim:
            case dims.CellDim:
                return self.cell_center_lat()
            case dims.EdgeDim:
                return self.edges_center_lat()
            case dims.VertexDim:
                return self.verts_vertex_lat()
            case _:
                raise ValueError

    def lon(self, dim: gtx.Dimension):
        match dim:
            case dims.CellDim:
                return self.cell_center_lon()
            case dims.EdgeDim:
                return self.edges_center_lon()
            case dims.VertexDim:
                return self.verts_vertex_lon()
            case _:
                raise ValueError

    def coordinates(self):
        coords = {
            dims.CellDim: {"lat": self.cell_center_lat(), "lon": self.cell_center_lon()},
            dims.EdgeDim: {"lat": self.edges_center_lat(), "lon": self.edges_center_lon()},
            dims.VertexDim: {"lat": self.verts_vertex_lat(), "lon": self.verts_vertex_lon()},
        }

        if self.grid_params.geometry_type == icon.GeometryType.TORUS:
            coords[dims.CellDim]["x"] = self.cell_center_cart_x()
            coords[dims.CellDim]["y"] = self.cell_center_cart_y()
            coords[dims.CellDim]["z"] = self.cell_center_cart_z()
            coords[dims.EdgeDim]["x"] = self.edges_center_cart_x()
            coords[dims.EdgeDim]["y"] = self.edges_center_cart_y()
            coords[dims.EdgeDim]["z"] = self.edges_center_cart_z()
            coords[dims.VertexDim]["x"] = self.verts_vertex_cart_x()
            coords[dims.VertexDim]["y"] = self.verts_vertex_cart_y()
            coords[dims.VertexDim]["z"] = self.verts_vertex_cart_z()

        return coords

    def cell_center_lat(self):
        return self._get_field("cell_center_lat", dims.CellDim)

    def cell_center_lon(self):
        return self._get_field("cell_center_lon", dims.CellDim)

    def cell_center_cart_x(self):
        """cell center cartesian x coordinate"""
        return self._get_field("cell_center_cart_x", dims.CellDim)

    def cell_center_cart_y(self):
        """cell center cartesian y coordinate"""
        return self._get_field("cell_center_cart_y", dims.CellDim)

    def cell_center_cart_z(self):
        """cell center cartesian z coordinate"""
        return self._get_field("cell_center_cart_z", dims.CellDim)

    def edge_center_lat(self):
        return self._get_field("edges_center_lat", dims.EdgeDim)

    def edge_center_lon(self):
        return self._get_field("edges_center_lon", dims.EdgeDim)

    def mean_cell_area(self):
        return self.serializer.read("mean_cell_area", self.savepoint).astype(float)[0]

    def edge_areas(self):
        return self._get_field("edge_areas", dims.EdgeDim)

    def inv_dual_edge_length(self):
        return self._get_field("inv_dual_edge_length", dims.EdgeDim)

    def dual_edge_length(self):
        return self._get_field("dual_edge_length", dims.EdgeDim)

    def edge_cell_length(self):
        """length of edge midpoint to cell center"""
        return self._get_field("edge_cell_length", dims.EdgeDim, dims.E2CDim)

    def cells_start_index(self):
        start_idx = self._read_int32("c_start_index")
        return self.xp.where(start_idx == 0, start_idx, start_idx - 1)

    def cells_end_index(self):
        return self._read_int32("c_end_index")

    def vertex_start_index(self):
        start_idx = self._read_int32("v_start_index")
        return self.xp.where(start_idx == 0, start_idx, start_idx - 1)

    def vertex_end_index(self):
        return self._read_int32("v_end_index")

    def edge_start_index(self):
        start_idx = self._read_int32("e_start_index")
        return self.xp.where(start_idx == 0, start_idx, start_idx - 1)

    def edge_end_index(self):
        # don't need to subtract 1, because FORTRAN slices  are inclusive [from:to] so the being
        # one off accounts for being exclusive [from:to)
        return self._read_int32("e_end_index")

    def start_index(self) -> dict[gtx.Dimension, data_alloc.NDArray]:
        return {
            dims.CellDim: self.cells_start_index(),
            dims.EdgeDim: self.edge_start_index(),
            dims.VertexDim: self.vertex_start_index(),
        }

    def end_index(self) -> dict[gtx.Dimension, data_alloc.NDArray]:
        return {
            dims.CellDim: self.cells_end_index(),
            dims.EdgeDim: self.edge_end_index(),
            dims.VertexDim: self.vertex_end_index(),
        }

    def nflatlev(self):
        return self._read_int32_shift1("nflatlev").item()

    def nflat_gradp(self):
        return self._read_int32_shift1("nflat_gradp").item()

    def v_owner_mask(self):
        return self._get_field("v_owner_mask", dims.VertexDim, dtype=bool)

    def c_owner_mask(self):
        return self._get_field("c_owner_mask", dims.CellDim, dtype=bool)

    def e_owner_mask(self):
        return self._get_field("e_owner_mask", dims.EdgeDim, dtype=bool)

    def f_e(self):
        return self._get_field("f_e", dims.EdgeDim)

    def print_connectivity_info(self, name: str, ar: data_alloc.NDArray):
        self.log.debug(f" connectivity {name} {ar.shape}")

    def c2e(self):
        return self._get_connectivity_array("c2e", dims.CellDim)

    def _get_connectivity_array(self, name: str, target_dim: gtx.Dimension, reverse: bool = False):
        if reverse:
            connectivity = self.xp.transpose(self._read_int32(name, offset=1))[
                : self.sizes[target_dim], :
            ]
        else:
            connectivity = self._read_int32(name, offset=1)[: self.sizes[target_dim], :]
        self.log.debug(f" connectivity {name} : {connectivity.shape}")
        return connectivity

    def c2e2c(self):
        return self._get_connectivity_array("c2e2c", dims.CellDim)

    def e2c2e(self):
        return self._get_connectivity_array("e2c2e", dims.EdgeDim)

    def c2e2c2e(self):
        if self._c2e2c2e() is None:
            return self.xp.zeros((self.sizes[dims.CellDim], 9), dtype=gtx.int32)
        else:
            return self._c2e2c2e()

    @IconSavepoint.optionally_registered()
    def _c2e2c2e(self):
        return self._get_connectivity_array("c2e2c2e", dims.CellDim, reverse=True)

    def e2c(self):
        return self._get_connectivity_array("e2c", dims.EdgeDim)

    def e2v(self):
        # array "e2v" is actually e2c2v
        v_ = self._get_connectivity_array("e2v", dims.EdgeDim)[:, 0:2]
        self.log.debug(f"real e2v {v_.shape}")
        return v_

    def e2c2v(self):
        # array "e2v" is actually e2c2v, that is hexagon or pentagon
        return self._get_connectivity_array("e2v", dims.EdgeDim)

    def v2e(self):
        return self._get_connectivity_array("v2e", dims.VertexDim)

    def v2c(self):
        return self._get_connectivity_array("v2c", dims.VertexDim)

    def c2v(self):
        return self._get_connectivity_array("c2v", dims.CellDim)

    def nrdmax(self):
        return gtx.int32(self._read_int32_shift1("nrdmax").item())

    def refin_ctrl(self, dim: gtx.Dimension):
        field_name = "refin_ctl"
        return gtx.as_field(
            (dim,),
            self._read_field_for_dim(field_name, self._read_int32, dim)[: self.num(dim)],
            allocator=self.backend,
        )

    def num(self, dim: gtx.Dimension):
        return self.sizes[dim]

    @staticmethod
    def _read_field_for_dim(field_name, read_func, dim: gtx.Dimension):
        match dim:
            case dims.CellDim:
                return read_func(f"c_{field_name}")
            case dims.EdgeDim:
                return read_func(f"e_{field_name}")
            case dims.VertexDim:
                return read_func(f"v_{field_name}")
            case _:
                raise NotImplementedError(
                    f"only {dims.CellDim, dims.EdgeDim, dims.VertexDim} are handled"
                )

    def owner_mask(self, dim: gtx.Dimension):
        return self.xp.squeeze(self._read_field_for_dim("owner_mask", self._read_bool, dim))

    def global_index(self, dim: gtx.Dimension):
        return self._read_field_for_dim("glb_index", self._read_int32_shift1, dim)

    def decomp_domain(self, dim):
        return self._read_field_for_dim("decomp_domain", self._read_int32, dim)

    def construct_decomposition_info(self) -> decomposition.DecompositionInfo:
        return (
            decomposition.DecompositionInfo()
            .set_dimension(*self._get_decomposition_fields(dims.CellDim))
            .set_dimension(*self._get_decomposition_fields(dims.EdgeDim))
            .set_dimension(*self._get_decomposition_fields(dims.VertexDim))
        )

    def _get_decomposition_fields(self, dim: gtx.Dimension):
        global_index = self.global_index(dim)
        mask = self.owner_mask(dim)[0 : self.num(dim)]
        halo_levels = self.decomp_domain(dim)[0 : self.num(dim)]
        return dim, global_index, mask, halo_levels

    def construct_icon_grid(
        self,
        backend: gtx_typing.Backend | None = None,
        keep_skip_values: bool = True,
        with_repeated_index: bool = True,
    ) -> icon.IconGrid:
        config = base.GridConfig(
            horizontal_config=base.HorizontalGridSize(
                num_vertices=self.num(dims.VertexDim),
                num_cells=self.num(dims.CellDim),
                num_edges=self.num(dims.EdgeDim),
            ),
            vertical_size=self.num(dims.KDim),
            limited_area=self.get_metadata("limited_area").get("limited_area"),
            distributed=self.construct_decomposition_info().is_distributed(),
            keep_skip_values=keep_skip_values,
        )

        if with_repeated_index:

            def potentially_revert_icon_index_transformation(ar):
                return ar
        else:
            potentially_revert_icon_index_transformation = (
                grid_utils.revert_repeated_index_to_invalid
            )

        c2e2c = self.c2e2c()
        e2c2e = potentially_revert_icon_index_transformation(self.e2c2e())
        c2e2c0 = self.xp.column_stack((self.xp.asarray(range(c2e2c.shape[0])), c2e2c))
        e2c2e0 = self.xp.column_stack((self.xp.asarray(range(e2c2e.shape[0])), e2c2e))

        constructor = functools.partial(
            h_grid.get_start_end_idx_from_icon_arrays,
            start_indices=self.start_index(),
            end_indices=self.end_index(),
        )
        c2e2c2e = potentially_revert_icon_index_transformation(self.c2e2c2e())
        v2e = potentially_revert_icon_index_transformation(self.v2e())

        start_index, end_index = icon.get_start_and_end_index(constructor)
        neighbor_tables = {
            dims.C2E: self.c2e(),
            dims.E2C: self.e2c(),
            dims.C2E2C: c2e2c,
            dims.C2E2CO: c2e2c0,
            dims.C2E2C2E: c2e2c2e,
            dims.E2C2E: e2c2e,
            dims.E2C2EO: e2c2e0,
            dims.E2V: self.e2v(),
            dims.V2E: v2e,
            dims.V2C: self.v2c(),
            dims.E2C2V: self.e2c2v(),
            dims.C2V: self.c2v(),
        }

        return icon.icon_grid(
            id_=self._grid_id,
            allocator=backend,
            config=config,
            neighbor_tables=neighbor_tables,
            grid_params=self.grid_params,
            start_index=start_index,
            end_index=end_index,
            refinement_control={
                dims.CellDim: self.refin_ctrl(dims.CellDim),
                dims.EdgeDim: self.refin_ctrl(dims.EdgeDim),
                dims.VertexDim: self.refin_ctrl(dims.VertexDim),
            },
        )

    def construct_edge_geometry(self) -> grid_states.EdgeParams:
        return grid_states.EdgeParams(
            tangent_orientation=self.tangent_orientation(),
            inverse_primal_edge_lengths=self.inverse_primal_edge_lengths(),
            inverse_dual_edge_lengths=self.inv_dual_edge_length(),
            inverse_vertex_vertex_lengths=self.inv_vert_vert_length(),
            primal_normal_vert_x=self.primal_normal_vert_x(),
            primal_normal_vert_y=self.primal_normal_vert_y(),
            dual_normal_vert_x=self.dual_normal_vert_x(),
            dual_normal_vert_y=self.dual_normal_vert_y(),
            primal_normal_cell_x=self.primal_normal_cell_x(),
            dual_normal_cell_x=self.dual_normal_cell_x(),
            primal_normal_cell_y=self.primal_normal_cell_y(),
            dual_normal_cell_y=self.dual_normal_cell_y(),
            edge_areas=self.edge_areas(),
            coriolis_frequency=self.f_e(),
            edge_center_lat=self.edge_center_lat(),
            edge_center_lon=self.edge_center_lon(),
            primal_normal_x=self.primal_normal_v1(),
            primal_normal_y=self.primal_normal_v2(),
        )

    def construct_cell_geometry(self) -> grid_states.CellParams:
        return grid_states.CellParams(
            cell_center_lat=self.cell_center_lat(),
            cell_center_lon=self.cell_center_lon(),
            area=self.cell_areas(),
            mean_cell_area=self.mean_cell_area(),
        )


class InterpolationSavepoint(IconSavepoint):
    def c_bln_avg(self):
        return self._get_field("c_bln_avg", dims.CellDim, dims.C2E2CODim)

    def c_intp(self):
        return self._get_field("c_intp", dims.VertexDim, dims.V2CDim)

    def c_lin_e(self):
        return self._get_field("c_lin_e", dims.EdgeDim, dims.E2CDim)

    def e_bln_c_s(self):
        return self._get_field("e_bln_c_s", dims.CellDim, dims.C2EDim)

    def e_flx_avg(self):
        return self._get_field("e_flx_avg", dims.EdgeDim, dims.E2C2EODim)

    def geofac_div(self):
        return self._get_field("geofac_div", dims.CellDim, dims.C2EDim)

    def geofac_grdiv(self):
        return self._get_field("geofac_grdiv", dims.EdgeDim, dims.E2C2EODim)

    def geofac_grg(self):
        grg = self.xp.squeeze(self.serializer.read("geofac_grg", self.savepoint))
        num_cells = self.sizes[dims.CellDim]
        return gtx.as_field(
            (dims.CellDim, dims.C2E2CODim), grg[:num_cells, :, 0], allocator=self.backend
        ), gtx.as_field(
            (dims.CellDim, dims.C2E2CODim), grg[:num_cells, :, 1], allocator=self.backend
        )

    def geofac_n2s(self):
        return self._get_field("geofac_n2s", dims.CellDim, dims.C2E2CODim)

    def geofac_rot(self):
        return self._get_field("geofac_rot", dims.VertexDim, dims.V2EDim)

    def nudgecoeff_e(self):
        return self._get_field("nudgecoeff_e", dims.EdgeDim)

    def pos_on_tplane_e_x(self):
        return self._get_field(
            "pos_on_tplane_e_x", dims.EdgeDim, dims.E2CDim, slice_=(slice(None), slice(0, 2))
        )

    def pos_on_tplane_e_y(self):
        return self._get_field(
            "pos_on_tplane_e_y", dims.EdgeDim, dims.E2CDim, slice_=(slice(None), slice(0, 2))
        )

    def rbf_vec_coeff_e(self):
        return self._get_field("rbf_vec_coeff_e", dims.EdgeDim, dims.E2C2EDim, transpose=(1, 0))

    @IconSavepoint.optionally_registered()
    def rbf_vec_coeff_c1(self):
        return self._get_field("rbf_vec_coeff_c1", dims.CellDim, dims.C2E2C2EDim, transpose=(1, 0))

    @IconSavepoint.optionally_registered()
    def rbf_vec_coeff_c2(self):
        return self._get_field("rbf_vec_coeff_c2", dims.CellDim, dims.C2E2C2EDim, transpose=(1, 0))

    def rbf_vec_coeff_v1(self):
        return self._get_field(
            "rbf_vec_coeff_v",
            dims.VertexDim,
            dims.V2EDim,
            slice_=(slice(None), 0, slice(None)),
            transpose=(1, 0),
        )

    def rbf_vec_coeff_v2(self):
        return self._get_field(
            "rbf_vec_coeff_v",
            dims.VertexDim,
            dims.V2EDim,
            slice_=(slice(None), 1, slice(None)),
            transpose=(1, 0),
        )

    def rbf_vec_idx_v(self):
        return self._get_field("rbf_vec_idx_v", dims.VertexDim, dims.V2EDim)

    def lsq_pseudoinv_1(self):
        return self._get_field("lsq_pseudoinv_1", dims.CellDim, dims.C2E2CDim)

    def lsq_pseudoinv_2(self):
        return self._get_field("lsq_pseudoinv_2", dims.CellDim, dims.C2E2CDim)


class MetricSavepoint(IconSavepoint):
    def d2dexdz2_fac1_mc(self):
        return self._get_field("d2dexdz2_fac1_mc", dims.CellDim, dims.KDim)

    def d2dexdz2_fac2_mc(self):
        return self._get_field("d2dexdz2_fac2_mc", dims.CellDim, dims.KDim)

    def d_exner_dz_ref_ic(self):
        return self._get_field("d_exner_dz_ref_ic", dims.CellDim, dims.KDim)

    def exner_exfac(self):
        return self._get_field("exner_exfac", dims.CellDim, dims.KDim)

    def exner_ref_mc(self):
        return self._get_field("exner_ref_mc", dims.CellDim, dims.KDim)

    def hmask_dd3d(self):
        return self._get_field("hmask_dd3d", dims.EdgeDim)

    def inv_ddqz_z_full(self):
        return self._get_field("inv_ddqz_z_full", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def ddqz_z_full(self):
        return self._get_field("ddqz_z_full", dims.CellDim, dims.KDim)

    def mask_prog_halo_c(self):
        return self._get_field("mask_prog_halo_c", dims.CellDim, dtype=bool)

    @IconSavepoint.optionally_registered()
    def pg_edgeidx(self):
        return self.xp.squeeze(self.serializer.read("pg_edgeidx", self.savepoint))

    @IconSavepoint.optionally_registered()
    def pg_vertidx(self):
        return self.xp.squeeze(self.serializer.read("pg_vertidx", self.savepoint))

    @IconSavepoint.optionally_registered()
    def pg_exdist(self):
        return self.xp.squeeze(self.serializer.read("pg_exdist", self.savepoint))

    def pg_exdist_dsl(self):
        pg_edgeidx = self.pg_edgeidx()
        pg_vertidx = self.pg_vertidx()
        pg_exdist = self.pg_exdist()
        domain = self.rho_ref_me().domain
        default_value = gtx.float64(0.0)
        if (pg_edgeidx is None) or (pg_vertidx is None) or (pg_exdist is None):
            # if any of the fields is missing, return a zero field with the correct shape
            return gtx.as_field(
                domain,
                self.xp.full(domain.shape, fill_value=default_value, dtype=gtx.float64),
                allocator=model_backends.get_allocator(self.backend),
            )
        else:
            return data_alloc.list2field(
                domain=domain,
                values=pg_exdist,
                indices=(
                    data_alloc.adjust_fortran_indices(pg_edgeidx),
                    data_alloc.adjust_fortran_indices(pg_vertidx),
                ),
                default_value=default_value,
                allocator=model_backends.get_allocator(self.backend),
            )

    def rayleigh_w(self):
        return self._get_field("rayleigh_w", dims.KDim)

    def rho_ref_mc(self):
        return self._get_field("rho_ref_mc", dims.CellDim, dims.KDim)

    def rho_ref_me(self):
        return self._get_field("rho_ref_me", dims.EdgeDim, dims.KDim)

    def scalfac_dd3d(self):
        return self._get_field("scalfac_dd3d", dims.KDim)

    def theta_ref_ic(self):
        return self._get_field("theta_ref_ic", dims.CellDim, dims.KDim)

    def z_ifc(self):
        return self._get_field("z_ifc", dims.CellDim, dims.KDim)

    def z_mc(self):
        return self._get_field("z_mc", dims.CellDim, dims.KDim)

    def theta_ref_me(self):
        return self._get_field("theta_ref_me", dims.EdgeDim, dims.KDim)

    def vwind_expl_wgt(self):
        return self._get_field("vwind_expl_wgt", dims.CellDim)

    def vwind_impl_wgt(self):
        return self._get_field("vwind_impl_wgt", dims.CellDim)

    def wgtfacq_c(self):
        # The Fortran array stores the surface levels in reversed order.
        wgtfacq_c_fortran = self._get_field("wgtfacq_c", dims.CellDim, dims.KDim)
        assert len(wgtfacq_c_fortran.domain[dims.KDim].unit_range) == 3
        nlev = self.sizes[dims.KDim]
        return field_utils.flip(
            wgtfacq_c_fortran(dims.KDim - (nlev - 3)),  # GT4Py embedded shift
            dims.KDim,
            allocator=model_backends.get_allocator(self.backend),
        )

    def zdiff_gradp(self):
        return self._get_field("zdiff_gradp", dims.EdgeDim, dims.E2CDim, dims.KDim)

    def vertoffset_gradp(self):
        # In Fortran `vertidx_gradp` contains `0`s in areas where the array is not used.
        # When we translate to offsets we just subtract the current index, therefore these values will be negative.
        # Since in Fortran accessing index `0` would be out-of-bounds, we should be safe.
        vertidx_gradp = data_alloc.adjust_fortran_indices(
            self._get_field("vertidx_gradp", dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32)
        )
        return field_utils.index2offset(vertidx_gradp, dims.KDim, self.backend)

    def coeff1_dwdz(self):
        return self._get_field("coeff1_dwdz", dims.CellDim, dims.KDim)

    def coeff2_dwdz(self):
        return self._get_field("coeff2_dwdz", dims.CellDim, dims.KDim)

    def coeff_gradekin(self):
        return self._get_field("coeff_gradekin", dims.EdgeDim, dims.E2CDim)

    def ddqz_z_full_e(self):
        return self._get_field("ddqz_z_full_e", dims.EdgeDim, dims.KDim)

    def ddqz_z_half(self):
        return self._get_field("ddqz_z_half", dims.CellDim, dims.KDim)

    def ddxn_z_full(self):
        return self._get_field("ddxn_z_full", dims.EdgeDim, dims.KDim)

    def ddxt_z_full(self):
        return self._get_field("ddxt_z_full", dims.EdgeDim, dims.KDim)

    def theta_ref_mc(self):
        return self._get_field("theta_ref_mc", dims.CellDim, dims.KDim)

    def wgtfac_c(self):
        return self._get_field("wgtfac_c", dims.CellDim, dims.KDim)

    def wgtfac_e(self):
        return self._get_field("wgtfac_e", dims.EdgeDim, dims.KDim)

    def wgtfacq_e(self):
        # The Fortran array stores the surface levels in reversed order.
        wgtfacq_e_fortran = self._get_field("wgtfacq_e", dims.EdgeDim, dims.KDim)
        assert len(wgtfacq_e_fortran.domain[dims.KDim].unit_range) == 3
        nlev = self.sizes[dims.KDim]
        return field_utils.flip(
            wgtfacq_e_fortran(dims.KDim - (nlev - 3)),  # GT4Py embedded shift
            dims.KDim,
            allocator=model_backends.get_allocator(self.backend),
        )

    def geopot(self):
        return self._get_field("geopot", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def zd_cellidx(self):
        return self.xp.squeeze(self.serializer.read("zd_cellidx", self.savepoint))

    @IconSavepoint.optionally_registered()
    def zd_vertidx(self):
        # this is the k list (with fortran 1-based indexing) for the central point of the C2E2C stencil
        return self.xp.squeeze(self.serializer.read("zd_vertidx", self.savepoint))[0, :]

    @IconSavepoint.optionally_registered(dims.CellDim, dims.C2E2CDim, dims.KDim, dtype=gtx.int32)
    def zd_vertoffset(self):
        zd_cellidx = self.zd_cellidx()
        zd_vertidx = self.zd_vertidx()
        # these are the three k offsets for the C2E2C neighbors
        zd_vertoffset = (
            self.xp.squeeze(self.serializer.read("zd_vertidx", self.savepoint))[1:, :] - zd_vertidx
        )
        cell_c2e2c_k_domain = gtx.domain(
            {
                dims.CellDim: self.theta_ref_mc().domain[dims.CellDim].unit_range,
                dims.C2E2CDim: 3,
                dims.KDim: self.theta_ref_mc().domain[dims.KDim].unit_range,
            }
        )
        return data_alloc.list2field(
            domain=cell_c2e2c_k_domain,
            values=zd_vertoffset.T,
            indices=(
                data_alloc.adjust_fortran_indices(zd_cellidx),
                slice(None),
                data_alloc.adjust_fortran_indices(zd_vertidx),
            ),
            default_value=gtx.int32(0),
            allocator=model_backends.get_allocator(self.backend),
        )

    @IconSavepoint.optionally_registered(dims.CellDim, dims.C2E2CDim, dims.KDim)
    def zd_intcoef(self):
        zd_cellidx = self.zd_cellidx()
        zd_vertidx = self.zd_vertidx()
        zd_intcoef = self.xp.squeeze(self.serializer.read("zd_intcoef", self.savepoint))
        cell_c2e2c_k_domain = gtx.domain(
            {
                dims.CellDim: self.theta_ref_mc().domain[dims.CellDim].unit_range,
                dims.C2E2CDim: 3,
                dims.KDim: self.theta_ref_mc().domain[dims.KDim].unit_range,
            }
        )
        return data_alloc.list2field(
            domain=cell_c2e2c_k_domain,
            values=zd_intcoef.T,
            indices=(
                data_alloc.adjust_fortran_indices(zd_cellidx),
                slice(None),
                data_alloc.adjust_fortran_indices(zd_vertidx),
            ),
            default_value=gtx.float64(0.0),
            allocator=model_backends.get_allocator(self.backend),
        )

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def zd_diffcoef(self):
        zd_cellidx = self.zd_cellidx()
        zd_vertidx = self.zd_vertidx()
        zd_diffcoef = self.xp.squeeze(self.serializer.read("zd_diffcoef", self.savepoint))
        return data_alloc.list2field(
            domain=self.geopot().domain,
            values=zd_diffcoef,
            indices=(
                data_alloc.adjust_fortran_indices(zd_cellidx),
                data_alloc.adjust_fortran_indices(zd_vertidx),
            ),
            default_value=gtx.float64(0.0),
            allocator=model_backends.get_allocator(self.backend),
        )


class AdvectionInitSavepoint(IconSavepoint):
    def airmass_now(self):
        return self._get_field("airmass_now", dims.CellDim, dims.KDim)

    def airmass_new(self):
        return self._get_field("airmass_new", dims.CellDim, dims.KDim)

    def vn_traj(self):
        return self._get_field("vn_traj", dims.EdgeDim, dims.KDim)

    def mass_flx_me(self):
        return self._get_field("mass_flx_me", dims.EdgeDim, dims.KDim)

    def mass_flx_ic(self):
        return self._get_field("mass_flx_ic", dims.CellDim, dims.KDim)

    def grf_tend_tracer(self, ntracer: int):
        return self._get_field_component("grf_tend_tracers", ntracer, (dims.CellDim, dims.KDim))

    def tracer(self, ntracer: int):
        return self._get_field_component("tracers_now", ntracer, (dims.CellDim, dims.KDim))


class AdvectionExitSavepoint(IconSavepoint):
    def hfl_tracer(self, ntracer: int):
        return self._get_field_component("hfl_tracers", ntracer, (dims.EdgeDim, dims.KDim))

    def vfl_tracer(self, ntracer: int):
        return self._get_field_component("vfl_tracers", ntracer, (dims.CellDim, dims.KDim))

    def tracer(self, ntracer: int):
        return self._get_field_component("tracers", ntracer, (dims.CellDim, dims.KDim))


class IconDiffusionInitSavepoint(IconSavepoint):
    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def hdef_ic(self):
        return self._get_field("hdef_ic", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def div_ic(self):
        return self._get_field("div_ic", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def dwdx(self):
        return self._get_field("dwdx", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered(dims.CellDim, dims.KDim)
    def dwdy(self):
        return self._get_field("dwdy", dims.CellDim, dims.KDim)

    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def theta_v(self):
        return self._get_field("theta_v", dims.CellDim, dims.KDim)

    def w(self):
        return self._get_field("w", dims.CellDim, dims.KDim)

    def exner(self):
        return self._get_field("exner", dims.CellDim, dims.KDim)

    def diff_multfac_smag(self):
        return self.xp.squeeze(self.serializer.read("diff_multfac_smag", self.savepoint))

    def enh_smag_fac(self):
        return self.xp.squeeze(self.serializer.read("enh_smag_fac", self.savepoint))

    def smag_limit(self):
        return self.xp.squeeze(self.serializer.read("smag_limit", self.savepoint))

    def diff_multfac_n2w(self):
        return self.xp.squeeze(self.serializer.read("diff_multfac_n2w", self.savepoint))

    def nudgezone_diff(self) -> int:
        return self.serializer.read("nudgezone_diff", self.savepoint)[0]

    def bdy_diff(self) -> int:
        return self.serializer.read("bdy_diff", self.savepoint)[0]

    def fac_bdydiff_v(self) -> int:
        return self.serializer.read("fac_bdydiff_v", self.savepoint)[0]

    def smag_offset(self):
        return self.serializer.read("smag_offset", self.savepoint)[0]

    def diff_multfac_w(self):
        return self.serializer.read("diff_multfac_w", self.savepoint)[0]

    def diff_multfac_vn(self):
        return self.serializer.read("diff_multfac_vn", self.savepoint)

    def rho(self):
        return self._get_field("rho", dims.CellDim, dims.KDim)

    def construct_prognostics(self) -> prognostic_state.PrognosticState:
        return prognostic_state.PrognosticState(
            w=self.w(),
            vn=self.vn(),
            exner=self.exner(),
            theta_v=self.theta_v(),
            rho=self.rho(),
        )


class IconDiffusionExitSavepoint(IconSavepoint):
    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def theta_v(self):
        return self._get_field("theta_v", dims.CellDim, dims.KDim)

    def w(self):
        return self._get_field("w", dims.CellDim, dims.KDim)

    def dwdx(self):
        return self._get_field("dwdx", dims.CellDim, dims.KDim)

    def dwdy(self):
        return self._get_field("dwdy", dims.CellDim, dims.KDim)

    def exner(self):
        return self._get_field("exner", dims.CellDim, dims.KDim)

    def div_ic(self):
        return self._get_field("div_ic", dims.CellDim, dims.KDim)

    def hdef_ic(self):
        return self._get_field("hdef_ic", dims.CellDim, dims.KDim)


class IconNonHydroInitSavepoint(IconSavepoint):
    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def bdy_divdamp(self):
        return self._get_field("bdy_divdamp", dims.KDim)

    def divdamp_fac_o2(self):
        return self.serializer.read("divdamp_fac_o2", self.savepoint).astype(float)[0]

    def ddt_exner_phy(self):
        return self._get_field("ddt_exner_phy", dims.CellDim, dims.KDim)

    def ddt_vn_phy(self):
        return self._get_field("ddt_vn_phy", dims.EdgeDim, dims.KDim)

    def exner_now(self):
        return self._get_field("exner_now", dims.CellDim, dims.KDim)

    def exner_new(self):
        return self._get_field("exner_new", dims.CellDim, dims.KDim)

    def theta_v_now(self):
        return self._get_field("theta_v_now", dims.CellDim, dims.KDim)

    def theta_v_new(self):
        return self._get_field("theta_v_new", dims.CellDim, dims.KDim)

    def rho_now(self):
        return self._get_field("rho_now", dims.CellDim, dims.KDim)

    def rho_new(self):
        return self._get_field("rho_new", dims.CellDim, dims.KDim)

    def exner_pr(self):
        return self._get_field("exner_pr", dims.CellDim, dims.KDim)

    def grf_tend_rho(self):
        return self._get_field("grf_tend_rho", dims.CellDim, dims.KDim)

    def grf_tend_thv(self):
        return self._get_field("grf_tend_thv", dims.CellDim, dims.KDim)

    def grf_tend_vn(self):
        return self._get_field("grf_tend_vn", dims.EdgeDim, dims.KDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", dims.CellDim, dims.KDim)

    def ddt_vn_apc_pc(self, ntnd):
        return self._get_field_component("ddt_vn_apc_pc", ntnd, (dims.EdgeDim, dims.KDim))

    def ddt_w_adv_pc(self, ntnd):
        return self._get_field_component("ddt_w_adv_pc", ntnd, (dims.CellDim, dims.KDim))

    def grf_tend_w(self):
        return self._get_field("grf_tend_w", dims.CellDim, dims.KDim)

    def mass_fl_e(self):
        return self._get_field("mass_fl_e", dims.EdgeDim, dims.KDim)

    def mass_flx_me(self):
        return self._get_field("mass_flx_me", dims.EdgeDim, dims.KDim)

    def mass_flx_ic(self):
        return self._get_field("mass_flx_ic", dims.CellDim, dims.KDim)

    def rho_ic(self):
        return self._get_field("rho_ic", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def rho_incr(self):
        return self._get_field("rho_incr", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def exner_incr(self):
        return self._get_field("exner_incr", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def vn_incr(self):
        return self._get_field("vn_incr", dims.EdgeDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def exner_dyn_incr(self):
        return self._get_field("exner_dyn_incr", dims.CellDim, dims.KDim)

    def scal_divdamp_o2(self) -> float:
        return self.serializer.read("scal_divdamp_o2", self.savepoint)[0]

    def scal_divdamp(self) -> fa.KField[float]:
        return self._get_field("scal_divdamp", dims.KDim)

    def theta_v_ic(self):
        return self._get_field("theta_v_ic", dims.CellDim, dims.KDim)

    def vn_traj(self):
        return self._get_field("vn_traj", dims.EdgeDim, dims.KDim)

    def z_dwdz_dd(self):
        return self._get_field("z_dwdz_dd", dims.CellDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)

    def z_theta_v_e(self):
        return self._get_field("z_theta_v_e", dims.EdgeDim, dims.KDim)

    def z_rho_e(self):
        return self._get_field("z_rho_e", dims.EdgeDim, dims.KDim)

    def z_gradh_exner(self):
        return self._get_field("z_gradh_exner", dims.EdgeDim, dims.KDim)

    def z_w_expl(self):
        return self._get_field("z_w_expl", dims.CellDim, dims.KDim)

    def z_rho_expl(self):
        return self._get_field("z_rho_expl", dims.CellDim, dims.KDim)

    def z_exner_expl(self):
        return self._get_field("z_exner_expl", dims.CellDim, dims.KDim)

    def z_alpha(self):
        return self._get_field("z_alpha", dims.CellDim, dims.KDim)

    def z_beta(self):
        return self._get_field("z_beta", dims.CellDim, dims.KDim)

    def z_contr_w_fl_l(self):
        return self._get_field("z_contr_w_fl_l", dims.CellDim, dims.KDim)

    def z_q(self):
        return self._get_field("z_q", dims.CellDim, dims.KDim)

    def wgt_nnow_rth(self) -> float:
        return self.serializer.read("wgt_nnow_rth", self.savepoint)[0]

    def wgt_nnew_rth(self) -> float:
        return self.serializer.read("wgt_nnew_rth", self.savepoint)[0]

    def wgt_nnow_vel(self) -> float:
        return self.serializer.read("wgt_nnow_vel", self.savepoint)[0]

    def wgt_nnew_vel(self) -> float:
        return self.serializer.read("wgt_nnew_vel", self.savepoint)[0]

    def w_now(self):
        return self._get_field("w_now", dims.CellDim, dims.KDim)

    def w_new(self):
        return self._get_field("w_new", dims.CellDim, dims.KDim)

    def vn_now(self):
        return self._get_field("vn_now", dims.EdgeDim, dims.KDim)

    def vn_new(self):
        return self._get_field("vn_new", dims.EdgeDim, dims.KDim)


class NonHydroInitEdgeDiagnosticsUpdateVnSavepoint(IconSavepoint):
    def rho_ic(self):
        return self._get_field("rho_ic", dims.CellDim, dims.KDim)

    def vn(self):
        return self._get_field("vn_now", dims.EdgeDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def z_rth_pr(self, ind: TwoIndex):
        return self._get_field_component("z_rth_pr", ind, (dims.CellDim, dims.KDim))

    def z_exner_ex_pr(self):
        return self._get_field("z_exner_ex_pr", dims.CellDim, dims.KDim)

    def z_dexner_dz_c(self, ntnd: TimeIndex):
        return self._get_field_component("z_dexner_dz_c", ntnd, (dims.CellDim, dims.KDim))

    def theta_v(self):
        return self._get_field("theta_v_now", dims.CellDim, dims.KDim)

    def theta_v_ic(self):
        return self._get_field("theta_v_ic", dims.CellDim, dims.KDim)

    def z_dwdz_dd(self):
        return self._get_field("z_dwdz_dd", dims.CellDim, dims.KDim)

    def ddt_vn_apc_ntl(self, ntnd):
        return self._get_field_component("ddt_vn_apc_pc", ntnd, (dims.EdgeDim, dims.KDim))

    def ddt_vn_phy(self):
        return self._get_field("ddt_vn_phy", dims.EdgeDim, dims.KDim)

    def vn_incr(self):
        return self._get_field("vn_now", dims.EdgeDim, dims.KDim)

    def bdy_divdamp(self):
        return self._get_field("bdy_divdamp", dims.KDim)

    def z_hydro_corr(self):
        return self._get_field("z_hydro_corr", dims.EdgeDim)

    def z_graddiv2_vn(self):
        return self._get_field("z_graddiv2_vn", dims.EdgeDim, dims.KDim)

    def scal_divdamp(self):
        return self._get_field("scal_divdamp", dims.KDim)

    def z_rho_e(self):
        return self._get_field("z_rho_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_e(self):
        return self._get_field("z_theta_v_e", dims.EdgeDim, dims.KDim)

    def z_gradh_exner(self):
        return self._get_field("z_gradh_exner", dims.EdgeDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)


class NonHydroInitVerticallyImplicitSolverSavepoint(IconSavepoint):
    def mass_fl_e(self):
        return self._get_field("mass_fl_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_fl_e(self):
        return self._get_field("z_theta_v_fl_e", dims.EdgeDim, dims.KDim)

    def z_flxdiv_mass(self):
        return self._get_field("z_flxdiv_mass", dims.CellDim, dims.KDim)

    def z_flxdiv_theta(self):
        return self._get_field("z_flxdiv_theta", dims.CellDim, dims.KDim)

    def z_w_expl(self):
        return self._get_field("z_w_expl", dims.CellDim, dims.KDim)

    def ddt_w_adv_pc(self, ntnd: TimeIndex):
        return self._get_field_component("ddt_w_adv_pc", ntnd, (dims.CellDim, dims.KDim))

    def z_th_ddz_exner_c(self):
        return self._get_field("z_th_ddz_exner_c", dims.CellDim, dims.KDim)

    def z_contr_w_fl_l(self):
        return self._get_field("z_contr_w_fl_l", dims.CellDim, dims.KDim)

    def rho_ic(self):
        return self._get_field("rho_ic", dims.CellDim, dims.KDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", dims.CellDim, dims.KDim)

    def exner_nnow(self):
        return self._get_field("exner_now", dims.CellDim, dims.KDim)

    def rho_nnow(self):
        return self._get_field("rho_now", dims.CellDim, dims.KDim)

    def theta_v_nnow(self):
        return self._get_field("theta_v_now", dims.CellDim, dims.KDim)

    def z_alpha(self):
        return self._get_field("z_alpha", dims.CellDim, dims.KDim)

    def z_beta(self):
        return self._get_field("z_beta", dims.CellDim, dims.KDim)

    def theta_v_ic(self):
        return self._get_field("theta_v_ic", dims.CellDim, dims.KDim)

    def z_q(self):
        return self._get_field("z_q", dims.CellDim, dims.KDim)

    def w(self):
        return self._get_field("w_now", dims.CellDim, dims.KDim)

    def z_rho_expl(self):
        return self._get_field("z_rho_expl", dims.CellDim, dims.KDim)

    def z_exner_expl(self):
        return self._get_field("z_exner_expl", dims.CellDim, dims.KDim)

    def exner_pr(self):
        return self._get_field("exner_pr", dims.CellDim, dims.KDim)

    def ddt_exner_phy(self):
        return self._get_field("ddt_exner_phy", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def rho_incr(self):
        return self._get_field("rho_now", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def exner_incr(self):
        return self._get_field("exner_now", dims.CellDim, dims.KDim)

    def z_raylfac(self):
        return self._get_field("z_raylfac", dims.KDim)

    def rho(self):
        return self._get_field("rho_now", dims.CellDim, dims.KDim)

    def exner(self):
        return self._get_field("exner_now", dims.CellDim, dims.KDim)

    def theta_v(self):
        return self._get_field("theta_v_now", dims.CellDim, dims.KDim)

    def z_dwdz_dd(self):
        return self._get_field("z_dwdz_dd", dims.CellDim, dims.KDim)

    @IconSavepoint.optionally_registered()
    def exner_dyn_incr(self):
        return self._get_field("exner_dyn_incr", dims.CellDim, dims.KDim)

    def mass_flx_ic(self):
        return self._get_field("mass_flx_ic", dims.CellDim, dims.KDim)

    def vol_flx_ic(self):
        return self._get_field("vol_flx_ic", dims.CellDim, dims.KDim)


class IconDycoreInit30To38Savepoint(IconSavepoint):
    def z_vn_avg(self):
        return self._get_field("z_vn_avg", dims.EdgeDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)

    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def z_rho_e(self):
        return self._get_field("z_rho_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_e(self):
        return self._get_field("z_theta_v_e", dims.EdgeDim, dims.KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def mass_fl_e(self):
        return self._get_field("mass_fl_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_fl_e(self):
        return self._get_field("z_theta_v_fl_e", dims.EdgeDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", dims.EdgeDim, dims.KDim)


class IconDycoreExit30To38Savepoint(IconSavepoint):
    def z_vn_avg(self):
        return self._get_field("z_vn_avg", dims.EdgeDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def mass_fl_e(self):
        return self._get_field("mass_fl_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_fl_e(self):
        return self._get_field("z_theta_v_fl_e", dims.EdgeDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", dims.EdgeDim, dims.KDim)

    def vn_traj(self):
        return self._get_field("vn_traj", dims.EdgeDim, dims.KDim)

    def mass_flx_me(self):
        return self._get_field("mass_flx_me", dims.EdgeDim, dims.KDim)


class IconNonHydroExitSavepoint(IconSavepoint):
    def z_exner_ex_pr(self):
        return self._get_field("z_exner_ex_pr", dims.CellDim, dims.KDim)  # KHalfDim

    def rho_ic(self):
        return self._get_field("rho_ic", dims.CellDim, dims.KDim)

    def z_rho_e(self):
        return self._get_field("z_rho_e", dims.EdgeDim, dims.KDim)

    def z_exner_expl(self):
        return self._get_field("z_exner_expl", dims.CellDim, dims.KDim)

    def z_rho_expl(self):
        return self._get_field("z_rho_expl", dims.CellDim, dims.KDim)

    def z_theta_v_e(self):
        return self._get_field("z_theta_v_e", dims.EdgeDim, dims.KDim)

    def theta_v_ic(self):
        return self._get_field("theta_v_ic", dims.CellDim, dims.KDim)

    def z_q(self):
        return self._get_field("z_q", dims.CellDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)

    def exner_pr(self):
        return self._get_field("exner_pr", dims.CellDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)

    def z_alpha(self):
        return self._get_field("z_alpha", dims.CellDim, dims.KDim)

    def z_beta(self):
        return self._get_field("z_beta", dims.CellDim, dims.KDim)

    def vn_new(self):
        return self._get_field("vn_new", dims.EdgeDim, dims.KDim)

    def theta_v_new(self):
        return self._get_field("theta_v_new", dims.CellDim, dims.KDim)

    def rho_new(self):
        return self._get_field("rho_new", dims.CellDim, dims.KDim)

    def exner_new(self):
        return self._get_field("exner_new", dims.CellDim, dims.KDim)

    def w_new(self):
        return self._get_field("w_new", dims.CellDim, dims.KDim)

    def z_vn_avg(self):
        return self._get_field("z_vn_avg", dims.EdgeDim, dims.KDim)

    def mass_fl_e(self):
        return self._get_field("mass_fl_e", dims.EdgeDim, dims.KDim)

    def mass_flx_ic(self):
        return self._get_field("mass_flx_ic", dims.CellDim, dims.KDim)

    def vol_flx_ic(self):
        return self._get_field("vol_flx_ic", dims.CellDim, dims.KDim)

    def mass_flx_me(self):
        return self._get_field("mass_flx_me", dims.EdgeDim, dims.KDim)

    def vn_traj(self):
        return self._get_field("vn_traj", dims.EdgeDim, dims.KDim)

    def exner_dyn_incr(self):
        return self._get_field("exner_dyn_incr", dims.CellDim, dims.KDim)

    def z_exner_ic(self):
        return self._get_field("z_exner_ic", dims.CellDim, dims.KDim)

    def z_dexner_dz_c(self, ntnd: TimeIndex):
        return self._get_field_component("z_dexner_dz_c", ntnd, (dims.CellDim, dims.KDim))

    def z_rth_pr(self, ind: TwoIndex):
        return self._get_field_component("z_rth_pr", ind, (dims.CellDim, dims.KDim))

    def z_grad_rth(self, ind: FourIndex):
        return self._get_field_component("z_grad_rth", ind, (dims.CellDim, dims.KDim))

    def z_th_ddz_exner_c(self):
        return self._get_field("z_th_ddz_exner_c", dims.CellDim, dims.KDim)

    def z_gradh_exner(self):
        return self._get_field("z_gradh_exner", dims.EdgeDim, dims.KDim)

    def z_hydro_corr(self):
        return self._get_field("z_hydro_corr", dims.EdgeDim, dims.KDim)

    def z_theta_v_pr_ic(self):
        return self._get_field("z_theta_v_pr_ic", dims.CellDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def z_flxdiv_mass(self):
        return self._get_field("z_flxdiv_mass", dims.CellDim, dims.KDim)

    def z_w_expl(self):
        return self._get_field("z_w_expl", dims.CellDim, dims.KDim)

    def z_flxdiv_theta(self):
        return self._get_field("z_flxdiv_theta", dims.CellDim, dims.KDim)

    def z_contr_w_fl_l(self):
        return self._get_field("z_contr_w_fl_l", dims.CellDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", dims.EdgeDim, dims.KDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", dims.CellDim, dims.KDim)

    def z_theta_v_fl_e(self):
        return self._get_field("z_theta_v_fl_e", dims.EdgeDim, dims.KDim)

    def z_dwdz_dd(self):
        return self._get_field("z_dwdz_dd", dims.CellDim, dims.KDim)


class NonHydroExitEdgeDiagnosticsUpdateVnSavepoint(IconSavepoint):
    def z_rho_e(self):
        return self._get_field("z_rho_e", dims.EdgeDim, dims.KDim)

    def z_theta_v_e(self):
        return self._get_field("z_theta_v_e", dims.EdgeDim, dims.KDim)

    def z_gradh_exner(self):
        return self._get_field("z_gradd_exner", dims.EdgeDim, dims.KDim)

    def vn(self):
        return self._get_field("vn_new", dims.EdgeDim, dims.KDim)

    def z_graddiv_vn(self):
        return self._get_field("z_graddiv_vn", dims.EdgeDim, dims.KDim)

    def z_graddiv2_vn(self):
        return self._get_field("z_graddiv2_vn", dims.EdgeDim, dims.KDim)


# TODO(halungge): rename?
class IconNonHydroFinalSavepoint(IconSavepoint):
    def theta_v_new(self):
        return self._get_field("theta_v", dims.CellDim, dims.KDim)

    def exner_new(self):
        return self._get_field("exner", dims.CellDim, dims.KDim)


class IconVelocityInitSavepoint(IconSavepoint):
    def cfl_w_limit(self) -> float:
        return self.serializer.read("cfl_w_limit", self.savepoint)[0]

    def vn_only(self) -> bool:
        return bool(self.serializer.read("vn_only", self.savepoint)[0])

    def max_vcfl_dyn(self):
        return self.serializer.read("max_vcfl_dyn", self.savepoint)[0]

    def scalfac_exdiff(self) -> float:
        return self.serializer.read("scalfac_exdiff", self.savepoint)[0]

    def ddt_vn_apc_pc(self, ntnd: TimeIndex):
        return self._get_field_component("ddt_vn_apc_pc", ntnd, (dims.EdgeDim, dims.KDim))

    def ddt_w_adv_pc(self, ntnd: TimeIndex):
        return self._get_field_component("ddt_w_adv_pc", ntnd, (dims.CellDim, dims.KDim))

    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def w(self):
        return self._get_field("w", dims.CellDim, dims.KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", dims.EdgeDim, dims.KDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", dims.CellDim, dims.KDim)

    def lvn_only(self) -> bool:
        return bool(self.serializer.read("vn_only", self.savepoint)[0])

    def z_w_con_c_full(self):
        return self._get_field("z_w_con_c_full", dims.CellDim, dims.KDim)


class IconVelocityExitSavepoint(IconSavepoint):
    def max_vcfl_dyn(self):
        return self.serializer.read("max_vcfl_dyn", self.savepoint)[0]

    def ddt_vn_apc_pc(self, ntnd: TimeIndex):
        return self._get_field_component("ddt_vn_apc_pc", ntnd, (dims.EdgeDim, dims.KDim))

    def ddt_w_adv_pc(self, ntnd: TimeIndex):
        return self._get_field_component("ddt_w_adv_pc", ntnd, (dims.CellDim, dims.KDim))

    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def w(self):
        return self._get_field("w", dims.CellDim, dims.KDim)

    def vt(self):
        return self._get_field("vt", dims.EdgeDim, dims.KDim)

    def vn_ie(self):
        return self._get_field("vn_ie", dims.EdgeDim, dims.KDim)

    def w_concorr_c(self):
        return self._get_field("w_concorr_c", dims.CellDim, dims.KDim)

    def z_vt_ie(self):
        return self._get_field("z_vt_ie", dims.EdgeDim, dims.KDim)

    def z_w_concorr_me(self):
        return self._get_field("z_w_concorr_me", dims.EdgeDim, dims.KDim)

    def z_w_concorr_mc(self):
        return self._get_field("z_w_concorr_mc", dims.CellDim, dims.KDim)

    def z_v_grad_w(self):
        return self._get_field("z_v_grad_w", dims.EdgeDim, dims.KDim)

    def z_w_con_c(self):
        return self._get_field("z_w_con_c", dims.CellDim, dims.KDim)  # KhalfDim

    def z_w_con_c_full(self):
        return self._get_field("z_w_con_c_full", dims.CellDim, dims.KDim)

    def z_ekinh(self):
        return self._get_field("z_ekinh", dims.CellDim, dims.KDim)

    def cfl_clipping(self):
        return self._get_field("cfl_clipping", dims.CellDim, dims.KDim, dtype=bool)

    def vcfl_dsl(self):
        return self._get_field("vcfl_dsl", dims.CellDim, dims.KDim)

    def z_kin_hor_e(self):
        return self._get_field("z_kin_hor_e", dims.EdgeDim, dims.KDim)


class IconJabwExitSavepoint(IconSavepoint):
    def exner(self):
        return self._get_field("exner", dims.CellDim, dims.KDim)

    def rho(self):
        return self._get_field("rho", dims.CellDim, dims.KDim)

    def vn(self):
        return self._get_field("vn", dims.EdgeDim, dims.KDim)

    def w(self):
        return self._get_field("w", dims.CellDim, dims.KDim)

    def theta_v(self):
        return self._get_field("theta_v", dims.CellDim, dims.KDim)

    def pressure(self):
        return self._get_field("pressure", dims.CellDim, dims.KDim)

    def temperature(self):
        return self._get_field("temperature", dims.CellDim, dims.KDim)

    # TODO(): change field name
    def pressure_sfc(self):
        return self._get_field("surface_pressure", dims.CellDim)


class IconDiagnosticsInitSavepoint(IconSavepoint):
    def pressure(self):
        return self._get_field("pressure", dims.CellDim, dims.KDim)

    def temperature(self):
        return self._get_field("temperature", dims.CellDim, dims.KDim)

    def exner_pr(self):
        return self._get_field("exner_pr", dims.CellDim, dims.KDim)

    def pressure_ifc(self):
        return self._get_field("pressure_ifc", dims.CellDim, dims.KDim)

    def pressure_sfc(self):
        return self._get_field("pressure_sfc", dims.CellDim)

    def virtual_temperature(self):
        return self._get_field("virtual_temperature", dims.CellDim, dims.KDim)

    def zonal_wind(self):
        return self._get_field("u", dims.CellDim, dims.KDim)

    def meridional_wind(self):
        return self._get_field("v", dims.CellDim, dims.KDim)


class IconPrognosticsInitSavepoint(IconSavepoint):
    def exner_now(self):
        return self._get_field("exner_now", dims.CellDim, dims.KDim)

    def rho_now(self):
        return self._get_field("rho_now", dims.CellDim, dims.KDim)

    def vn_now(self):
        return self._get_field("vn_now", dims.EdgeDim, dims.KDim)

    def w_now(self):
        return self._get_field("w_now", dims.CellDim, dims.KDim)

    def theta_v_now(self):
        return self._get_field("theta_v_now", dims.CellDim, dims.KDim)

    def tracer_now(self, ntracer: TracerIndex):
        return self._get_field_component("tracers_now", ntracer, (dims.CellDim, dims.KDim))


class IconGraupelSavepoint(IconSavepoint):
    def temperature(self):
        return self._get_field("temperature", dims.CellDim, dims.KDim)

    def pressure(self):
        return self._get_field("pressure", dims.CellDim, dims.KDim)

    def rho(self):
        return self._get_field("rho", dims.CellDim, dims.KDim)

    def tracer(self, ntracer: TracerIndex):
        return self._get_field_component("tracers", ntracer, (dims.CellDim, dims.KDim))

    def ddt_tend_t(self):
        return self._get_field("ddt_tend_t", dims.CellDim, dims.KDim)

    def ddt_tend_qv(self):
        return self._get_field("ddt_tend_qv", dims.CellDim, dims.KDim)

    def ddt_tend_qc(self):
        return self._get_field("ddt_tend_qc", dims.CellDim, dims.KDim)

    def ddt_tend_qi(self):
        return self._get_field("ddt_tend_qi", dims.CellDim, dims.KDim)

    def ddt_tend_qr(self):
        return self._get_field("ddt_tend_qr", dims.CellDim, dims.KDim)

    def ddt_tend_qs(self):
        return self._get_field("ddt_tend_qs", dims.CellDim, dims.KDim)

    def rain_flux(self):
        return self._get_field("rain_gsp_rate", dims.CellDim)

    def snow_flux(self):
        return self._get_field("snow_gsp_rate", dims.CellDim)

    def graupel_flux(self):
        return self._get_field("graupel_gsp_rate", dims.CellDim)

    def ice_flux(self):
        return self._get_field("ice_gsp_rate", dims.CellDim)

    def qv(self):
        return self.tracer(QV)

    def qc(self):
        return self.tracer(QC)

    def qi(self):
        return self.tracer(QI)

    def qr(self):
        return self.tracer(QR)

    def qs(self):
        return self.tracer(QS)

    def qg(self):
        return self.tracer(QG)

    def qnc(self):
        return self._get_field("qnc", dims.CellDim)

    def dtime(self):
        return self.serializer.read("dtime", self.savepoint)[0]


class IconSatadExitSavepoint(IconSavepoint):
    def temperature(self):
        return self._get_field("temperature", dims.CellDim, dims.KDim)

    def tracer(self, ntracer: TracerIndex):
        return self._get_field_component("tracers", ntracer, (dims.CellDim, dims.KDim))

    def qv(self):
        return self.tracer(QV)

    def qc(self):
        return self.tracer(QC)

    def qi(self):
        return self.tracer(QI)

    def qr(self):
        return self.tracer(QR)

    def qs(self):
        return self.tracer(QS)

    def qg(self):
        return self.tracer(QG)

    def exner(self):
        return self._get_field("exner", dims.CellDim, dims.KDim)

    def virtual_temperature(self):
        return self._get_field("virtual_temperature", dims.CellDim, dims.KDim)

    def pressure(self):
        return self._get_field("pressure", dims.CellDim, dims.KDim)

    def pressure_ifc(self):
        return self._get_field("pressure_ifc", dims.CellDim, dims.KDim)

    def pressure_sfc(self):
        return self._get_field("pressure_sfc", dims.CellDim)


class IconSatadInitSavepoint(IconSatadExitSavepoint):
    def rho(self):
        return self._get_field("rho", dims.CellDim, dims.KDim)


class TopographySavepoint(IconSavepoint):
    def topo_c(self):
        return self._get_field("topography", dims.CellDim)

    def topo_smt_c(self):
        return self._get_field("smooth_topography", dims.CellDim)


class IconSerialDataProvider:
    def __init__(
        self,
        *,
        backend: gtx_typing.Backend | None,
        fname_prefix,
        path=".",
        do_print=False,
        mpi_rank=0,
    ):
        self.rank = mpi_rank
        self.serializer: serialbox.Serializer = None
        self.file_path: str = path
        self.fname = f"{fname_prefix}_rank{self.rank!s}"
        self.log = logging.getLogger(__name__)
        self._init_serializer(do_print)
        self.backend = backend

    def _init_serializer(self, do_print: bool):
        if not self.fname:
            self.log.warning(" WARNING: no filename! closing serializer")
        self.serializer = serialbox.Serializer(
            serialbox.OpenModeKind.Read, self.file_path, self.fname
        )
        if do_print:
            self.print_info()

    def print_info(self):
        self.log.info(f"SAVEPOINTS: {self.serializer.savepoint_list()}")
        self.log.info(f"FIELDNAMES: {self.serializer.fieldnames()}")

    @functools.cached_property
    def grid_size(self):
        sp = self._get_icon_grid_savepoint()
        grid_sizes = {
            dims.CellDim: self.serializer.read("num_cells", savepoint=sp).astype(gtx.int32)[0],
            dims.EdgeDim: self.serializer.read("num_edges", savepoint=sp).astype(gtx.int32)[0],
            dims.VertexDim: self.serializer.read("num_vert", savepoint=sp).astype(gtx.int32)[0],
            dims.KDim: sp.metainfo.to_dict()["nlev"],
        }
        return grid_sizes

    def from_savepoint_grid(self, grid_id: str, grid_params: icon.GridParams) -> IconGridSavepoint:
        savepoint = self._get_icon_grid_savepoint()
        return IconGridSavepoint(
            sp=savepoint,
            ser=self.serializer,
            grid_id=grid_id,
            size=self.grid_size,
            grid_params=grid_params,
            backend=self.backend,
        )

    def _get_icon_grid_savepoint(self):
        savepoint = self.serializer.savepoint["icon-grid"].id[1].as_savepoint()
        return savepoint

    def from_savepoint_diffusion_init(
        self,
        linit: bool,
        date: str,
    ) -> IconDiffusionInitSavepoint:
        savepoint = (
            self.serializer.savepoint["diffusion-init"].linit[linit].date[date].as_savepoint()
        )
        return IconDiffusionInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_velocity_init(
        self, istep: int, date: str, substep: int
    ) -> IconVelocityInitSavepoint:
        savepoint = (
            self.serializer.savepoint["velocity-tendencies-init"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconVelocityInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_nonhydro_init(
        self, istep: int, date: str, substep: int
    ) -> IconNonHydroInitSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-init"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconNonHydroInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init(
        self, istep: int, date: str, substep: int
    ) -> NonHydroInitEdgeDiagnosticsUpdateVnSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-14to28-init_1to13-exit"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return NonHydroInitEdgeDiagnosticsUpdateVnSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_vertically_implicit_dycore_solver_init(
        self, istep: int, date: str, substep: int
    ) -> NonHydroInitVerticallyImplicitSolverSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-41to60-init"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return NonHydroInitVerticallyImplicitSolverSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_30_to_38_init(
        self, istep: int, date: str, substep: int
    ) -> IconDycoreInit30To38Savepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-30to38-init"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconDycoreInit30To38Savepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_interpolation_savepoint(self) -> InterpolationSavepoint:
        savepoint = self.serializer.savepoint["interpolation-state"].as_savepoint()
        return InterpolationSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_metrics_savepoint(self) -> MetricSavepoint:
        savepoint = self.serializer.savepoint["metric-state"].as_savepoint()
        return MetricSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_topography_savepoint(self) -> TopographySavepoint:
        savepoint = self.serializer.savepoint["smooth-topo-savepoint"].as_savepoint()
        return TopographySavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_advection_init_savepoint(self, size: dict, date: str) -> AdvectionInitSavepoint:
        savepoint = self.serializer.savepoint["advection-init"].id[1].date[date].as_savepoint()
        return AdvectionInitSavepoint(savepoint, self.serializer, size=size, backend=self.backend)

    def from_advection_exit_savepoint(self, size: dict, date: str) -> AdvectionExitSavepoint:
        savepoint = self.serializer.savepoint["advection-exit"].id[1].date[date].as_savepoint()
        return AdvectionExitSavepoint(savepoint, self.serializer, size=size, backend=self.backend)

    def from_savepoint_diffusion_exit(self, linit: bool, date: str) -> IconDiffusionExitSavepoint:
        savepoint = (
            self.serializer.savepoint["diffusion-exit"].linit[linit].date[date].as_savepoint()
        )
        return IconDiffusionExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_velocity_exit(
        self, istep: int, date: str, substep: int
    ) -> IconVelocityExitSavepoint:
        savepoint = (
            self.serializer.savepoint["velocity-tendencies-exit"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconVelocityExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_30_to_38_exit(
        self, istep: int, date: str, substep: int
    ) -> IconDycoreExit30To38Savepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-30to38-exit"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconDycoreExit30To38Savepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_nonhydro_exit(
        self, istep: int, date: str, substep: int
    ) -> IconNonHydroExitSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-exit"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconNonHydroExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit(
        self, istep: int, date: str, substep: int
    ) -> NonHydroExitEdgeDiagnosticsUpdateVnSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-14to28-exit"]
            .istep[istep]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return NonHydroExitEdgeDiagnosticsUpdateVnSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_nonhydro_step_final(
        self, date: str, substep: int
    ) -> IconNonHydroFinalSavepoint:
        savepoint = (
            self.serializer.savepoint["solve-nonhydro-final"]
            .date[date]
            .dyn_timestep[substep]
            .as_savepoint()
        )
        return IconNonHydroFinalSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_jabw_exit(self) -> IconJabwExitSavepoint:
        savepoint = self.serializer.savepoint["jabw-initial-state-exit"].id[1].as_savepoint()
        return IconJabwExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_prognostics_initial(self) -> IconPrognosticsInitSavepoint:
        savepoint = (
            self.serializer.savepoint["prognostics"].id[1].location["initial-state"].as_savepoint()
        )
        return IconPrognosticsInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_diagnostics_initial(self) -> IconDiagnosticsInitSavepoint:
        savepoint = (
            self.serializer.savepoint["diagnostics"].id[1].location["initial-state"].as_savepoint()
        )
        return IconDiagnosticsInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_weisman_klemp_graupel_entry(self, date: str) -> IconGraupelSavepoint:
        savepoint = self.serializer.savepoint["microphysics-init"].date[date].as_savepoint()
        return IconGraupelSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_weisman_klemp_graupel_exit(self, date: str) -> IconGraupelSavepoint:
        savepoint = self.serializer.savepoint["microphysics-exit"].date[date].as_savepoint()
        return IconGraupelSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_satad_init(self, location: str, date: str) -> IconSatadInitSavepoint:
        savepoint = (
            self.serializer.savepoint["satad-init"].location[location].date[date].as_savepoint()
        )
        return IconSatadInitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )

    def from_savepoint_satad_exit(self, location: str, date: str) -> IconSatadExitSavepoint:
        savepoint = (
            self.serializer.savepoint["satad-exit"].date[date].location[location].as_savepoint()
        )
        return IconSatadExitSavepoint(
            savepoint, self.serializer, size=self.grid_size, backend=self.backend
        )
