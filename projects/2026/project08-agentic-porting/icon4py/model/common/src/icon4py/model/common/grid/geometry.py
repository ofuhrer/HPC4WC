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
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import gt4py.next.typing as gtx_typing
import numpy as np
from gt4py import next as gtx

from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    geometry_attributes as attrs,
    geometry_config,
    geometry_stencils as stencils,
    grid_manager as gm,
    gridfile,
    horizontal as h_grid,
    icon,
)
from icon4py.model.common.math import coordinate_transformations as coord_trans, utils as math_utils
from icon4py.model.common.states import factory, model, utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils


log = logging.getLogger(__name__)


class GridGeometry(factory.FieldSource):
    """
    Factory for the ICON grid geometry fields.

    Computes geometry fields from the grid geographical coordinates fo cells, edges, vertices.
    Computations are triggered upon first request.

    Can be queried for geometry fields and metadata

    Examples:
        >>> geometry = GridGeometry(
        ...     grid=grid,
        ...     decomposition_info=decomposition_info,
        ...     backend=backend,
        ...     coordinates=coordinates,
        ...     extra_fields=extra_fields,
        ...     metadata=geometry_attributes.attrs,
        ...     config=geometry_config.GeometryConfig(),
        ...     process_props=process_props,
        ...     exchange=exchange,
        ...     global_reductions=global_reductions,
        ... )
        GridGeometry for geometry_type=SPHERE grid=f2e06839-694a-cca1-a3d5-028e0ff326e0 : R9B4
        >>> geometry.get("edge_length")
        NumPyArrayField(_domain=Domain(dims=(Dimension(value='Edge', kind=<DimensionKind.HORIZONTAL: 'horizontal'>),), ranges=(UnitRange(0, 31558),)), _ndarray=array([3746.2669054 , 3746.2669066 , 3746.33418138, ..., 3736.61622936, 3792.41317057]))
        >>> geometry.get("edge_length", RetrievalType.METADATA)
        {'standard_name': 'edge_length',
        'long_name': 'edge length',
        'units': 'm',
        'dims': (Dimension(value='Edge', kind=<DimensionKind.HORIZONTAL: 'horizontal'>),),
        'icon_var_name': 't_grid_edges%primal_edge_length',
        'dtype': numpy.float64}
        >>> geometry.get("edge_length", RetrievalType.DATA_ARRAY)
        <xarray.DataArray (dim_0: 31558)> Size: 252kB
        array([3746.2669054 , 3746.2669066 , 3746.33418138, ..., 3889.53098062, 3736.61622936, 3792.41317057])
        Dimensions without coordinates: dim_0
        .Attributes:
        standard_name:  edge_length
        long_name:      edge length
        units:          m
        dims:           (Dimension(value='Edge', kind=<DimensionKind.HORIZONTAL: ...
        icon_var_name:  t_grid_edges%primal_edge_length
        dtype:          <class 'numpy.float64'>


    """

    def __init__(
        self,
        *,
        grid: icon.IconGrid,
        decomposition_info: decomposition.DecompositionInfo,
        backend: gtx_typing.Backend | None,
        coordinates: gm.CoordinateDict,
        extra_fields: gm.GeometryDict,
        metadata: dict[str, model.FieldMetaData],
        config: geometry_config.GeometryConfig,
        process_props: decomposition.ProcessProperties,
        # TODO(msimberg): There's no need to pass exchange and global_reductions
        # if process_props is passed. The former can all be constructed from
        # process_props. Refactor this consistently across the code base to use
        # process_props only. We may need special care to make sure that we
        # don't create many different GHEX communication objects.
        exchange: decomposition.ExchangeRuntime,
        global_reductions: decomposition.Reductions = decomposition.single_node_reductions,
    ) -> None:
        """
        Args:
            grid: IconGrid the grid topology
            decomposition_info: data structure containing owner masks for field dimensions
            backend: backend used for memory allocation and computation
            coordinates: dictionary containing geographical coordinates for grid cells, edges and vertices,
            extra_fields: fields that are not computed but directly read off the grid file,
                currently only the edge_system_orientation cell_area. Should eventually disappear.
            metadata: a dictionary of FieldMetaData for all fields computed in GridGeometry.
            config: configuration options controlling geometry computation.
            process_props: process properties including the MPI communicator.

        """
        self._providers = {}
        self._backend = backend
        self._xp = data_alloc.import_array_ns(backend)
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)
        self._grid = grid
        self._decomposition_info = decomposition_info
        self._attrs = metadata
        self._geometry_type: icon.GeometryType = grid.grid_params.geometry_type
        self._edge_domain = h_grid.domain(dims.EdgeDim)
        self._config = config
        self._exchange = exchange
        self._process_props = process_props
        self._global_reductions = global_reductions
        log.info(
            f"initializing geometry for backend = '{self._backend_name()}' and grid = '{self._grid}'"
        )

        # Setup coordinates based on geometry type
        coordinates_ = {
            attrs.CELL_LON: coordinates[dims.CellDim]["lon"],
            attrs.CELL_LAT: coordinates[dims.CellDim]["lat"],
            attrs.EDGE_LON: coordinates[dims.EdgeDim]["lon"],
            attrs.EDGE_LAT: coordinates[dims.EdgeDim]["lat"],
            attrs.VERTEX_LON: coordinates[dims.VertexDim]["lon"],
            attrs.VERTEX_LAT: coordinates[dims.VertexDim]["lat"],
        }
        if self._geometry_type == icon.GeometryType.TORUS:
            coordinates_[attrs.CELL_CENTER_X] = coordinates[dims.CellDim]["x"]
            coordinates_[attrs.CELL_CENTER_Y] = coordinates[dims.CellDim]["y"]
            coordinates_[attrs.CELL_CENTER_Z] = coordinates[dims.CellDim]["z"]
            coordinates_[attrs.EDGE_CENTER_X] = coordinates[dims.EdgeDim]["x"]
            coordinates_[attrs.EDGE_CENTER_Y] = coordinates[dims.EdgeDim]["y"]
            coordinates_[attrs.EDGE_CENTER_Z] = coordinates[dims.EdgeDim]["z"]
            coordinates_[attrs.VERTEX_X] = coordinates[dims.VertexDim]["x"]
            coordinates_[attrs.VERTEX_Y] = coordinates[dims.VertexDim]["y"]
            coordinates_[attrs.VERTEX_Z] = coordinates[dims.VertexDim]["z"]

        coordinate_provider = factory.PrecomputedFieldProvider(fields=coordinates_)
        self.register_provider(coordinate_provider)

        # Setup input fields
        input_fields_provider = factory.PrecomputedFieldProvider(
            fields={
                # TODO(halungge): rescaled by grid_length_rescale_factor (mo_grid_tools.f90)
                attrs.EDGE_LENGTH: extra_fields[gridfile.GeometryName.EDGE_LENGTH],
                attrs.DUAL_EDGE_LENGTH: extra_fields[gridfile.GeometryName.DUAL_EDGE_LENGTH],
                attrs.EDGE_CELL_DISTANCE: extra_fields[gridfile.GeometryName.EDGE_CELL_DISTANCE],
                attrs.EDGE_VERTEX_DISTANCE: extra_fields[
                    gridfile.GeometryName.EDGE_VERTEX_DISTANCE
                ],
                attrs.CELL_AREA: extra_fields[gridfile.GeometryName.CELL_AREA],
                attrs.DUAL_AREA: extra_fields[gridfile.GeometryName.DUAL_AREA],
                attrs.TANGENT_ORIENTATION: extra_fields[gridfile.GeometryName.TANGENT_ORIENTATION],
                attrs.CELL_NORMAL_ORIENTATION: extra_fields[
                    gridfile.GeometryName.CELL_NORMAL_ORIENTATION
                ],
                attrs.VERTEX_EDGE_ORIENTATION: extra_fields[
                    gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX
                ],
                "edge_owner_mask": gtx.as_field(
                    (dims.EdgeDim,),
                    decomposition_info.owner_mask(dims.EdgeDim),
                    dtype=bool,
                    allocator=self._backend,
                ),
                "vertex_owner_mask": gtx.as_field(
                    (dims.VertexDim,),
                    decomposition_info.owner_mask(dims.VertexDim),
                    allocator=self._backend,
                    dtype=bool,
                ),
                "cell_owner_mask": gtx.as_field(
                    (dims.CellDim,),
                    decomposition_info.owner_mask(dims.CellDim),
                    allocator=self._backend,
                    dtype=bool,
                ),
            }
        )
        self.register_provider(input_fields_provider)
        self._register_computed_fields()

    def _compute_analytical_means(self) -> dict[str, float]:
        """Compute mean geometry values analytically from grid parameters.

        For regular grids (global icosahedron and torus) the mean cell area,
        edge length and their dual counterparts can be computed directly from
        the grid parameters, avoiding non-deterministic global reductions.

        These values are computed from the *global* grid counts, so they are
        identical regardless of whether the grid is full-sphere or a
        limited-area cut from the same global grid, and regardless of whether
        the run is single- or multi-rank.

        For the torus all triangles are assumed equilateral and identical.

        Returns:
            A dictionary of scalar mean values.
        """
        grid_params = self._grid.grid_params

        match self._geometry_type:
            case icon.GeometryType.ICOSAHEDRON:
                radius = grid_params.radius
                subdivision = grid_params.subdivision
                root = subdivision.root
                level = subdivision.level
                num_cells = 20 * root**2 * 4**level
                num_vertices = num_cells // 2 + 2
                mean_cell_area = 4.0 * math.pi * radius**2 / num_cells
                mean_dual_area = 4.0 * math.pi * radius**2 / num_vertices
                mean_edge_length = math.sqrt(4.0 * mean_cell_area / math.sqrt(3.0))
                mean_dual_edge_length = mean_edge_length / math.sqrt(3.0)
            case icon.GeometryType.TORUS:
                # For a uniform torus grid all cells are identical equilateral
                # triangles. Read the common edge length directly from the edge
                # length field (the grid file stores it on every edge).
                # TODO(msimberg): Check if we can/should get it from the grid
                # file directly instead (e.g. via
                # MPIMPropertyName.MEAN_EDGE_LENGTH).
                edge_length = self.get(attrs.EDGE_LENGTH).ndarray
                if self._process_props.comm is not None:
                    assert edge_length.size > 0
                    send_buffer = np.empty(1, dtype=edge_length.dtype)
                    send_buffer[0] = edge_length[0]
                    self._process_props.comm.Bcast(send_buffer, root=0)
                    mean_edge_length = float(send_buffer[0])
                else:
                    mean_edge_length = float(edge_length[0])
                mean_cell_area = mean_edge_length**2 * math.sqrt(3.0) / 4.0
                mean_dual_area = 2.0 * mean_cell_area
                mean_dual_edge_length = mean_edge_length / math.sqrt(3.0)
            case _:
                raise ValueError(f"Invalid geometry type {self._geometry_type}")

        return {
            attrs.MEAN_CELL_AREA: mean_cell_area,
            attrs.MEAN_DUAL_AREA: mean_dual_area,
            attrs.MEAN_EDGE_LENGTH: mean_edge_length,
            attrs.MEAN_DUAL_EDGE_LENGTH: mean_dual_edge_length,
            attrs.CHARACTERISTIC_LENGTH: math.sqrt(mean_cell_area),
        }

    def _inverse_field_provider(self, field_name: str) -> factory.FieldProvider:
        meta = attrs.metadata_for_inverse(attrs.attrs[field_name])
        name = meta["standard_name"]
        self._attrs.update({name: meta})
        provider = factory.ProgramFieldProvider(
            func=math_utils.compute_inverse_on_edges,
            deps={"f": field_name},
            fields={"f_inverse": name},
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
            do_exchange=True,
        )
        return provider

    def _register_computed_fields(self) -> None:
        """Register all computed geometry fields."""
        # Common fields for both geometries
        meta = attrs.metadata_for_inverse(attrs.attrs[attrs.EDGE_LENGTH])
        name = meta["standard_name"]
        self._attrs.update({name: meta})

        inverse_edge_length = self._inverse_field_provider(attrs.EDGE_LENGTH)
        self.register_provider(inverse_edge_length)

        inverse_dual_edge_length = self._inverse_field_provider(attrs.DUAL_EDGE_LENGTH)
        self.register_provider(inverse_dual_edge_length)

        match self._geometry_type:
            case icon.GeometryType.ICOSAHEDRON:
                self._register_cartesian_coordinates_icosahedron()

                vertex_vertex_distance = factory.ProgramFieldProvider(
                    func=stencils.compute_arc_distance_of_far_edges_in_diamond,
                    domain={
                        dims.EdgeDim: (
                            self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                            self._edge_domain(h_grid.Zone.LOCAL),
                        )
                    },
                    fields={"far_vertex_distance": attrs.VERTEX_VERTEX_LENGTH},
                    deps={
                        "vertex_lat": attrs.VERTEX_LAT,
                        "vertex_lon": attrs.VERTEX_LON,
                    },
                    params={"radius": self._grid.grid_params.radius},
                    do_exchange=True,
                )
                self.register_provider(vertex_vertex_distance)

                coriolis_param = factory.ProgramFieldProvider(
                    func=stencils.compute_coriolis_parameter_on_edges,
                    deps={"edge_center_lat": attrs.EDGE_LAT},
                    params={"angular_velocity": constants.EARTH_ANGULAR_VELOCITY},
                    fields={"coriolis_parameter": attrs.CORIOLIS_PARAMETER},
                    domain={
                        dims.EdgeDim: (
                            self._edge_domain(h_grid.Zone.LOCAL),
                            self._edge_domain(h_grid.Zone.END),
                        )
                    },
                    do_exchange=False,
                )
                self.register_provider(coriolis_param)

                self._register_normals_and_tangents_icosahedron()

            case icon.GeometryType.TORUS:
                vertex_vertex_distance = factory.ProgramFieldProvider(
                    func=stencils.compute_distance_of_far_edges_in_diamond_torus,
                    domain={
                        dims.EdgeDim: (
                            self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                            self._edge_domain(h_grid.Zone.LOCAL),
                        )
                    },
                    fields={"far_vertex_distance": attrs.VERTEX_VERTEX_LENGTH},
                    deps={
                        "vertex_x": attrs.VERTEX_X,
                        "vertex_y": attrs.VERTEX_Y,
                    },
                    params={
                        "domain_length": self._grid.grid_params.domain_length,
                        "domain_height": self._grid.grid_params.domain_height,
                    },
                    do_exchange=True,
                )
                self.register_provider(vertex_vertex_distance)

                coriolis_param = factory.PrecomputedFieldProvider(
                    fields={
                        # TODO(jcanton): this constant (0.0) should eventually
                        # come from the config
                        "coriolis_parameter": stencils.coriolis_parameter_on_edges_torus(
                            coriolis_coefficient=0.0,
                            num_edges=self._grid.num_edges,
                            backend=self._backend,
                        )
                    }
                )
                self.register_provider(coriolis_param)

                self._register_normals_and_tangents_torus()

        # Inverse of vertex-vertex distance
        inverse_far_edge_distance_provider = self._inverse_field_provider(
            attrs.VERTEX_VERTEX_LENGTH
        )
        self.register_provider(inverse_far_edge_distance_provider)

        # Edge areas
        edge_areas = factory.ProgramFieldProvider(
            func=stencils.compute_edge_area,
            deps={
                "owner_mask": "edge_owner_mask",
                "primal_edge_length": attrs.EDGE_LENGTH,
                "dual_edge_length": attrs.DUAL_EDGE_LENGTH,
            },
            fields={"area": attrs.EDGE_AREA},
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
            do_exchange=True,
        )
        self.register_provider(edge_areas)

        if self._config.use_analytical_means:
            analytical_means = self._compute_analytical_means()
            mean_provider = factory.PrecomputedFieldProvider(analytical_means)
            self.register_provider(mean_provider)
        else:
            mean_edge_length_np = factory.NumpyDataProvider(
                func=self._global_reductions.mean,
                domain=(),
                deps={
                    "buffer": attrs.EDGE_LENGTH,
                },
                fields=(attrs.MEAN_EDGE_LENGTH,),
            )
            self.register_provider(mean_edge_length_np)

            mean_dual_edge_length_np = factory.NumpyDataProvider(
                func=self._global_reductions.mean,
                domain=(),
                deps={
                    "buffer": attrs.DUAL_EDGE_LENGTH,
                },
                fields=(attrs.MEAN_DUAL_EDGE_LENGTH,),
            )
            self.register_provider(mean_dual_edge_length_np)

            mean_cell_area_np = factory.NumpyDataProvider(
                func=self._global_reductions.mean,
                domain=(),
                deps={
                    "buffer": attrs.CELL_AREA,
                },
                fields=(attrs.MEAN_CELL_AREA,),
            )
            self.register_provider(mean_cell_area_np)

            mean_dual_cell_area_np = factory.NumpyDataProvider(
                func=self._global_reductions.mean,
                domain=(),
                deps={
                    "buffer": attrs.DUAL_AREA,
                },
                fields=(attrs.MEAN_DUAL_AREA,),
            )
            self.register_provider(mean_dual_cell_area_np)

            characteristic_length_np = factory.NumpyDataProvider(
                func=math_utils.compute_sqrt,
                domain=(),
                deps={
                    "input_val": attrs.MEAN_CELL_AREA,
                },
                fields=(attrs.CHARACTERISTIC_LENGTH,),
            )
            self.register_provider(characteristic_length_np)

    def _register_normals_and_tangents_icosahedron(self) -> None:
        """Register normals and tangents specific to icosahedron geometry."""
        # 1. edges%primal_cart_normal (cartesian coordinates for primal_normal)
        tangent_normal_coordinates = factory.ProgramFieldProvider(
            func=stencils.compute_cartesian_coordinates_of_edge_tangent_and_normal,
            deps={
                "vertex_lat": attrs.VERTEX_LAT,
                "vertex_lon": attrs.VERTEX_LON,
                "edge_lat": attrs.EDGE_LAT,
                "edge_lon": attrs.EDGE_LON,
                "edge_orientation": attrs.TANGENT_ORIENTATION,
            },
            fields={
                "tangent_x": attrs.EDGE_TANGENT_X,
                "tangent_y": attrs.EDGE_TANGENT_Y,
                "tangent_z": attrs.EDGE_TANGENT_Z,
                "normal_x": attrs.EDGE_NORMAL_X,
                "normal_y": attrs.EDGE_NORMAL_Y,
                "normal_z": attrs.EDGE_NORMAL_Z,
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
            do_exchange=False,
        )
        self.register_provider(tangent_normal_coordinates)

        # 2. primal_normals: gridfile%zonal_normal_primal_edge - edges%primal_normal%v1, gridfile%meridional_normal_primal_edge - edges%primal_normal%v2,
        normal_uv = factory.ProgramFieldProvider(
            func=coord_trans.compute_zonal_and_meridional_components_on_edges,
            deps={
                "lat": attrs.EDGE_LAT,
                "lon": attrs.EDGE_LON,
                "x": attrs.EDGE_NORMAL_X,
                "y": attrs.EDGE_NORMAL_Y,
                "z": attrs.EDGE_NORMAL_Z,
            },
            fields={
                "u": attrs.EDGE_NORMAL_U,
                "v": attrs.EDGE_NORMAL_V,
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
            do_exchange=False,
        )
        self.register_provider(normal_uv)

        dual_uv = factory.ProgramFieldProvider(
            func=coord_trans.compute_zonal_and_meridional_components_on_edges,
            deps={
                "lat": attrs.EDGE_LAT,
                "lon": attrs.EDGE_LON,
                "x": attrs.EDGE_TANGENT_X,
                "y": attrs.EDGE_TANGENT_Y,
                "z": attrs.EDGE_TANGENT_Z,
            },
            fields={
                "u": attrs.EDGE_DUAL_U,
                "v": attrs.EDGE_DUAL_V,
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
            do_exchange=False,
        )
        self.register_provider(dual_uv)

        # 3. primal_normal_vert, primal_normal_cell
        normal_vert = factory.ProgramFieldProvider(
            func=stencils.compute_zonal_and_meridional_component_of_edge_field_at_vertex,
            deps={
                "vertex_lat": attrs.VERTEX_LAT,
                "vertex_lon": attrs.VERTEX_LON,
                "x": attrs.EDGE_NORMAL_X,
                "y": attrs.EDGE_NORMAL_Y,
                "z": attrs.EDGE_NORMAL_Z,
            },
            fields={
                "u_vertex_1": "u_vertex_1",
                "v_vertex_1": "v_vertex_1",
                "u_vertex_2": "u_vertex_2",
                "v_vertex_2": "v_vertex_2",
                "u_vertex_3": "u_vertex_3",
                "v_vertex_3": "v_vertex_3",
                "u_vertex_4": "u_vertex_4",
                "v_vertex_4": "v_vertex_4",
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
            do_exchange=False,
        )
        normal_vert_wrapper = SparseFieldProviderWrapper(
            field_provider=normal_vert,
            target_dims=attrs.attrs[attrs.EDGE_NORMAL_VERTEX_U]["dims"],
            fields=(attrs.EDGE_NORMAL_VERTEX_U, attrs.EDGE_NORMAL_VERTEX_V),
            pairs=(
                ("u_vertex_1", "u_vertex_2", "u_vertex_3", "u_vertex_4"),
                ("v_vertex_1", "v_vertex_2", "v_vertex_3", "v_vertex_4"),
            ),
            do_exchange=True,
        )
        self.register_provider(normal_vert_wrapper)

        normal_cell = factory.ProgramFieldProvider(
            func=stencils.compute_zonal_and_meridional_component_of_edge_field_at_cell_center,
            deps={
                "cell_lat": attrs.CELL_LAT,
                "cell_lon": attrs.CELL_LON,
                "x": attrs.EDGE_NORMAL_X,
                "y": attrs.EDGE_NORMAL_Y,
                "z": attrs.EDGE_NORMAL_Z,
            },
            fields={
                "u_cell_1": "u_cell_1",
                "v_cell_1": "v_cell_1",
                "u_cell_2": "u_cell_2",
                "v_cell_2": "v_cell_2",
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
            do_exchange=False,
        )
        normal_cell_wrapper = SparseFieldProviderWrapper(
            field_provider=normal_cell,
            target_dims=attrs.attrs[attrs.EDGE_NORMAL_CELL_U]["dims"],
            fields=(attrs.EDGE_NORMAL_CELL_U, attrs.EDGE_NORMAL_CELL_V),
            pairs=(("u_cell_1", "u_cell_2"), ("v_cell_1", "v_cell_2")),
            do_exchange=True,
        )
        self.register_provider(normal_cell_wrapper)

        # dual normals: the dual normals are the edge tangents
        tangent_vert = factory.ProgramFieldProvider(
            func=stencils.compute_zonal_and_meridional_component_of_edge_field_at_vertex,
            deps={
                "vertex_lat": attrs.VERTEX_LAT,
                "vertex_lon": attrs.VERTEX_LON,
                "x": attrs.EDGE_TANGENT_X,
                "y": attrs.EDGE_TANGENT_Y,
                "z": attrs.EDGE_TANGENT_Z,
            },
            fields={
                "u_vertex_1": "u_vertex_1",
                "v_vertex_1": "v_vertex_1",
                "u_vertex_2": "u_vertex_2",
                "v_vertex_2": "v_vertex_2",
                "u_vertex_3": "u_vertex_3",
                "v_vertex_3": "v_vertex_3",
                "u_vertex_4": "u_vertex_4",
                "v_vertex_4": "v_vertex_4",
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
            do_exchange=False,
        )
        tangent_vert_wrapper = SparseFieldProviderWrapper(
            field_provider=tangent_vert,
            target_dims=attrs.attrs[attrs.EDGE_TANGENT_VERTEX_U]["dims"],
            fields=(attrs.EDGE_TANGENT_VERTEX_U, attrs.EDGE_TANGENT_VERTEX_V),
            pairs=(
                ("u_vertex_1", "u_vertex_2", "u_vertex_3", "u_vertex_4"),
                ("v_vertex_1", "v_vertex_2", "v_vertex_3", "v_vertex_4"),
            ),
            do_exchange=True,
        )
        self.register_provider(tangent_vert_wrapper)

        tangent_cell = factory.ProgramFieldProvider(
            func=stencils.compute_zonal_and_meridional_component_of_edge_field_at_cell_center,
            deps={
                "cell_lat": attrs.CELL_LAT,
                "cell_lon": attrs.CELL_LON,
                "x": attrs.EDGE_TANGENT_X,
                "y": attrs.EDGE_TANGENT_Y,
                "z": attrs.EDGE_TANGENT_Z,
            },
            fields={
                "u_cell_1": "u_cell_1",
                "v_cell_1": "v_cell_1",
                "u_cell_2": "u_cell_2",
                "v_cell_2": "v_cell_2",
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2),
                    self._edge_domain(h_grid.Zone.LOCAL),
                )
            },
            do_exchange=False,
        )
        tangent_cell_wrapper = SparseFieldProviderWrapper(
            field_provider=tangent_cell,
            target_dims=attrs.attrs[attrs.EDGE_TANGENT_CELL_U]["dims"],
            fields=(attrs.EDGE_TANGENT_CELL_U, attrs.EDGE_TANGENT_CELL_V),
            pairs=(("u_cell_1", "u_cell_2"), ("v_cell_1", "v_cell_2")),
            do_exchange=True,
        )
        self.register_provider(tangent_cell_wrapper)

    def _register_normals_and_tangents_torus(self) -> None:
        """Register normals and tangents specific to torus geometry."""
        # cartesian coordinates for primal_normal
        tangent_normal_coordinates = factory.ProgramFieldProvider(
            func=stencils.compute_cartesian_coordinates_of_edge_tangent_and_normal_torus,
            deps={
                "vertex_x": attrs.VERTEX_X,
                "vertex_y": attrs.VERTEX_Y,
                "edge_x": attrs.EDGE_CENTER_X,
                "edge_y": attrs.EDGE_CENTER_Y,
                "edge_orientation": attrs.TANGENT_ORIENTATION,
            },
            fields={
                "tangent_x": attrs.EDGE_TANGENT_X,
                "tangent_y": attrs.EDGE_TANGENT_Y,
                "tangent_z": attrs.EDGE_TANGENT_Z,
                "tangent_u": attrs.EDGE_DUAL_U,
                "tangent_v": attrs.EDGE_DUAL_V,
                "normal_x": attrs.EDGE_NORMAL_X,
                "normal_y": attrs.EDGE_NORMAL_Y,
                "normal_z": attrs.EDGE_NORMAL_Z,
                "normal_u": attrs.EDGE_NORMAL_U,
                "normal_v": attrs.EDGE_NORMAL_V,
            },
            domain={
                dims.EdgeDim: (
                    self._edge_domain(h_grid.Zone.LOCAL),
                    self._edge_domain(h_grid.Zone.END),
                )
            },
            params={
                "domain_length": self._grid.grid_params.domain_length,
                "domain_height": self._grid.grid_params.domain_height,
            },
            do_exchange=False,
        )
        self.register_provider(tangent_normal_coordinates)

        # primal_normal_vert, primal_normal_cell
        normal_vert_wrapper = SparseFieldProviderWrapper(
            field_provider=tangent_normal_coordinates,
            target_dims=attrs.attrs[attrs.EDGE_NORMAL_VERTEX_U]["dims"],
            fields=(attrs.EDGE_NORMAL_VERTEX_U, attrs.EDGE_NORMAL_VERTEX_V),
            pairs=(
                (
                    attrs.EDGE_NORMAL_X,
                    attrs.EDGE_NORMAL_X,
                    attrs.EDGE_NORMAL_X,
                    attrs.EDGE_NORMAL_X,
                ),
                (
                    attrs.EDGE_NORMAL_Y,
                    attrs.EDGE_NORMAL_Y,
                    attrs.EDGE_NORMAL_Y,
                    attrs.EDGE_NORMAL_Y,
                ),
            ),
            do_exchange=True,
        )
        self.register_provider(normal_vert_wrapper)

        normal_cell_wrapper = SparseFieldProviderWrapper(
            field_provider=tangent_normal_coordinates,
            target_dims=attrs.attrs[attrs.EDGE_NORMAL_CELL_U]["dims"],
            fields=(attrs.EDGE_NORMAL_CELL_U, attrs.EDGE_NORMAL_CELL_V),
            pairs=(
                (attrs.EDGE_NORMAL_X, attrs.EDGE_NORMAL_X),
                (attrs.EDGE_NORMAL_Y, attrs.EDGE_NORMAL_Y),
            ),
            do_exchange=True,
        )
        self.register_provider(normal_cell_wrapper)

        # dual normals: the dual normals are the edge tangents
        tangent_vert_wrapper = SparseFieldProviderWrapper(
            field_provider=tangent_normal_coordinates,
            target_dims=attrs.attrs[attrs.EDGE_TANGENT_VERTEX_U]["dims"],
            fields=(attrs.EDGE_TANGENT_VERTEX_U, attrs.EDGE_TANGENT_VERTEX_V),
            pairs=(
                (
                    attrs.EDGE_TANGENT_X,
                    attrs.EDGE_TANGENT_X,
                    attrs.EDGE_TANGENT_X,
                    attrs.EDGE_TANGENT_X,
                ),
                (
                    attrs.EDGE_TANGENT_Y,
                    attrs.EDGE_TANGENT_Y,
                    attrs.EDGE_TANGENT_Y,
                    attrs.EDGE_TANGENT_Y,
                ),
            ),
            do_exchange=True,
        )
        self.register_provider(tangent_vert_wrapper)

        tangent_cell_wrapper = SparseFieldProviderWrapper(
            field_provider=tangent_normal_coordinates,
            target_dims=attrs.attrs[attrs.EDGE_TANGENT_CELL_U]["dims"],
            fields=(attrs.EDGE_TANGENT_CELL_U, attrs.EDGE_TANGENT_CELL_V),
            pairs=(
                (attrs.EDGE_TANGENT_X, attrs.EDGE_TANGENT_X),
                (attrs.EDGE_TANGENT_Y, attrs.EDGE_TANGENT_Y),
            ),
            do_exchange=True,
        )
        self.register_provider(tangent_cell_wrapper)

    def _register_cartesian_coordinates_icosahedron(self) -> None:
        """Register Cartesian coordinate conversions for icosahedron geometry."""
        cartesian_vertices = factory.EmbeddedFieldOperatorProvider(
            func=coord_trans.geographical_to_cartesian_on_vertices.with_backend(self.backend),
            domain={
                dims.VertexDim: (
                    h_grid.vertex_domain(h_grid.Zone.LOCAL),
                    h_grid.vertex_domain(h_grid.Zone.END),
                )
            },
            fields={
                attrs.VERTEX_X: attrs.VERTEX_X,
                attrs.VERTEX_Y: attrs.VERTEX_Y,
                attrs.VERTEX_Z: attrs.VERTEX_Z,
            },
            deps={
                "lat": attrs.VERTEX_LAT,
                "lon": attrs.VERTEX_LON,
            },
            do_exchange=False,
        )
        self.register_provider(cartesian_vertices)
        cartesian_edge_centers = factory.EmbeddedFieldOperatorProvider(
            func=coord_trans.geographical_to_cartesian_on_edges.with_backend(self.backend),
            domain={
                dims.EdgeDim: (
                    h_grid.edge_domain(h_grid.Zone.LOCAL),
                    h_grid.edge_domain(h_grid.Zone.END),
                )
            },
            fields={
                attrs.EDGE_CENTER_X: attrs.EDGE_CENTER_X,
                attrs.EDGE_CENTER_Y: attrs.EDGE_CENTER_Y,
                attrs.EDGE_CENTER_Z: attrs.EDGE_CENTER_Z,
            },
            deps={
                "lat": attrs.EDGE_LAT,
                "lon": attrs.EDGE_LON,
            },
            do_exchange=False,
        )
        self.register_provider(cartesian_edge_centers)
        cartesian_cell_centers = factory.EmbeddedFieldOperatorProvider(
            func=coord_trans.geographical_to_cartesian_on_cells.with_backend(self.backend),
            domain={
                dims.CellDim: (
                    h_grid.cell_domain(h_grid.Zone.LOCAL),
                    h_grid.cell_domain(h_grid.Zone.END),
                )
            },
            fields={
                attrs.CELL_CENTER_X: attrs.CELL_CENTER_X,
                attrs.CELL_CENTER_Y: attrs.CELL_CENTER_Y,
                attrs.CELL_CENTER_Z: attrs.CELL_CENTER_Z,
            },
            deps={
                "lat": attrs.CELL_LAT,
                "lon": attrs.CELL_LON,
            },
            do_exchange=False,
        )
        self.register_provider(cartesian_cell_centers)

    def __repr__(self) -> str:
        geometry_name = self._geometry_type._name_ if self._geometry_type else ""
        return (
            f"{self.__class__.__name__} for geometry_type={geometry_name} (grid={self._grid.id!r})"
        )

    def get_wpfloat(self, name: str) -> float:
        return ta.wpfloat(self.get(name, type_=factory.RetrievalType.SCALAR))

    @property
    def metadata(self) -> dict[str, model.FieldMetaData]:
        return self._attrs

    @property
    def backend(self) -> gtx_typing.Backend:
        return self._backend

    @property
    def grid(self) -> icon.IconGrid:
        return self._grid

    @property
    def vertical_grid(self) -> None:
        return None


class SparseFieldProviderWrapper(factory.FieldProvider, factory.NeedsExchange):
    def __init__(
        self,
        *,
        field_provider: factory.FieldProvider,
        target_dims: Sequence[gtx.Dimension],
        fields: Sequence[str],
        pairs: Sequence[tuple[str, ...]],
        do_exchange: bool,
    ):
        assert len(target_dims) == 2
        assert target_dims[1].kind == gtx.DimensionKind.LOCAL
        self._wrapped_provider = field_provider
        self._fields = {name: None for name in fields}
        self._func = functools.partial(as_sparse_field, target_dims)
        self._pairs = pairs
        self._do_exchange = do_exchange

    def __call__(
        self,
        *,
        field_name: str,
        field_src: factory.FieldSource | None,
        backend: gtx_typing.Backend | None,
        grid: factory.GridProvider,
        exchange: decomposition.ExchangeRuntime,
    ) -> state_utils.GTXFieldType | None:
        if self._fields.get(field_name) is None:
            # get the fields from the wrapped provider
            input_fields = []
            for p in self._pairs:
                t = tuple(
                    [
                        self._wrapped_provider(
                            field_name=name,
                            field_src=field_src,
                            backend=backend,
                            grid=grid,
                            exchange=exchange,
                        )
                        for name in p
                    ]
                )
                input_fields.append(t)
            sparse_fields = self.func(input_fields, backend=backend)
            self._fields = {k: sparse_fields[i] for i, k in enumerate(self.fields)}
            self.exchange(self.fields, exchange)
        return self._fields[field_name]

    @property
    def dependencies(self) -> Sequence[str]:
        return self._wrapped_provider.dependencies

    @property
    def fields(self) -> Mapping[str, Any]:
        return self._fields

    @property
    def func(self) -> Callable:
        return self._func

    def needs_exchange(self) -> bool:
        return self._do_exchange


def as_sparse_field(
    target_dims: tuple[gtx.Dimension, gtx.Dimension],
    data: Sequence[tuple[gtx.Field[gtx.Dims[gtx.Dimension], state_utils.ScalarType], ...]],
    backend: gtx_typing.Backend | None = None,
) -> Sequence[state_utils.GTXFieldType]:
    assert len(target_dims) == 2
    assert target_dims[0].kind == gtx.DimensionKind.HORIZONTAL
    assert target_dims[1].kind == gtx.DimensionKind.LOCAL
    on_gpu = device_utils.is_cupy_device(backend)
    xp = data_alloc.array_ns(on_gpu)
    fields = []
    for t in data:
        buffers = list(b.ndarray for b in t)
        field = gtx.as_field(
            target_dims, data=(xp.vstack(buffers).T), dtype=buffers[0].dtype, allocator=backend
        )
        fields.append(field)
    return fields


def create_auxiliary_coordinate_arrays_for_orientation(
    *,
    grid: icon.IconGrid,
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    allocator: gtx_typing.Allocator | None,
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """
    Construct auxiliary arrays of geographical coordinates used in the computation of edge normal fields.

    The resulting fields are based on edges and contain geographical coordinates (lat, lon) that are
    - either the coordinates (lat, lon) of an edge's neighboring cell centers
    - or for boundary edges (that have no cell neighbor) the coordinates of the edge center

    Args:
        grid: icon grid
        cell_lat: latitude of cell centers
        cell_lon: longitude of cell centers
        edge_lat: latitude of edge centers
        edge_lon: longitude of edge centers

    Returns:
        latitude of first neighbor
        longitude of first neighbor
        latitude of second neighbor
        longitude of second neighbor
    """
    xp = data_alloc.array_ns(device_utils.is_cupy_device(allocator))
    e2c_table = grid.get_connectivity(dims.E2C).ndarray
    lat = cell_lat.ndarray[e2c_table]
    lon = cell_lon.ndarray[e2c_table]
    for i in (0, 1):
        boundary_edges = xp.where(e2c_table[:, i] == gridfile.GridFile.INVALID_INDEX)
        lat[boundary_edges, i] = edge_lat.ndarray[boundary_edges]
        lon[boundary_edges, i] = edge_lon.ndarray[boundary_edges]

    return (
        gtx.as_field((dims.EdgeDim,), lat[:, 0], allocator=allocator),
        gtx.as_field((dims.EdgeDim,), lon[:, 0], allocator=allocator),
        gtx.as_field((dims.EdgeDim,), lat[:, 1], allocator=allocator),
        gtx.as_field((dims.EdgeDim,), lon[:, 1], allocator=allocator),
    )
