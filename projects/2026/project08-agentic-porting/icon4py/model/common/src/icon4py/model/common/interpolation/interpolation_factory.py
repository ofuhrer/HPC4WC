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
import logging
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

import icon4py.model.common.interpolation.stencils.compute_nudgecoeffs as nudgecoeffs
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    geometry,
    geometry_attributes as geometry_attrs,
    grid_refinement as refinement,
    horizontal as h_grid,
    icon,
)
from icon4py.model.common.interpolation import (
    interpolation_attributes as attrs,
    interpolation_fields,
    rbf_interpolation as rbf,
)
from icon4py.model.common.states import factory, model
from icon4py.model.common.utils import data_allocation as data_alloc


cell_domain = h_grid.domain(dims.CellDim)
edge_domain = h_grid.domain(dims.EdgeDim)
vertex_domain = h_grid.domain(dims.VertexDim)

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InterpolationConfig:
    divergence_averaging_central_cell_weight: float = 0.5  # divavg_cntrwgt in ICON
    """
    Central-cell weight used in divergence averaging.
    """

    max_nudging_coefficient: float | None = None  # default: 0.375, set in __post_init__
    """
    Maximum nudging coefficient applied in the lateral nudging zone.
    """

    #: Raw namelist value (nudge_max_coeff in mo_interpol_nml.f90), scaled to
    #: max_nudging_coefficient in __post_init__ if provided.
    _nudge_max_coeff: float | None = None

    nudge_efold_width: float = 2.0
    """
    E-folding width controlling the exponential decay of nudging strength.
    """

    nudge_zone_width: int = 10
    """
    Width of the lateral nudging zone in grid refinement levels.
    """

    rbf_kernel_cell: rbf.InterpolationKernel = rbf.DEFAULT_RBF_KERNEL[rbf.RBFDimension.CELL]
    """
    Radial basis function kernel used for cell-based interpolation.
    """

    rbf_kernel_edge: rbf.InterpolationKernel = rbf.DEFAULT_RBF_KERNEL[rbf.RBFDimension.EDGE]
    """
    Radial basis function kernel used for edge-based interpolation.
    """

    rbf_kernel_vertex: rbf.InterpolationKernel = rbf.DEFAULT_RBF_KERNEL[rbf.RBFDimension.VERTEX]
    """
    Radial basis function kernel used for vertex-based interpolation.
    """

    lsq_dim_unk: int = 2
    """
    Number of unknowns in the least-squares reconstruction.
    Hardcoded in Fortran mo_intp_coeffs_lsq_bln.f90, not a namelist parameter.
    """

    lsq_dim_c: int = 3
    """
    Dimension of the least-squares coefficient space.
    Hardcoded in Fortran mo_intp_coeffs_lsq_bln.f90, not a namelist parameter.
    """

    lsq_wgt_exp: int = 2
    """
    Exponent used in distance-based least-squares weighting.
    Hardcoded in Fortran mo_intp_coeffs_lsq_bln.f90, not a namelist parameter.
    """

    lsq_dim_stencil: int = 3
    """
    Stencil size used for least-squares reconstruction.
    """

    def __post_init__(self):
        if self._nudge_max_coeff is not None and self.max_nudging_coefficient is not None:
            raise ValueError("Cannot set both '_nudge_max_coeff' and 'max_nudging_coefficient'.")
        elif self.max_nudging_coefficient is not None:
            pass
        elif self._nudge_max_coeff is not None:
            self.max_nudging_coefficient = (
                constants.DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO * self._nudge_max_coeff
            )
        else:  # default value in ICON
            self.max_nudging_coefficient = 0.375

    @classmethod
    def from_fortran_dict(cls, atmo_dict: dict[str, Any], **overrides: Any) -> InterpolationConfig:
        interpol_nml = atmo_dict["interpol_nml"]
        dynamics_nml = atmo_dict["dynamics_nml"]
        return cls(
            divergence_averaging_central_cell_weight=dynamics_nml["divavg_cntrwgt"],
            _nudge_max_coeff=interpol_nml["nudge_max_coeff"],
            nudge_efold_width=interpol_nml["nudge_efold_width"],
            nudge_zone_width=interpol_nml["nudge_zone_width"],
            rbf_kernel_cell=rbf.InterpolationKernel(interpol_nml["rbf_vec_kern_c"]),
            rbf_kernel_edge=rbf.InterpolationKernel(interpol_nml["rbf_vec_kern_e"]),
            rbf_kernel_vertex=rbf.InterpolationKernel(interpol_nml["rbf_vec_kern_v"]),
            lsq_dim_stencil=interpol_nml["lsq_high_ord"],
            **overrides,
        )


class InterpolationFieldsFactory(factory.FieldSource, factory.GridProvider):
    def __init__(
        self,
        *,
        grid: icon.IconGrid,
        decomposition_info: decomposition.DecompositionInfo,
        geometry_source: geometry.GridGeometry,
        backend: gtx_typing.Backend | None,
        metadata: dict[str, model.FieldMetaData],
        config: InterpolationConfig,
        exchange: decomposition.ExchangeRuntime = decomposition.single_node_exchange,
    ):
        self._backend = backend
        self._xp = data_alloc.import_array_ns(backend)
        self._allocator = gtx.constructors.zeros.partial(allocator=backend)
        self._grid = grid
        self._decomposition_info = decomposition_info
        self._attrs = metadata
        self._providers: dict[str, factory.FieldProvider] = {}
        self._geometry = geometry_source
        self._exchange = exchange
        self._config = config
        domain_length = self.grid.grid_params.domain_length
        domain_height = self.grid.grid_params.domain_height
        self._domain_length = 0.0 if domain_length is None else domain_length
        self._domain_height = 0.0 if domain_height is None else domain_height
        log.info(
            f"initialized interpolation factory for backend = '{self._backend_name()}' and grid = '{self._grid}'"
        )
        log.debug(f"using array_ns {self._xp} ")

        self.register_provider(
            factory.PrecomputedFieldProvider(
                fields={
                    "refinement_control_at_edges": self._grid.refinement_control[dims.EdgeDim],
                }
            )
        )

        self._register_computed_fields()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} on (grid={self._grid!r}) providing fields f{self.metadata.keys()}"

    @property
    def _sources(self) -> factory.FieldSource:
        return factory.CompositeSource(me=self, others=(self._geometry,))

    def _register_computed_fields(self) -> None:
        nudging_coefficients_for_edges = factory.ProgramFieldProvider(
            func=nudgecoeffs.compute_nudgecoeffs,
            domain={
                dims.EdgeDim: (
                    edge_domain(h_grid.Zone.NUDGING_LEVEL_2),
                    edge_domain(h_grid.Zone.END),
                )
            },
            fields={attrs.NUDGECOEFFS_E: attrs.NUDGECOEFFS_E},
            deps={
                "refin_ctrl": "refinement_control_at_edges",
            },
            params={
                "grf_nudge_start_e": refinement.get_nudging_refinement_value(dims.EdgeDim),
                "max_nudging_coefficient": self._config.max_nudging_coefficient,
                "nudge_efold_width": self._config.nudge_efold_width,
                "nudge_zone_width": self._config.nudge_zone_width,
            },
            do_exchange=True,
        )
        self.register_provider(nudging_coefficients_for_edges)

        geofac_div = factory.EmbeddedFieldOperatorProvider(
            # needs to be computed on fieldview-embedded backend
            func=interpolation_fields.compute_geofac_div.with_backend(None),
            domain=(dims.CellDim, dims.C2EDim),
            fields={attrs.GEOFAC_DIV: attrs.GEOFAC_DIV},
            deps={
                "primal_edge_length": geometry_attrs.EDGE_LENGTH,
                "edge_orientation": geometry_attrs.CELL_NORMAL_ORIENTATION,
                "area": geometry_attrs.CELL_AREA,
            },
            do_exchange=True,
        )
        self.register_provider(geofac_div)

        geofac_rot = factory.EmbeddedFieldOperatorProvider(
            # needs to be computed on fieldview-embedded backend
            func=interpolation_fields.compute_geofac_rot.with_backend(None),
            domain=(dims.VertexDim, dims.V2EDim),
            fields={attrs.GEOFAC_ROT: attrs.GEOFAC_ROT},
            deps={
                "dual_edge_length": geometry_attrs.DUAL_EDGE_LENGTH,
                "edge_orientation": geometry_attrs.VERTEX_EDGE_ORIENTATION,
                "dual_area": geometry_attrs.DUAL_AREA,
                "owner_mask": "vertex_owner_mask",
            },
            do_exchange=True,
        )
        self.register_provider(geofac_rot)

        geofac_n2s = factory.NumpyDataProvider(
            func=interpolation_fields.compute_geofac_n2s,
            fields=(attrs.GEOFAC_N2S,),
            domain=(dims.CellDim, dims.C2E2CODim),
            deps={
                "dual_edge_length": geometry_attrs.DUAL_EDGE_LENGTH,
                "geofac_div": attrs.GEOFAC_DIV,
            },
            connectivities={"c2e": dims.C2EDim, "e2c": dims.E2CDim, "c2e2c": dims.C2E2CDim},
            params={
                "horizontal_start": self._grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                )
            },
            do_exchange=True,
        )
        self.register_provider(geofac_n2s)

        geofac_grdiv = factory.NumpyDataProvider(
            func=interpolation_fields.compute_geofac_grdiv,
            fields=(attrs.GEOFAC_GRDIV,),
            domain=(dims.EdgeDim, dims.E2C2EODim),
            deps={
                "geofac_div": attrs.GEOFAC_DIV,
                "inv_dual_edge_length": f"inverse_of_{geometry_attrs.DUAL_EDGE_LENGTH}",
                "owner_mask": "edge_owner_mask",
            },
            connectivities={"c2e": dims.C2EDim, "e2c": dims.E2CDim, "e2c2e": dims.E2C2EDim},
            params={
                "horizontal_start": self._grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                )
            },
            do_exchange=True,
        )

        self.register_provider(geofac_grdiv)

        rbf_scale_cell_np = factory.NumpyDataProvider(
            func=rbf.compute_default_rbf_scale_cell,
            domain=(),
            deps={
                "mean_characteristic_length": geometry_attrs.CHARACTERISTIC_LENGTH,
                "mean_dual_edge_length": geometry_attrs.MEAN_DUAL_EDGE_LENGTH,
            },
            params={
                "geometry_type": self.grid.grid_params.geometry_type.value,
            },
            fields=(attrs.RBF_SCALE_CELL,),
        )
        self.register_provider(rbf_scale_cell_np)

        rbf_scale_edge_np = factory.NumpyDataProvider(
            func=rbf.compute_default_rbf_scale_edge,
            domain=(),
            deps={
                "mean_characteristic_length": geometry_attrs.CHARACTERISTIC_LENGTH,
                "mean_dual_edge_length": geometry_attrs.MEAN_DUAL_EDGE_LENGTH,
            },
            params={
                "geometry_type": self.grid.grid_params.geometry_type.value,
            },
            fields=(attrs.RBF_SCALE_EDGE,),
        )
        self.register_provider(rbf_scale_edge_np)

        rbf_scale_vertex_np = factory.NumpyDataProvider(
            func=rbf.compute_default_rbf_scale_vertex,
            domain=(),
            deps={
                "mean_characteristic_length": geometry_attrs.CHARACTERISTIC_LENGTH,
                "mean_dual_edge_length": geometry_attrs.MEAN_DUAL_EDGE_LENGTH,
            },
            params={
                "geometry_type": self.grid.grid_params.geometry_type.value,
            },
            fields=(attrs.RBF_SCALE_VERTEX,),
        )
        self.register_provider(rbf_scale_vertex_np)

        lsq_pseudoinv = factory.NumpyDataProvider(
            func=functools.partial(
                interpolation_fields.compute_lsq_coeffs,
                exchange=self._exchange,
            ),
            fields=(attrs.LSQ_PSEUDOINV,),
            domain=(dims.CellDim, dims.LsqUnkDim, dims.C2E2CDim),
            deps={
                "cell_center_x": geometry_attrs.CELL_CENTER_X,
                "cell_center_y": geometry_attrs.CELL_CENTER_Y,
                "cell_lat": geometry_attrs.CELL_LAT,
                "cell_lon": geometry_attrs.CELL_LON,
                "cell_owner_mask": "cell_owner_mask",
            },
            connectivities={"c2e2c": dims.C2E2CDim},
            params={
                "domain_length": self._domain_length,
                "domain_height": self._domain_height,
                "grid_sphere_radius": constants.EARTH_RADIUS,
                "lsq_dim_unk": self._config.lsq_dim_unk,
                "lsq_dim_c": self._config.lsq_dim_c,
                "lsq_wgt_exp": self._config.lsq_wgt_exp,
                "lsq_dim_stencil": self._config.lsq_dim_stencil,
                "start_idx": self.grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
                "min_rlcell_int": self.grid.end_index(cell_domain(h_grid.Zone.HALO_LEVEL_2)),
                "geometry_type": self.grid.grid_params.geometry_type.value,
            },
        )
        self.register_provider(lsq_pseudoinv)

        match self.grid.grid_params.geometry_type:
            case icon.GeometryType.ICOSAHEDRON:
                cell_average_weight = factory.NumpyDataProvider(
                    func=functools.partial(
                        interpolation_fields.compute_mass_conserving_bilinear_cell_average_weight,
                        exchange=self._exchange,
                    ),
                    fields=(attrs.C_BLN_AVG,),
                    domain=(dims.CellDim, dims.C2E2CODim),
                    deps={
                        "lat": geometry_attrs.CELL_LAT,
                        "lon": geometry_attrs.CELL_LON,
                        "cell_areas": geometry_attrs.CELL_AREA,
                        "cell_owner_mask": "cell_owner_mask",
                    },
                    connectivities={"c2e2c0": dims.C2E2CODim},
                    params={
                        "horizontal_start": self.grid.start_index(
                            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                        ),
                        "horizontal_start_level_3": self.grid.start_index(
                            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
                        ),
                        "divergence_averaging_central_cell_weight": self._config.divergence_averaging_central_cell_weight,
                    },
                )
                self.register_provider(cell_average_weight)

                e_bln_c_s = factory.NumpyDataProvider(
                    func=functools.partial(
                        interpolation_fields.compute_e_bln_c_s,
                    ),
                    fields=(attrs.E_BLN_C_S,),
                    domain=(dims.CellDim, dims.C2EDim),
                    deps={
                        "cells_lat": geometry_attrs.CELL_LAT,
                        "cells_lon": geometry_attrs.CELL_LON,
                        "edges_lat": geometry_attrs.EDGE_LAT,
                        "edges_lon": geometry_attrs.EDGE_LON,
                    },
                    connectivities={"c2e": dims.C2EDim},
                )
                self.register_provider(e_bln_c_s)

                pos_on_tplane_e_x_y = factory.NumpyDataProvider(
                    func=interpolation_fields.compute_pos_on_tplane_e_x_y,
                    fields=(attrs.POS_ON_TPLANE_E_X, attrs.POS_ON_TPLANE_E_Y),
                    domain=(dims.EdgeDim, dims.E2CDim),
                    deps={
                        "primal_normal_v1": geometry_attrs.EDGE_NORMAL_U,
                        "primal_normal_v2": geometry_attrs.EDGE_NORMAL_V,
                        "dual_normal_v1": geometry_attrs.EDGE_DUAL_U,
                        "dual_normal_v2": geometry_attrs.EDGE_DUAL_V,
                        "cells_lon": geometry_attrs.CELL_LON,
                        "cells_lat": geometry_attrs.CELL_LAT,
                        "edges_lon": geometry_attrs.EDGE_LON,
                        "edges_lat": geometry_attrs.EDGE_LAT,
                        "owner_mask": "edge_owner_mask",
                    },
                    connectivities={"e2c": dims.E2CDim},
                    params={
                        "grid_sphere_radius": constants.EARTH_RADIUS,
                        "horizontal_start": self.grid.start_index(
                            edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                        ),
                    },
                    do_exchange=True,
                )
                self.register_provider(pos_on_tplane_e_x_y)

            case icon.GeometryType.TORUS:
                cell_average_weight = factory.NumpyDataProvider(
                    func=functools.partial(
                        interpolation_fields.compute_mass_conserving_bilinear_cell_average_weight_torus,
                        exchange=self._exchange,
                    ),
                    fields=(attrs.C_BLN_AVG,),
                    domain=(dims.CellDim, dims.C2E2CODim),
                    deps={
                        "cell_areas": geometry_attrs.CELL_AREA,
                        "cell_owner_mask": "cell_owner_mask",
                    },
                    connectivities={"c2e2c0": dims.C2E2CODim},
                    params={
                        "horizontal_start": self.grid.start_index(
                            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                        ),
                        "horizontal_start_level_3": self.grid.start_index(
                            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
                        ),
                        "divergence_averaging_central_cell_weight": self._config.divergence_averaging_central_cell_weight,
                    },
                )
                self.register_provider(cell_average_weight)

                e_bln_c_s = factory.NumpyDataProvider(
                    func=functools.partial(
                        interpolation_fields.compute_e_bln_c_s_torus,
                    ),
                    fields=(attrs.E_BLN_C_S,),
                    domain=(dims.CellDim, dims.C2EDim),
                    deps={},
                    connectivities={"c2e": dims.C2EDim},
                    params={},
                )
                self.register_provider(e_bln_c_s)

                pos_on_tplane_e_x_y = factory.NumpyDataProvider(
                    func=interpolation_fields.compute_pos_on_tplane_e_x_y_torus,
                    fields=(attrs.POS_ON_TPLANE_E_X, attrs.POS_ON_TPLANE_E_Y),
                    domain=(dims.EdgeDim, dims.E2CDim),
                    deps={
                        "dual_edge_length": geometry_attrs.DUAL_EDGE_LENGTH,
                    },
                    connectivities={"e2c": dims.E2CDim},
                    params={},
                    do_exchange=True,
                )
                self.register_provider(pos_on_tplane_e_x_y)

        c_lin_e = factory.NumpyDataProvider(
            func=interpolation_fields.compute_c_lin_e,
            fields=(attrs.C_LIN_E,),
            domain=(dims.EdgeDim, dims.E2CDim),
            deps={
                "edge_cell_length": geometry_attrs.EDGE_CELL_DISTANCE,
                "inv_dual_edge_length": f"inverse_of_{geometry_attrs.DUAL_EDGE_LENGTH}",
                "edge_owner_mask": "edge_owner_mask",
            },
            params={
                "horizontal_start": self._grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                )
            },
            do_exchange=True,
        )
        self.register_provider(c_lin_e)

        geofac_grg = factory.NumpyDataProvider(
            func=functools.partial(
                interpolation_fields.compute_geofac_grg,
                exchange=self._exchange,
            ),
            fields=(attrs.GEOFAC_GRG_X, attrs.GEOFAC_GRG_Y),
            domain=(dims.CellDim, dims.C2E2CODim),
            deps={
                "primal_normal_cell_x": geometry_attrs.EDGE_NORMAL_CELL_U,
                "primal_normal_cell_y": geometry_attrs.EDGE_NORMAL_CELL_V,
                "owner_mask": "cell_owner_mask",
                "geofac_div": attrs.GEOFAC_DIV,
                "c_lin_e": attrs.C_LIN_E,
            },
            connectivities={"c2e": dims.C2EDim, "e2c": dims.E2CDim, "c2e2c": dims.C2E2CDim},
            params={
                "horizontal_start": self.grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                )
            },
        )
        self.register_provider(geofac_grg)

        e_flx_avg = factory.NumpyDataProvider(
            func=functools.partial(
                interpolation_fields.compute_e_flx_avg,
                exchange=self._exchange,
            ),
            fields=(attrs.E_FLX_AVG,),
            domain=(dims.EdgeDim, dims.E2C2EODim),
            deps={
                "c_bln_avg": attrs.C_BLN_AVG,
                "geofac_div": attrs.GEOFAC_DIV,
                "owner_mask": "edge_owner_mask",
                "primal_cart_normal_x": geometry_attrs.EDGE_NORMAL_X,
                "primal_cart_normal_y": geometry_attrs.EDGE_NORMAL_Y,
                "primal_cart_normal_z": geometry_attrs.EDGE_NORMAL_Z,
            },
            connectivities={
                "e2c": dims.E2CDim,
                "c2e": dims.C2EDim,
                "c2e2c": dims.C2E2CDim,
                "e2c2e": dims.E2C2EDim,
            },
            params={
                "horizontal_start_p3": self.grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)
                ),
                "horizontal_start_p4": self.grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
                ),
            },
        )
        self.register_provider(e_flx_avg)

        cells_aw_verts = factory.NumpyDataProvider(
            func=interpolation_fields.compute_cells_aw_verts,
            fields=(attrs.CELL_AW_VERTS,),
            domain=(dims.VertexDim, dims.V2CDim),
            deps={
                "dual_area": geometry_attrs.DUAL_AREA,
                "edge_vert_length": geometry_attrs.EDGE_VERTEX_DISTANCE,
                "edge_cell_length": geometry_attrs.EDGE_CELL_DISTANCE,
            },
            connectivities={
                "v2e": dims.V2EDim,
                "e2v": dims.E2VDim,
                "v2c": dims.V2CDim,
                "e2c": dims.E2CDim,
            },
            params={
                "horizontal_start": self.grid.start_index(
                    vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                )
            },
            do_exchange=True,
        )
        self.register_provider(cells_aw_verts)

        rbf_vec_coeff_c = factory.NumpyDataProvider(
            func=rbf.compute_rbf_interpolation_coeffs_cell,
            fields=(attrs.RBF_VEC_COEFF_C1, attrs.RBF_VEC_COEFF_C2),
            domain=(dims.CellDim, dims.C2E2C2EDim),
            deps={
                "cell_center_lat": geometry_attrs.CELL_LAT,
                "cell_center_lon": geometry_attrs.CELL_LON,
                "cell_center_x": geometry_attrs.CELL_CENTER_X,
                "cell_center_y": geometry_attrs.CELL_CENTER_Y,
                "cell_center_z": geometry_attrs.CELL_CENTER_Z,
                "edge_center_x": geometry_attrs.EDGE_CENTER_X,
                "edge_center_y": geometry_attrs.EDGE_CENTER_Y,
                "edge_center_z": geometry_attrs.EDGE_CENTER_Z,
                "edge_normal_x": geometry_attrs.EDGE_NORMAL_X,
                "edge_normal_y": geometry_attrs.EDGE_NORMAL_Y,
                "edge_normal_z": geometry_attrs.EDGE_NORMAL_Z,
                "scale_factor": attrs.RBF_SCALE_CELL,
            },
            connectivities={"rbf_offset": dims.C2E2C2EDim},
            params={
                "rbf_kernel": self._config.rbf_kernel_cell.value,
                "geometry_type": self._grid.grid_params.geometry_type.value,
                "horizontal_start": self.grid.start_index(
                    cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
                "horizontal_end": self.grid.end_index(cell_domain(h_grid.Zone.LOCAL)),
                "domain_length": self._grid.grid_params.domain_length
                if self._grid.grid_params.domain_length
                else -1.0,
                "domain_height": self._grid.grid_params.domain_height
                if self._grid.grid_params.domain_height
                else -1.0,
            },
            do_exchange=True,
        )
        self.register_provider(rbf_vec_coeff_c)

        rbf_vec_coeff_e = factory.NumpyDataProvider(
            func=rbf.compute_rbf_interpolation_coeffs_edge,
            fields=(attrs.RBF_VEC_COEFF_E,),
            domain=(dims.EdgeDim, dims.E2C2EDim),
            deps={
                "edge_lat": geometry_attrs.EDGE_LAT,
                "edge_lon": geometry_attrs.EDGE_LON,
                "edge_center_x": geometry_attrs.EDGE_CENTER_X,
                "edge_center_y": geometry_attrs.EDGE_CENTER_Y,
                "edge_center_z": geometry_attrs.EDGE_CENTER_Z,
                "edge_normal_x": geometry_attrs.EDGE_NORMAL_X,
                "edge_normal_y": geometry_attrs.EDGE_NORMAL_Y,
                "edge_normal_z": geometry_attrs.EDGE_NORMAL_Z,
                "edge_dual_normal_u": geometry_attrs.EDGE_DUAL_U,
                "edge_dual_normal_v": geometry_attrs.EDGE_DUAL_V,
                "scale_factor": attrs.RBF_SCALE_EDGE,
            },
            connectivities={"rbf_offset": dims.E2C2EDim},
            params={
                "rbf_kernel": self._config.rbf_kernel_edge.value,
                "geometry_type": self._grid.grid_params.geometry_type.value,
                "horizontal_start": self.grid.start_index(
                    edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
                "horizontal_end": self.grid.end_index(edge_domain(h_grid.Zone.LOCAL)),
                "domain_length": self._grid.grid_params.domain_length
                if self._grid.grid_params.domain_length
                else -1.0,
                "domain_height": self._grid.grid_params.domain_height
                if self._grid.grid_params.domain_height
                else -1.0,
            },
            do_exchange=True,
        )
        self.register_provider(rbf_vec_coeff_e)

        rbf_vec_coeff_v = factory.NumpyDataProvider(
            func=rbf.compute_rbf_interpolation_coeffs_vertex,
            fields=(attrs.RBF_VEC_COEFF_V1, attrs.RBF_VEC_COEFF_V2),
            domain=(dims.VertexDim, dims.V2EDim),
            deps={
                "vertex_lat": geometry_attrs.VERTEX_LAT,
                "vertex_lon": geometry_attrs.VERTEX_LON,
                "vertex_x": geometry_attrs.VERTEX_X,
                "vertex_y": geometry_attrs.VERTEX_Y,
                "vertex_z": geometry_attrs.VERTEX_Z,
                "edge_center_x": geometry_attrs.EDGE_CENTER_X,
                "edge_center_y": geometry_attrs.EDGE_CENTER_Y,
                "edge_center_z": geometry_attrs.EDGE_CENTER_Z,
                "edge_normal_x": geometry_attrs.EDGE_NORMAL_X,
                "edge_normal_y": geometry_attrs.EDGE_NORMAL_Y,
                "edge_normal_z": geometry_attrs.EDGE_NORMAL_Z,
                "scale_factor": attrs.RBF_SCALE_VERTEX,
            },
            connectivities={"rbf_offset": dims.V2EDim},
            params={
                "rbf_kernel": self._config.rbf_kernel_vertex.value,
                "geometry_type": self._grid.grid_params.geometry_type.value,
                "horizontal_start": self.grid.start_index(
                    vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
                ),
                "horizontal_end": self.grid.end_index(vertex_domain(h_grid.Zone.LOCAL)),
                "domain_length": self._grid.grid_params.domain_length
                if self._grid.grid_params.domain_length
                else -1.0,
                "domain_height": self._grid.grid_params.domain_height
                if self._grid.grid_params.domain_height
                else -1.0,
            },
            do_exchange=True,
        )
        self.register_provider(rbf_vec_coeff_v)

    @property
    def metadata(self) -> dict[str, model.FieldMetaData]:
        return self._attrs

    @property
    def backend(self) -> gtx_typing.Backend | None:
        return self._backend

    @property
    def grid(self) -> icon.IconGrid:
        return self._grid

    @property
    def vertical_grid(self) -> None:
        return None
