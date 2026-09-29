# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import math

import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base as base_grid, icon as icon_grid
from icon4py.model.common.utils import data_allocation as data_alloc


class RBFDimension(enum.Enum):
    CELL = "cell"
    EDGE = "edge"
    VERTEX = "vertex"
    GRADIENT = "midpoint_gradient"


RBF_STENCIL_SIZE: dict[RBFDimension, int] = {
    RBFDimension.CELL: 9,
    RBFDimension.EDGE: 4,
    RBFDimension.VERTEX: 6,
    RBFDimension.GRADIENT: 10,
}


class InterpolationKernel(enum.IntEnum):
    GAUSSIAN = 1
    INVERSE_MULTIQUADRATIC = 3


DEFAULT_RBF_KERNEL: dict[RBFDimension, InterpolationKernel] = {
    RBFDimension.CELL: InterpolationKernel.GAUSSIAN,
    RBFDimension.EDGE: InterpolationKernel.INVERSE_MULTIQUADRATIC,
    RBFDimension.VERTEX: InterpolationKernel.GAUSSIAN,
    RBFDimension.GRADIENT: InterpolationKernel.GAUSSIAN,
}


def compute_default_rbf_scale_cell(
    geometry_type: int,
    mean_characteristic_length: ta.wpfloat,
    mean_dual_edge_length: ta.wpfloat,
) -> ta.wpfloat:
    """Compute the default RBF scale factor for cells. This assumes that the Gaussian
    kernel is used."""

    match icon_grid.GeometryType(geometry_type):
        case icon_grid.GeometryType.ICOSAHEDRON:
            threshold = 2.5
            c1 = 1.8
            c2 = 3.75
            c3 = 0.9

            resol = mean_characteristic_length / 1000.0
            scale = (
                0.5 / (1.0 + c1 * math.log(threshold / resol) ** c2) if resol < threshold else 0.5
            )
            return astype(scale * (resol / 0.125) ** c3 if resol <= 0.125 else scale, ta.wpfloat)
        case icon_grid.GeometryType.TORUS:
            return mean_dual_edge_length


def compute_default_rbf_scale_edge(
    geometry_type: int,
    mean_characteristic_length: ta.wpfloat,
    mean_dual_edge_length: ta.wpfloat,
) -> ta.wpfloat:
    """Compute the default RBF scale factor for edges. This assumes that the inverse multiquadratic
    kernel is used."""

    match icon_grid.GeometryType(geometry_type):
        case icon_grid.GeometryType.ICOSAHEDRON:
            threshold = 2.0
            c1 = 0.4
            c2 = 2.0
            c3 = 0.325

            resol = mean_characteristic_length / 1000.0
            scale = (
                0.5 / (1.0 + c1 * math.log(threshold / resol) ** c2) if resol < threshold else 0.5
            )
            return astype(scale * (resol / 0.125) ** c3 if resol <= 0.125 else scale, ta.wpfloat)
        case icon_grid.GeometryType.TORUS:
            return mean_dual_edge_length


def compute_default_rbf_scale_vertex(
    geometry_type: int,
    mean_characteristic_length: ta.wpfloat,
    mean_dual_edge_length: ta.wpfloat,
) -> ta.wpfloat:
    """Compute the default RBF scale factor for vertices. This assumes that the Gaussian
    kernel is used."""

    match icon_grid.GeometryType(geometry_type):
        case icon_grid.GeometryType.ICOSAHEDRON:
            threshold = 2.0
            c1 = 1.8
            c2 = 3.0
            c3 = 0.96

            resol = mean_characteristic_length / 1000.0
            scale = (
                0.5 / (1.0 + c1 * math.log(threshold / resol) ** c2) if resol < threshold else 0.5
            )
            return astype(scale * (resol / 0.125) ** c3 if resol <= 0.125 else scale, ta.wpfloat)
        case icon_grid.GeometryType.TORUS:
            return mean_dual_edge_length


def construct_rbf_matrix_offsets_tables_for_cells(
    grid: base_grid.Grid,
) -> data_alloc.NDArray:
    """Compute the neighbor tables for the cell RBF matrix: rbf_vec_index_c"""
    connectivity = grid.get_connectivity(dims.C2E2C2E).ndarray
    assert connectivity.shape == (grid.num_cells, RBF_STENCIL_SIZE[RBFDimension.CELL])
    return connectivity


def construct_rbf_matrix_offsets_tables_for_edges(
    grid: base_grid.Grid,
) -> data_alloc.NDArray:
    """Compute the neighbor tables for the edge RBF matrix: rbf_vec_index_e"""
    connectivity = grid.get_connectivity(dims.E2C2E).ndarray
    assert connectivity.shape == (grid.num_edges, RBF_STENCIL_SIZE[RBFDimension.EDGE])
    return connectivity


def construct_rbf_matrix_offsets_tables_for_vertices(
    grid: base_grid.Grid,
) -> data_alloc.NDArray:
    """Compute the neighbor tables for the edge RBF matrix: rbf_vec_index_v"""
    connectivity = grid.get_connectivity(dims.V2E).ndarray
    assert connectivity.shape == (grid.num_vertices, RBF_STENCIL_SIZE[RBFDimension.VERTEX])
    return connectivity


def _dot_product(v1: data_alloc.NDArray, v2: data_alloc.NDArray) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(v1)
    # alias: array_ns.transpose(v2, axes=(0, 2, 1)) for 3d array
    v2_tilde = array_ns.moveaxis(v2, 1, -1)
    # use linalg.matmul (array API compatible)
    return array_ns.matmul(v1, v2_tilde)


def _compute_distance_pairwise(
    geometry_type: icon_grid.GeometryType,
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
    v: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """
    Compute the distance between points in each row of v.
    For the icosahedron geometry this is the arc lengths (in radians), for the
    torus it is the Euclidean distance taking into account periodic boundaries.

    Args:
        v: 3D array of shape (n, m, 3) where n is the number of elements,
           m is the number of points per row (RBF dimension), and 3 is the
           dimension of the points.
    """
    array_ns = data_alloc.array_namespace(v)
    match geometry_type:
        case icon_grid.GeometryType.ICOSAHEDRON:
            # For pairs of points p1 and p2 compute:
            # arccos(dot(p1, p2) / (norm(p1) * norm(p2))) noqa: ERA001
            # Compute all pairs of dot products
            arc_lengths = _dot_product(v, v)
            # Use the dot product of the diagonals to get the norm of each point
            norms = array_ns.sqrt(array_ns.diagonal(arc_lengths, axis1=1, axis2=2))
            # Divide the dot products by the broadcasted norms
            array_ns.divide(arc_lengths, norms[:, :, array_ns.newaxis], out=arc_lengths)
            array_ns.divide(arc_lengths, norms[:, array_ns.newaxis, :], out=arc_lengths)
            # Ensure all points are within [-1.0, 1.0] (may be outside due to numerical
            # inaccuracies)
            array_ns.clip(arc_lengths, -1.0, 1.0, out=arc_lengths)
            return array_ns.arccos(arc_lengths)
        case icon_grid.GeometryType.TORUS:
            # For pairs of points p1 and p2 compute:
            # norm(p1 - p2), taking into account the periodic boundaries noqa: ERA001
            diff = array_ns.abs(v[:, :, array_ns.newaxis, :] - v[:, array_ns.newaxis, :, :])
            domain_size = array_ns.asarray([domain_length, domain_height, ta.wpfloat(0.0)])
            domain_size_expanded = domain_size[array_ns.newaxis, array_ns.newaxis, :]
            inverted_diff = array_ns.subtract(domain_size_expanded, diff)
            array_ns.minimum(diff, inverted_diff, out=diff)
            return array_ns.linalg.norm(diff, axis=-1)


def _compute_distance_vector_matrix(
    geometry_type: icon_grid.GeometryType,
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
    v1: data_alloc.NDArray,
    v2: data_alloc.NDArray,
) -> data_alloc.NDArray:
    """
    Compute the distance between each point in v1 and the points in v2 at the
    same row.
    For the icosahedron geometry this is the arc lengths (in radians), for the
    torus it is the Euclidean distance taking into account periodic boundaries.

    Args:
        v1: 2D array of shape (n, 3) where n is the number of elements and 3 is
            the dimension of the points.
        v2: 3D array of shape (n, m, 3) where n is the number of elements,  m is
            the number of points per row (RBF dimension), and 3 is the dimension
            of the points.
    """
    array_ns = data_alloc.array_namespace(v1)
    match geometry_type:
        case icon_grid.GeometryType.ICOSAHEDRON:
            # For pairs of points p1 and p2 compute:
            # arccos(dot(p1, p2) / (norm(p1) * norm(p2))) noqa: ERA001
            # Compute all pairs of dot products
            arc_lengths = _dot_product(v1, v2)
            v1_norm = array_ns.linalg.norm(v1, axis=-1)
            v2_norm = array_ns.linalg.norm(v2, axis=-1)
            # Divide the dot products by the broadcasted norms
            array_ns.divide(arc_lengths, v1_norm[:, :, array_ns.newaxis], out=arc_lengths)
            array_ns.divide(arc_lengths, v2_norm[:, array_ns.newaxis, :], out=arc_lengths)
            # Ensure all points are within [-1.0, 1.0] (may be outside due to numerical
            # inaccuracies)
            array_ns.clip(arc_lengths, -1.0, 1.0, out=arc_lengths)
            return array_ns.squeeze(array_ns.arccos(arc_lengths), axis=1)
        case icon_grid.GeometryType.TORUS:
            # For pairs of points p1 and p2 compute:
            # norm(p1 - p2) noqa: ERA001
            diff = array_ns.abs(v1 - v2)
            domain_size = array_ns.asarray([domain_length, domain_height, ta.wpfloat(0.0)])
            domain_size_expanded = domain_size[array_ns.newaxis, array_ns.newaxis, :]
            inverted_diff = array_ns.subtract(domain_size_expanded, diff)
            diff = array_ns.minimum(diff, inverted_diff, out=diff)
            return array_ns.linalg.norm(diff, axis=-1)


def _gaussian(lengths: data_alloc.NDArray, scale: ta.wpfloat) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(lengths)
    val = lengths / scale
    return array_ns.exp(-1.0 * val * val)


def _inverse_multiquadratic(
    distance: data_alloc.NDArray,
    scale: ta.wpfloat,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(distance)
    val = distance / scale
    return 1.0 / array_ns.sqrt(1.0 + val * val)


def _kernel(
    kernel: InterpolationKernel,
    lengths: data_alloc.NDArray,
    scale: ta.wpfloat,
):
    match kernel:
        case InterpolationKernel.GAUSSIAN:
            return _gaussian(lengths, scale)
        case InterpolationKernel.INVERSE_MULTIQUADRATIC:
            return _inverse_multiquadratic(lengths, scale)
        case _:
            raise ValueError(f"Unsupported kernel: {kernel}")


def _cartesian_coordinates_from_zonal_and_meridional_components(
    geometry_type: icon_grid.GeometryType,
    lat: data_alloc.NDArray,
    lon: data_alloc.NDArray,
    u: data_alloc.NDArray,
    v: data_alloc.NDArray,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(lat)
    match geometry_type:
        case icon_grid.GeometryType.ICOSAHEDRON:
            cos_lat = array_ns.cos(lat)
            sin_lat = array_ns.sin(lat)
            cos_lon = array_ns.cos(lon)
            sin_lon = array_ns.sin(lon)

            x = -u * sin_lon - v * sin_lat * cos_lon
            y = u * cos_lon - v * sin_lat * sin_lon
            z = cos_lat * v

            return x, y, z
        case icon_grid.GeometryType.TORUS:
            return u, v, array_ns.zeros_like(u)


def _compute_rbf_interpolation_coeffs(
    *,
    element_center_lat: data_alloc.NDArray,
    element_center_lon: data_alloc.NDArray,
    element_center_x: data_alloc.NDArray,
    element_center_y: data_alloc.NDArray,
    element_center_z: data_alloc.NDArray,
    edge_center_x: data_alloc.NDArray,
    edge_center_y: data_alloc.NDArray,
    edge_center_z: data_alloc.NDArray,
    edge_normal_x: data_alloc.NDArray,
    edge_normal_y: data_alloc.NDArray,
    edge_normal_z: data_alloc.NDArray,
    uv: tuple[tuple[data_alloc.NDArray, data_alloc.NDArray], ...],
    rbf_offset: data_alloc.NDArray,
    rbf_kernel: InterpolationKernel,
    geometry_type: icon_grid.GeometryType,
    scale_factor: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
) -> tuple[data_alloc.NDArray, ...]:
    array_ns = data_alloc.array_namespace(element_center_lat)
    rbf_offset_shape_full = rbf_offset.shape
    assert 0 <= horizontal_start <= horizontal_end <= rbf_offset_shape_full[0]
    rbf_offset = rbf_offset[horizontal_start:horizontal_end]

    # Pad edge normals and centers with a dummy zero for easier vectorized
    # computation. This may produce nans (e.g. arc length between (0,0,0) and
    # another point on the sphere), but these don't hurt the computation.
    def pad(f):
        return array_ns.pad(f, (0, 1), mode="constant", constant_values=0.0)

    def index_offset(f):
        return f[rbf_offset]

    edge_normal = array_ns.stack(
        (
            index_offset(pad(edge_normal_x)),
            index_offset(pad(edge_normal_y)),
            index_offset(pad(edge_normal_z)),
        ),
        axis=-1,
    )
    assert edge_normal.shape == (*rbf_offset.shape, 3)

    edge_center = array_ns.stack(
        (
            index_offset(pad(edge_center_x)),
            index_offset(pad(edge_center_y)),
            index_offset(pad(edge_center_z)),
        ),
        axis=-1,
    )
    assert edge_center.shape == (*rbf_offset.shape, 3)

    # Compute distances for right hand side(s) of linear system
    element_center = array_ns.stack(
        (
            element_center_x[horizontal_start:horizontal_end],
            element_center_y[horizontal_start:horizontal_end],
            element_center_z[horizontal_start:horizontal_end],
        ),
        axis=-1,
    )
    assert element_center.shape == (rbf_offset.shape[0], 3)
    vector_dist = _compute_distance_vector_matrix(
        geometry_type,
        domain_length,
        domain_height,
        element_center[:, array_ns.newaxis, :],
        edge_center,
    )
    assert vector_dist.shape == rbf_offset.shape
    rbf_val = _kernel(rbf_kernel, vector_dist, scale_factor)
    assert rbf_val.shape == rbf_offset.shape

    # Set up right hand side(s) of linear system
    z_nx = []
    nxnx = []
    rhs = []
    num_zonal_meridional_components = len(uv)

    assert 1 <= num_zonal_meridional_components <= 2
    for i in range(num_zonal_meridional_components):
        z_nx_x, z_nx_y, z_nx_z = _cartesian_coordinates_from_zonal_and_meridional_components(
            geometry_type,
            element_center_lat[horizontal_start:horizontal_end],
            element_center_lon[horizontal_start:horizontal_end],
            uv[i][0][horizontal_start:horizontal_end],
            uv[i][1][horizontal_start:horizontal_end],
        )
        z_nx.append(array_ns.stack((z_nx_x, z_nx_y, z_nx_z), axis=-1))
        assert z_nx[i].shape == (rbf_offset.shape[0], 3)

        nxnx.append(
            array_ns.matmul(
                z_nx[i][:, array_ns.newaxis, :], edge_normal.transpose(0, 2, 1)
            ).squeeze()
        )
        rhs.append(rbf_val * nxnx[i])
        assert rhs[i].shape == rbf_offset.shape

    # Compute dot product of normal vectors for RBF interpolation matrix
    z_nxprod = _dot_product(edge_normal, edge_normal)
    assert z_nxprod.shape == (
        rbf_offset.shape[0],
        rbf_offset.shape[1],
        rbf_offset.shape[1],
    )

    # Distance between edge midpoints for RBF interpolation matrix
    z_dist = _compute_distance_pairwise(geometry_type, domain_length, domain_height, edge_center)
    assert z_dist.shape == (
        rbf_offset.shape[0],
        rbf_offset.shape[1],
        rbf_offset.shape[1],
    )

    # Set up RBF interpolation matrix
    z_rbfmat = z_nxprod * _kernel(rbf_kernel, z_dist, scale_factor)
    assert z_rbfmat.shape == (
        rbf_offset.shape[0],
        rbf_offset.shape[1],
        rbf_offset.shape[1],
    )

    # Solve linear system for coefficients.
    rbf_vec_coeff = [
        array_ns.zeros(rbf_offset_shape_full, dtype=ta.wpfloat)
        for _ in range(num_zonal_meridional_components)
    ]
    # Batch solve by grouping elements with the same number of valid neighbors.
    # ASSUMPTIONS FOR MAKING THE FOLLOWING BATCH SOLVE POSSIBLE:
    #   (1) In ICON grids, valid entries in connectivity tables are contiguous from the start.
    #       In other words, those invalid neighbors must be located at the end of the neighbor list.
    #       Therefore, we cannot compute rbf all the way into halo cells or the first boundary layer
    #       because the invalid neighbors may appear in the middle the the neighbor list.
    #   (2) Invalid indices must be a negative.
    n_valid = (rbf_offset >= 0).sum(axis=1)
    for nv in (u := array_ns.unique(n_valid))[u != 0]:
        group_idx = array_ns.where(n_valid == nv)[0]
        valid_cols = array_ns.arange(nv)
        mat_batch = z_rbfmat[array_ns.ix_(group_idx, valid_cols, valid_cols)]
        for j in range(num_zonal_meridional_components):
            rhs_batch = rhs[j][array_ns.ix_(group_idx, valid_cols)]
            # array_ns.linalg.solve supports batched inputs: mat_batch (B, nv, nv),
            # rhs_batch (B, nv, 1). The solution of mat_batch x = rhs_batch is sol.
            # rhs_batch is expanded to 3D so both numpy and cupy treat it as a
            # batched column vector (core dims (nv,1)) rather than a matrix
            # (core dims (B, nv)), which would mismatch m=nv from the LHS.
            # This problem is well explained in https://github.com/numpy/numpy/issues/26598
            # The RBF matrix is symmetric and positive definite. However,
            # the Cholesky method is not chosen, as in ICON, simply because scipy
            # does not support batched solving of the linear equation,
            # necessitating a Python loop and resulting in poor performance.
            sol = array_ns.linalg.solve(mat_batch, rhs_batch[..., array_ns.newaxis]).squeeze(-1)
            rbf_vec_coeff[j][group_idx + horizontal_start, :nv] = sol

    rbf_vec_coeff = tuple(rbf_vec_coeff)

    # Normalize coefficients
    for j in range(num_zonal_meridional_components):
        rbf_vec_coeff[j][horizontal_start:horizontal_end] /= array_ns.sum(
            nxnx[j] * rbf_vec_coeff[j][horizontal_start:horizontal_end], axis=1
        )[:, array_ns.newaxis]
    return rbf_vec_coeff


def compute_rbf_interpolation_coeffs_cell(
    *,
    cell_center_lat: data_alloc.NDArray,
    cell_center_lon: data_alloc.NDArray,
    cell_center_x: data_alloc.NDArray,
    cell_center_y: data_alloc.NDArray,
    cell_center_z: data_alloc.NDArray,
    edge_center_x: data_alloc.NDArray,
    edge_center_y: data_alloc.NDArray,
    edge_center_z: data_alloc.NDArray,
    edge_normal_x: data_alloc.NDArray,
    edge_normal_y: data_alloc.NDArray,
    edge_normal_z: data_alloc.NDArray,
    rbf_offset: data_alloc.NDArray,
    # TODO(): Can't pass enum as "params" in NumpyFieldsProvider?
    rbf_kernel: int,
    geometry_type: int,
    scale_factor: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
) -> tuple[data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(cell_center_lat)
    zeros = array_ns.zeros(rbf_offset.shape[0], dtype=ta.wpfloat)
    ones = array_ns.ones(rbf_offset.shape[0], dtype=ta.wpfloat)

    return _compute_rbf_interpolation_coeffs(
        element_center_lat=cell_center_lat,
        element_center_lon=cell_center_lon,
        element_center_x=cell_center_x,
        element_center_y=cell_center_y,
        element_center_z=cell_center_z,
        edge_center_x=edge_center_x,
        edge_center_y=edge_center_y,
        edge_center_z=edge_center_z,
        edge_normal_x=edge_normal_x,
        edge_normal_y=edge_normal_y,
        edge_normal_z=edge_normal_z,
        uv=((ones, zeros), (zeros, ones)),
        rbf_offset=rbf_offset,
        rbf_kernel=InterpolationKernel(rbf_kernel),
        geometry_type=icon_grid.GeometryType(geometry_type),
        scale_factor=scale_factor,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        domain_length=domain_length,
        domain_height=domain_height,
    )


def compute_rbf_interpolation_coeffs_edge(
    *,
    edge_lat: data_alloc.NDArray,
    edge_lon: data_alloc.NDArray,
    edge_center_x: data_alloc.NDArray,
    edge_center_y: data_alloc.NDArray,
    edge_center_z: data_alloc.NDArray,
    edge_normal_x: data_alloc.NDArray,
    edge_normal_y: data_alloc.NDArray,
    edge_normal_z: data_alloc.NDArray,
    edge_dual_normal_u: data_alloc.NDArray,
    edge_dual_normal_v: data_alloc.NDArray,
    rbf_offset: data_alloc.NDArray,
    rbf_kernel: int,
    geometry_type: int,
    scale_factor: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
) -> data_alloc.NDArray:
    return _compute_rbf_interpolation_coeffs(
        element_center_lat=edge_lat,
        element_center_lon=edge_lon,
        element_center_x=edge_center_x,
        element_center_y=edge_center_y,
        element_center_z=edge_center_z,
        edge_center_x=edge_center_x,
        edge_center_y=edge_center_y,
        edge_center_z=edge_center_z,
        edge_normal_x=edge_normal_x,
        edge_normal_y=edge_normal_y,
        edge_normal_z=edge_normal_z,
        uv=((edge_dual_normal_u, edge_dual_normal_v),),
        rbf_offset=rbf_offset,
        rbf_kernel=InterpolationKernel(rbf_kernel),
        geometry_type=icon_grid.GeometryType(geometry_type),
        scale_factor=scale_factor,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        domain_length=domain_length,
        domain_height=domain_height,
    )[0]


def compute_rbf_interpolation_coeffs_vertex(
    *,
    vertex_lat: data_alloc.NDArray,
    vertex_lon: data_alloc.NDArray,
    vertex_x: data_alloc.NDArray,
    vertex_y: data_alloc.NDArray,
    vertex_z: data_alloc.NDArray,
    edge_center_x: data_alloc.NDArray,
    edge_center_y: data_alloc.NDArray,
    edge_center_z: data_alloc.NDArray,
    edge_normal_x: data_alloc.NDArray,
    edge_normal_y: data_alloc.NDArray,
    edge_normal_z: data_alloc.NDArray,
    rbf_offset: data_alloc.NDArray,
    rbf_kernel: int,
    geometry_type: int,
    scale_factor: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(vertex_lat)
    zeros = array_ns.zeros(rbf_offset.shape[0], dtype=ta.wpfloat)
    ones = array_ns.ones(rbf_offset.shape[0], dtype=ta.wpfloat)

    return _compute_rbf_interpolation_coeffs(
        element_center_lat=vertex_lat,
        element_center_lon=vertex_lon,
        element_center_x=vertex_x,
        element_center_y=vertex_y,
        element_center_z=vertex_z,
        edge_center_x=edge_center_x,
        edge_center_y=edge_center_y,
        edge_center_z=edge_center_z,
        edge_normal_x=edge_normal_x,
        edge_normal_y=edge_normal_y,
        edge_normal_z=edge_normal_z,
        uv=((ones, zeros), (zeros, ones)),
        rbf_offset=rbf_offset,
        rbf_kernel=InterpolationKernel(rbf_kernel),
        geometry_type=icon_grid.GeometryType(geometry_type),
        scale_factor=scale_factor,
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        domain_length=domain_length,
        domain_height=domain_height,
    )
