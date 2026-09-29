# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Mathematical operations for ICON grid computations.

This package provides mathematical operations organized into the following submodules:

- ``coordinate_transformations``: Conversions between geographical (lat/lon) and cartesian
  coordinates, and between zonal/meridional and cartesian vector components.
- ``derivative``: Vertical derivative computations at cell centers.
- ``distance``: Arc length on spheres and distance/difference operations on torus geometries.
- ``gradient``: Finite difference gradient operators (normal and tangential).
- ``utils``: General-purpose mathematical utility functions (typed sqrt, field inversion, etc.).
- ``operators``: Laplacian (nabla²) and difference operators on cell fields.
- ``projection``: Gnomonic projection and NumPy-based torus operations.
- ``smagorinsky``: Smagorinsky diffusion enhancement factor computation.
- ``vector_operations``: Dot product, cross product, norms and normalization of vectors on cell, edge and vertex fields.
- ``vertical_operations``: Averaging and differencing of adjacent vertical levels.
- ``stencils``: GT4Py program wrappers for operators with domain control.
"""
