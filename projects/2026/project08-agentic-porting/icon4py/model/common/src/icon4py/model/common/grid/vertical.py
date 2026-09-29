# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import enum
import functools
import logging
import math
import pathlib
from typing import Any, Final

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np

import icon4py.model.common.states.metadata as data
import icon4py.model.common.type_alias as ta
from icon4py.model.common import (
    dimension as dims,
    exceptions,
    field_type_aliases as fa,
    topography as topo,
)
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.utils import data_allocation as data_alloc, fortran_config


log = logging.getLogger(__name__)


class Zone(enum.Enum):
    """
    Vertical zone markers to be used to select vertical domain indices.

    The values here are taken from the special indices computed in `VerticalGridParams`.
    """

    TOP = 0
    BOTTOM = 1
    DAMPING = 2
    MOIST = 3
    FLAT = 4


@dataclasses.dataclass(frozen=True)
class Domain:
    """
    Simple data class used to specify a vertical domain such that index lookup and domain specification can be separated.
    """

    dim: gtx.Dimension
    marker: Zone
    offset: int = 0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        assert self.dim.kind == gtx.DimensionKind.VERTICAL
        if self.marker == Zone.TOP:
            assert self.offset >= 0, (
                f"{self.marker} needs to be combined with positive offest, but offset = {self.offset}"
            )


def domain(dim: gtx.Dimension):
    def _domain(marker: Zone):
        assert dim.kind == gtx.DimensionKind.VERTICAL, "Only vertical dimensions are supported"
        return Domain(dim, marker)

    return _domain


@dataclasses.dataclass(frozen=True)
class VerticalGridConfig:
    """
    Contains necessary parameter to configure vertical grid.

    Encapsulates namelist parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.
    """

    #: Number of full levels.
    num_levels: int
    #: Defined as max_lay_thckn in ICON namelist mo_sleve_nml. Maximum thickness of grid cells below top_height_limit_for_maximal_layer_thickness.
    maximal_layer_thickness: Final[ta.wpfloat] = 25000.0
    #: Defined as htop_thcknlimit in ICON namelist mo_sleve_nml. Height below which thickness of grid cells must not exceed maximal_layer_thickness.
    top_height_limit_for_maximal_layer_thickness: Final[ta.wpfloat] = 15000.0
    #: Defined as min_lay_thckn in ICON namelist mo_sleve_nml. Thickness of lowest level grid cells.
    lowest_layer_thickness: Final[ta.wpfloat] = 50.0
    #: Model top height in ICON namelist mo_sleve_nml.
    model_top_height: Final[ta.wpfloat] = 23500.0
    #: Defined in ICON namelist mo_sleve_nml. Height above which coordinate surfaces are flat
    flat_height: Final[ta.wpfloat] = 16000.0
    #: Defined as stretch_fac in ICON namelist mo_sleve_nml. Scaling factor for stretching/squeezing the model layer distribution.
    stretch_factor: Final[ta.wpfloat] = 1.0
    #: Defined as damp_height in ICON namelist nonhydrostatic_nml. Height [m] at which Rayleigh damping of vertical wind starts.
    rayleigh_damping_height: Final[ta.wpfloat] = 45000.0
    #: Defined in ICON namelist nonhydrostatic_nml. Height [m] above which moist physics and advection of cloud and precipitation variables are turned off.
    htop_moist_proc: Final[ta.wpfloat] = 22500.0
    #: file name containing vct_a and vct_b table
    file_path: pathlib.Path | None = None

    # Parameters for setting up the decay function of the topographic signal for
    # SLEVE. decay_scale_1, decay_scale_2 and decay_exp are from mo_sleve_nml.
    #: Decay scale for large-scale topography component
    SLEVE_decay_scale_1: Final[ta.wpfloat] = 4000.0
    #: Decay scale for small-scale topography component
    SLEVE_decay_scale_2: Final[ta.wpfloat] = 2500.0
    #: Exponent for decay function
    SLEVE_decay_exponent: Final[ta.wpfloat] = 1.2
    #: minimum absolute layer thickness 1 for SLEVE coordinates (hardcoded in init_vert_coord, not a namelist parameter)
    _SLEVE_minimum_layer_thickness_1: Final[ta.wpfloat] = 100.0
    #: minimum absolute layer thickness 2 for SLEVE coordinates (hardcoded in init_vert_coord, not a namelist parameter)
    _SLEVE_minimum_layer_thickness_2: Final[ta.wpfloat] = 500.0
    #: minimum relative layer thickness for nominal thicknesses <= _SLEVE_minimum_layer_thickness_1 (hardcoded in init_vert_coord, not a namelist parameter)
    _SLEVE_minimum_relative_layer_thickness_1: Final[ta.wpfloat] = 1.0 / 3.0
    #: minimum relative layer thickness for a nominal thickness of _SLEVE_minimum_layer_thickness_2 (hardcoded in init_vert_coord, not a namelist parameter)
    _SLEVE_minimum_relative_layer_thickness_2: Final[ta.wpfloat] = 0.5

    @classmethod
    def from_fortran_dict(cls, atmo_dict: dict[str, Any], **overrides: Any) -> VerticalGridConfig:
        sleve_nml = atmo_dict["sleve_nml"]
        nonhydrostatic_nml = atmo_dict["nonhydrostatic_nml"]
        run_nml = atmo_dict["run_nml"]
        return cls(
            num_levels=fortran_config.list_to_value(run_nml["num_lev"]),
            maximal_layer_thickness=sleve_nml["max_lay_thckn"],
            top_height_limit_for_maximal_layer_thickness=sleve_nml["htop_thcknlimit"],
            lowest_layer_thickness=sleve_nml["min_lay_thckn"],
            model_top_height=sleve_nml["top_height"],
            flat_height=sleve_nml["flat_height"],
            stretch_factor=sleve_nml["stretch_fac"],
            rayleigh_damping_height=fortran_config.list_to_value(nonhydrostatic_nml["damp_height"]),
            htop_moist_proc=nonhydrostatic_nml["htop_moist_proc"],
            SLEVE_decay_scale_1=sleve_nml["decay_scale_1"],
            SLEVE_decay_scale_2=sleve_nml["decay_scale_2"],
            SLEVE_decay_exponent=sleve_nml["decay_exp"],
            **overrides,
        )


@dataclasses.dataclass(frozen=True)
class VerticalGrid:
    """
    Contains vertical physical parameters defined on the vertical grid derived from vertical grid configuration.

    _vct_a and _vct_b: See docstring of get_vct_a_and_vct_b. Note that the height index starts from the model top.
    _end_index_of_damping_layer: Height index above which Rayleigh damping of vertical wind is applied.
    _start_index_for_moist_physics: Height index above which moist physics and advection of cloud and precipitation variables are turned off.
    _end_index_of_flat_layer: Height index above which coordinate surfaces are flat.
    """

    config: VerticalGridConfig
    vct_a: dataclasses.InitVar[fa.KField[ta.wpfloat]]
    vct_b: dataclasses.InitVar[fa.KField[ta.wpfloat] | None]
    _vct_a: fa.KField[ta.wpfloat] = dataclasses.field(init=False)
    _vct_b: fa.KField[ta.wpfloat] | None = dataclasses.field(init=False)
    _end_index_of_damping_layer: Final[gtx.int32] = dataclasses.field(init=False)
    _start_index_for_moist_physics: Final[gtx.int32] = dataclasses.field(init=False)
    _end_index_of_flat_layer: Final[gtx.int32] = dataclasses.field(init=False)

    def __post_init__(self, vct_a, vct_b):
        object.__setattr__(
            self,
            "_vct_a",
            vct_a,
        )
        object.__setattr__(
            self,
            "_vct_b",
            vct_b,
        )
        vct_a_array = self._vct_a.asnumpy()
        object.__setattr__(
            self,
            "_end_index_of_damping_layer",
            self._determine_damping_height_index(vct_a_array, self.config.rayleigh_damping_height),
        )
        object.__setattr__(
            self,
            "_start_index_for_moist_physics",
            self._determine_start_level_of_moist_physics(vct_a_array, self.config.htop_moist_proc),
        )
        object.__setattr__(
            self,
            "_end_index_of_flat_layer",
            self._determine_end_index_of_flat_layers(vct_a_array, self.config.flat_height),
        )
        log.info(f"computation of moist physics start on layer: {self.kstart_moist}")
        log.info(f"end index of Rayleigh damping layer for w: {self.end_index_of_damping_layer} ")
        log.info(f"number of levels: {self.num_levels}")
        log.info(f"start of flat levels: {self.nflatlev}")

    def __str__(self) -> str:
        vertical_params_properties = ["Model interface height properties:"]
        for key, value in self.metadata_interface_physical_height.items():
            vertical_params_properties.append(f"    {key}: {value}")
        vertical_params_properties.append("Level    Coordinate    Thickness:")
        vct_a_array = self._vct_a.ndarray
        dvct = vct_a_array[:-1] - vct_a_array[1:]
        array_value = [
            f"   0   {vct_a_array[0]:12.3f}             ",
            *(
                f"{k + 1:4d}   {vct_a_array[k + 1]:12.3f} {dvct[k]:12.3f}"
                for k in range(vct_a_array.shape[0] - 1)
            ),
        ]
        array_value[self._end_index_of_flat_layer] += " End of flat layer "
        array_value[self._end_index_of_damping_layer] += " End of damping layer "
        array_value[self._start_index_for_moist_physics] += " Start of moist physics"
        vertical_params_properties.extend(array_value)
        return "\n".join(vertical_params_properties)

    @property
    def metadata_interface_physical_height(self):
        return data.attrs["model_interface_height"]

    @property
    def num_levels(self):
        return self.config.num_levels

    def index(self, domain: Domain) -> gtx.int32:
        match domain.marker:
            case Zone.TOP:
                index = 0
            case Zone.BOTTOM:
                index = self._bottom_level(domain)
            case Zone.MOIST:
                index = self._start_index_for_moist_physics
            case Zone.FLAT:
                index = self._end_index_of_flat_layer
            case Zone.DAMPING:
                index = self._end_index_of_damping_layer
            case _:
                raise exceptions.IconGridError(f"not a valid vertical zone: {domain.marker}")

        index += domain.offset
        assert 0 <= index <= self._bottom_level(domain), (
            f"vertical index {index} outside of grid levels for {domain.dim}"
        )
        return gtx.int32(index)

    def _bottom_level(self, domain: Domain) -> int:
        return self.size(domain.dim)

    @property
    def interface_physical_height(self) -> fa.KField[ta.wpfloat]:
        return self._vct_a

    @functools.cached_property
    def kstart_moist(self) -> gtx.int32:
        """Vertical index for start level of moist physics."""
        return self.index(Domain(dims.KDim, Zone.MOIST))

    @functools.cached_property
    def nflatlev(self) -> gtx.int32:
        """Vertical index for bottom most level at which coordinate surfaces are flat."""
        return gtx.int32(self.index(Domain(dims.KDim, Zone.FLAT)))

    @functools.cached_property
    def end_index_of_damping_layer(self) -> gtx.int32:
        """Vertical index where damping ends."""
        return self.index(Domain(dims.KDim, Zone.DAMPING))

    @property
    def vct_a(self) -> fa.KField:
        return self._vct_a

    @property
    def vct_b(self) -> fa.KField | None:
        return self._vct_b

    def size(self, dim: gtx.Dimension) -> int:
        assert dim.kind == gtx.DimensionKind.VERTICAL, "Only vertical dimensions are supported."
        match dim:
            case dims.KDim:
                return self.config.num_levels
            case dims.KHalfDim:
                return self.config.num_levels + 1
            case _:
                raise ValueError(f"Unknown dimension {dim}.")

    @classmethod
    def _determine_start_level_of_moist_physics(
        cls, vct_a: np.ndarray, top_moist_threshold: ta.wpfloat, nshift_total: int = 0
    ) -> gtx.int32:
        n_levels = vct_a.shape[0]
        interface_height = 0.5 * (vct_a[: n_levels - 1 - nshift_total] + vct_a[1 + nshift_total :])
        return gtx.int32(np.min(np.where(interface_height < top_moist_threshold)[0]).item())

    @classmethod
    def _determine_damping_height_index(
        cls, vct_a: np.ndarray, damping_height: ta.wpfloat
    ) -> gtx.int32:
        assert damping_height >= 0.0, "Damping height must be positive."
        return (
            0
            if damping_height > vct_a[0]
            else gtx.int32(np.argmax(np.where(vct_a >= damping_height)[0]).item())
        )

    @classmethod
    def _determine_end_index_of_flat_layers(
        cls, vct_a: np.ndarray, flat_height: ta.wpfloat
    ) -> gtx.int32:
        assert flat_height >= 0.0, "Flat surface height must be positive."
        return (
            0
            if flat_height > vct_a[0]
            else gtx.int32(np.max(np.where(vct_a >= flat_height)[0]).item())
        )


def _read_vct_a_and_vct_b_from_file(
    file_path: pathlib.Path, num_levels: int, allocator: gtx_typing.Allocator
) -> tuple[fa.KField, fa.KField]:
    """
    Read vct_a and vct_b from a file.
    The file format should be as follows (the same format used for icon):
        #    k     vct_a(k) [Pa]   vct_b(k) []
        1  12000.0000000000  0.0000000000
        2  11800.0000000000  0.0000000000
        3  11600.0000000000  0.0000000000
        4  11400.0000000000  0.0000000000
        5  11200.0000000000  0.0000000000
        ...

    Args:
        file_path: Path to the vertical grid file
        num_levels: number of cell levels
        allocator: GT4Py field allocator
    Returns:  one dimensional vct_a and vct_b arrays.
    """
    num_levels_plus_one = num_levels + 1
    vct_a = np.zeros(num_levels_plus_one, dtype=ta.wpfloat)
    vct_b = np.zeros(num_levels_plus_one, dtype=ta.wpfloat)
    try:
        with file_path.open() as vertical_grid_file:
            # skip the first line that contains titles
            vertical_grid_file.readline()
            for k in range(num_levels_plus_one):
                grid_content = vertical_grid_file.readline().split()
                vct_a[k] = ta.wpfloat(grid_content[1])
                vct_b[k] = ta.wpfloat(grid_content[2])
    except OSError as err:
        raise FileNotFoundError(
            f"Vertical coord table file {file_path} could not be read."
        ) from err
    except IndexError as err:
        raise IndexError(
            f"The number of levels in the vertical coord table file {file_path} is possibly not the same as {num_levels_plus_one} or data is missing at {k}-th line."
        ) from err
    except ValueError as err:
        raise ValueError(f"data is not float at {k}-th line.") from err
    return gtx.as_field((dims.KDim,), vct_a, allocator=allocator), gtx.as_field(
        (dims.KDim,), vct_b, allocator=allocator
    )


def _compute_vct_a_and_vct_b(  # noqa: PLR0912 [too-many-branches]
    vertical_config: VerticalGridConfig, allocator: gtx_typing.Allocator
) -> tuple[fa.KField, fa.KField]:
    """
    Compute vct_a and vct_b.

    When the thickness of lowest level grid cells is larger than 0.01:
        vct_a[k] = H (2/pi arccos((k/N)^s) )^d      N = num_levels      s = stretch_factor in vertical configuration
        d is such that the lowest level grid cell thickness is equal to lowest_layer_thickness in vertical configuration.
        d = ln(lowest_layer_thickness/H) / ln(2/pi arccos(((N-1)/N)^s) )
        When top_height_limit_for_maximal_layer_thicknessis larger than model_top_height, the final model top height will be adjusted if there are layers thicker than maximal_layer_thickness.
        Otherwise, layers above top_height_limit_for_maximal_layer_thickness will be modified such that h_0 is equal to model_top_height.
         ------- 0     model top, z_0 = Z, h_0 = H, Z is the model top height after layer thickness > m is reduced to m, and H is model_top_height.
        ...
         ------- k-2
        k-2
         ------- k-1
        k-1
         ------- k
        k            ------ top_height_limit_for_maximal_layer_thickness
         ------- k+1
        k+1
         ------- k+2
        k+2
         ------- k+3
        h_k..N+1 = z_k..N+1,  h and z are the same at levels k to N+1.
        h_k-1 = h_k + m + (dh_k-1 - m) * a, the stretch factor aims to stretch layer thickness different from m. dh_k-1 = dh_k-1+shift, shift is equal to (k + s) - k, where k+s is the lowest layer index that needs to be modified, this is to prevent sudden jump.
        h_0 = z_k + a * sum_i=0^k-1 (dh_i) + k * (1 - a) * m = H, h is vct_a before layer thickness > m is reduced to m, z is modified_vct_a after layer thickness > m is reduced, m = maximal_layer_thickness. dh_i is the layer thickness of vct_a.
        z_0 = z_k + sum_i=0^k-1 (dh_i) = Z.
        Hence, the stretch factor a = (H - z_k - m * k) / (Z - z_k - m * k).

        An additional smoothing is performed for levels 2, 3, 4, ..., k if modified_vct_a[0] after additional smoothing is still the same as model_top_height.

        THERE IS A WARNING MESSAGE IF vct_a[0] AND model_top_height ARE NOT EQUAL.

    When the thickness of lowest level grid cells is smaller than or equal to 0.01:
        a uniform vertical grid is generated based on model top height and number of levels.

    Args:
        vertical_config: Vertical grid configuration
        backend: GT4Py backend
    Returns:  one dimensional (dims.KDim) vct_a and vct_b gt4py fields.
    """
    num_levels_plus_one = vertical_config.num_levels + 1
    if vertical_config.lowest_layer_thickness > 0.01:
        vct_a_exponential_factor = np.log(
            vertical_config.lowest_layer_thickness / vertical_config.model_top_height
        ) / np.log(
            2.0
            / math.pi
            * np.arccos(
                ta.wpfloat(vertical_config.num_levels - 1) ** vertical_config.stretch_factor
                / ta.wpfloat(vertical_config.num_levels) ** vertical_config.stretch_factor
            )
        )

        vct_a = (
            vertical_config.model_top_height
            * (
                2.0
                / math.pi
                * np.arccos(
                    np.arange(vertical_config.num_levels + 1, dtype=ta.wpfloat)
                    ** vertical_config.stretch_factor
                    / ta.wpfloat(vertical_config.num_levels) ** vertical_config.stretch_factor
                )
            )
            ** vct_a_exponential_factor
        )

        if (
            2.0 * vertical_config.lowest_layer_thickness
            < vertical_config.maximal_layer_thickness
            < 0.5 * vertical_config.top_height_limit_for_maximal_layer_thickness
        ):
            layer_thickness = vct_a[: vertical_config.num_levels] - vct_a[1:]
            lowest_level_exceeding_limit = np.max(
                np.where(layer_thickness > vertical_config.maximal_layer_thickness)
            )
            modified_vct_a = np.zeros(num_levels_plus_one, dtype=ta.wpfloat)
            lowest_level_unmodified_thickness = 0
            shifted_levels = 0
            for k in range(vertical_config.num_levels - 1, -1, -1):
                if (
                    modified_vct_a[k + 1]
                    < vertical_config.top_height_limit_for_maximal_layer_thickness
                ):
                    modified_vct_a[k] = modified_vct_a[k + 1] + np.minimum(
                        vertical_config.maximal_layer_thickness, layer_thickness[k]
                    )
                elif lowest_level_unmodified_thickness == 0:
                    lowest_level_unmodified_thickness = k + 1
                    shifted_levels = max(
                        0, lowest_level_exceeding_limit - lowest_level_unmodified_thickness
                    )
                    modified_vct_a[k] = modified_vct_a[k + 1] + layer_thickness[k + shifted_levels]
                else:
                    modified_vct_a[k] = modified_vct_a[k + 1] + layer_thickness[k + shifted_levels]

            stretchfac = (
                1.0
                if shifted_levels == 0
                else (
                    vct_a[0]
                    - modified_vct_a[lowest_level_unmodified_thickness]
                    - ta.wpfloat(lowest_level_unmodified_thickness)
                    * vertical_config.maximal_layer_thickness
                )
                / (
                    modified_vct_a[0]
                    - modified_vct_a[lowest_level_unmodified_thickness]
                    - ta.wpfloat(lowest_level_unmodified_thickness)
                    * vertical_config.maximal_layer_thickness
                )
            )

            for k in range(vertical_config.num_levels - 1, -1, -1):
                if vct_a[k + 1] < vertical_config.top_height_limit_for_maximal_layer_thickness:
                    vct_a[k] = vct_a[k + 1] + np.minimum(
                        vertical_config.maximal_layer_thickness, layer_thickness[k]
                    )
                else:
                    vct_a[k] = (
                        vct_a[k + 1]
                        + vertical_config.maximal_layer_thickness
                        + (
                            layer_thickness[k + shifted_levels]
                            - vertical_config.maximal_layer_thickness
                        )
                        * stretchfac
                    )

            # Try to apply additional smoothing on the stretching factor above the constant-thickness layer
            if stretchfac != 1.0 and lowest_level_exceeding_limit < vertical_config.num_levels - 4:
                for k in range(vertical_config.num_levels - 1, -1, -1):
                    if (
                        modified_vct_a[k + 1]
                        < vertical_config.top_height_limit_for_maximal_layer_thickness
                    ):
                        modified_vct_a[k] = vct_a[k]
                    else:
                        modified_layer_thickness = np.minimum(
                            1.025 * (vct_a[k] - vct_a[k + 1]),
                            1.025
                            * (
                                modified_vct_a[lowest_level_exceeding_limit + 1]
                                - modified_vct_a[lowest_level_exceeding_limit + 2]
                            )
                            / (
                                modified_vct_a[lowest_level_exceeding_limit + 2]
                                - modified_vct_a[lowest_level_exceeding_limit + 3]
                            )
                            * (modified_vct_a[k + 1] - modified_vct_a[k + 2]),
                        )
                        modified_vct_a[k] = np.minimum(
                            vct_a[k], modified_vct_a[k + 1] + modified_layer_thickness
                        )
                if modified_vct_a[0] == vct_a[0]:
                    vct_a[0:2] = modified_vct_a[0:2]
                    vct_a[lowest_level_unmodified_thickness + 1 : vertical_config.num_levels] = (
                        modified_vct_a[
                            lowest_level_unmodified_thickness + 1 : vertical_config.num_levels
                        ]
                    )
                    vct_a[2 : lowest_level_unmodified_thickness + 1] = 0.5 * (
                        modified_vct_a[1:lowest_level_unmodified_thickness]
                        + modified_vct_a[3 : lowest_level_unmodified_thickness + 2]
                    )
    else:
        vct_a = (
            vertical_config.model_top_height
            * (
                ta.wpfloat(vertical_config.num_levels)
                - np.arange(num_levels_plus_one, dtype=ta.wpfloat)
            )
            / ta.wpfloat(vertical_config.num_levels)
        )
    vct_b = np.exp(-vct_a / 5000.0)

    if not np.allclose(vct_a[0], vertical_config.model_top_height):
        log.warning(
            f" Warning. vct_a[0], {vct_a[0]}, is not equal to model top height, {vertical_config.model_top_height}, of vertical configuration. Please consider changing the vertical setting."
        )

    return gtx.as_field((dims.KDim,), vct_a, allocator=allocator), gtx.as_field(
        (dims.KDim,), vct_b, allocator=allocator
    )


def get_vct_a_and_vct_b(
    vertical_config: VerticalGridConfig,
    allocator: gtx_typing.Allocator,
) -> tuple[fa.KField, fa.KField]:
    """
    get vct_a and vct_b.
    vct_a is an array that contains the height of grid interfaces (or half levels) from model surface to model top, before terrain-following coordinates are applied.
    vct_b is an array that is used to initialize vertical wind speed above surface by a prescribed vertical profile when the surface vertical wind is given.
    It is also used to modify the initial vertical wind speed above surface according to a prescribed vertical profile by linearly merging the surface vertical wind with the existing vertical wind.
    See init_w and adjust_w in mo_nh_init_utils.f90.

    When file_name is given in vertical_config, it will read both vct_a and vct_b from that file. Otherwise, they are analytically derived based on vertical configuration.

    Args:
        vertical_config: Vertical grid configuration
        backend: GT4Py backend
    Returns:  one dimensional (dims.KDim) vct_a and vct_b gt4py fields.
    """

    return (
        _read_vct_a_and_vct_b_from_file(
            vertical_config.file_path, vertical_config.num_levels, allocator
        )
        if vertical_config.file_path
        else _compute_vct_a_and_vct_b(vertical_config, allocator)
    )


def _compute_SLEVE_coordinate_from_vcta_and_topography(
    *,
    vct_a: data_alloc.NDArray,
    topography: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    geofac_n2s: data_alloc.NDArray,
    c2e2co: data_alloc.NDArray,
    nflatlev: int,
    model_top_height: ta.wpfloat,
    SLEVE_decay_scale_1: ta.wpfloat,
    SLEVE_decay_exponent: ta.wpfloat,
    SLEVE_decay_scale_2: ta.wpfloat,
    exchange: decomposition.ExchangeRuntime,
) -> data_alloc.NDArray:
    """
    Compute the 3D vertical coordinate field using the SLEVE coordinate
    https://doi.org/10.1175/1520-0493(2002)130%3C2459:ANTFVC%3E2.0.CO;2

    This is the same as vct_a in the flat levels (above nflatlev).
    Below it is vct_a corrected by smothed and decaying topography such that it
    blends smothly into the surface layer at num_lev + 1 which is the
    topography.
    """
    array_ns = data_alloc.array_namespace(vct_a)
    num_cells = cell_areas.shape[0]
    num_levels = vct_a.shape[0] - 1

    def _decay_func(
        vct_a: data_alloc.NDArray,
        model_top_height: ta.wpfloat,
        decay_scale: ta.wpfloat,
        decay_exponent: ta.wpfloat,
    ) -> data_alloc.NDArray:
        return array_ns.sinh(
            (model_top_height / decay_scale) ** decay_exponent
            - (vct_a / decay_scale) ** decay_exponent
        ) / array_ns.sinh((model_top_height / decay_scale) ** decay_exponent)

    smoothed_topography = topo.smooth_topography(
        topography=topography,
        cell_areas=cell_areas,
        geofac_n2s=geofac_n2s,
        c2e2co=c2e2co,
        exchange=exchange,
    )

    vertical_coordinate = array_ns.zeros((num_cells, num_levels + 1), dtype=ta.wpfloat)
    vertical_coordinate[:, num_levels] = topography

    # Small-scale topography (i.e. full topo - smooth topo)
    small_scale_topography = topography - smoothed_topography

    k = range(nflatlev + 1)
    vertical_coordinate[:, k] = vct_a[k]

    k = range(nflatlev + 1, num_levels)
    # Scaling factors for large-scale and small-scale topography
    z_fac1 = _decay_func(
        vct_a[k],
        model_top_height,
        SLEVE_decay_scale_1,
        SLEVE_decay_exponent,
    )
    z_fac2 = _decay_func(
        vct_a[k],
        model_top_height,
        SLEVE_decay_scale_2,
        SLEVE_decay_exponent,
    )
    vertical_coordinate[:, k] = (
        vct_a[k][array_ns.newaxis, :]
        + smoothed_topography[:, array_ns.newaxis] * z_fac1
        + small_scale_topography[:, array_ns.newaxis] * z_fac2
    )

    return vertical_coordinate


def _check_and_correct_layer_thickness(
    *,
    vertical_coordinate: data_alloc.NDArray,
    vct_a: data_alloc.NDArray,
    SLEVE_minimum_layer_thickness_1: ta.wpfloat,
    SLEVE_minimum_relative_layer_thickness_1: ta.wpfloat,
    SLEVE_minimum_layer_thickness_2: ta.wpfloat,
    SLEVE_minimum_relative_layer_thickness_2: ta.wpfloat,
    lowest_layer_thickness: ta.wpfloat,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(vertical_coordinate)
    num_cells = vertical_coordinate.shape[0]
    num_levels = vertical_coordinate.shape[1] - 1
    ktop_thicklimit = array_ns.asarray(num_cells * [num_levels], dtype=int)
    # Ensure that layer thicknesses are not too small; this would potentially
    # cause instabilities in vertical advection
    for k in reversed(range(num_levels)):
        delta_vct_a = vct_a[k] - vct_a[k + 1]
        if delta_vct_a < SLEVE_minimum_layer_thickness_1:
            # limit layer thickness to SLEVE_minimum_relative_layer_thickness_1 times its nominal value
            minimum_layer_thickness = SLEVE_minimum_relative_layer_thickness_1 * delta_vct_a
        elif delta_vct_a < SLEVE_minimum_layer_thickness_2:
            # limitation factor changes from SLEVE_minimum_relative_layer_thickness_1 to SLEVE_minimum_relative_layer_thickness_2
            layer_thickness_adjustment_factor = (
                (SLEVE_minimum_layer_thickness_2 - delta_vct_a)
                / (SLEVE_minimum_layer_thickness_2 - SLEVE_minimum_layer_thickness_1)
            ) ** 2
            minimum_layer_thickness = (
                SLEVE_minimum_relative_layer_thickness_1 * layer_thickness_adjustment_factor
                + SLEVE_minimum_relative_layer_thickness_2
                * (1.0 - layer_thickness_adjustment_factor)
            ) * delta_vct_a
        else:
            # limitation factor decreases again
            minimum_layer_thickness = (
                SLEVE_minimum_relative_layer_thickness_2
                * SLEVE_minimum_layer_thickness_2
                * (delta_vct_a / SLEVE_minimum_layer_thickness_2) ** (1.0 / 3.0)
            )

        minimum_layer_thickness = max(minimum_layer_thickness, min(50, lowest_layer_thickness))

        # Ensure that the layer thickness is not too small, if so fix it and
        # save the layer index
        cell_ids = array_ns.argwhere(
            vertical_coordinate[:, k + 1] + minimum_layer_thickness > vertical_coordinate[:, k]
        )
        vertical_coordinate[cell_ids, k] = (
            vertical_coordinate[cell_ids, k + 1] + minimum_layer_thickness
        )
        ktop_thicklimit[cell_ids] = k

    # Smooth layer thickness ratios in the transition layer of columns where the
    # thickness limiter has been active (exclude lowest and highest layers)
    cell_ids = array_ns.argwhere((ktop_thicklimit <= num_levels - 3) & (ktop_thicklimit >= 3))
    if cell_ids.size > 0:
        delta_z1 = (
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] + 1]
            - vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] + 2]
        )
        delta_z2 = (
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] - 3]
            - vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] - 2]
        )
        stretching_factor = (delta_z2 / delta_z1) ** 0.25
        delta_z3 = (
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] - 2]
            - vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] + 1]
        ) / (stretching_factor * (1.0 + stretching_factor * (1.0 + stretching_factor)))
        vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids]] = array_ns.maximum(
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids]],
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] + 1]
            + delta_z3 * stretching_factor,
        )
        vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] - 1] = array_ns.maximum(
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids] - 1],
            vertical_coordinate[cell_ids, ktop_thicklimit[cell_ids]]
            + delta_z3 * stretching_factor**2,
        )

    # Check if ktop_thicklimit is sufficiently far away from the model top
    if not array_ns.all(ktop_thicklimit > 2):
        if num_levels > 6:
            raise exceptions.InvalidConfigError(
                f"Model top is too low and num_levels, {num_levels}, > 6."
            )
        else:
            log.warning(f"Model top is too low. But num_levels, {num_levels}, <= 6. ")

    return vertical_coordinate


def _check_flatness_of_flat_level(
    vertical_coordinate: data_alloc.NDArray,
    vct_a: data_alloc.NDArray,
    nflatlev: int,
) -> None:
    # Check if level nflatlev is still flat
    if not (vertical_coordinate[:, max(0, nflatlev - 1)] == vct_a[max(0, nflatlev - 1)]).all():
        raise exceptions.InvalidComputationError("Level nflatlev is not flat")


def compute_vertical_coordinate(
    *,
    vct_a: data_alloc.NDArray,
    topography: data_alloc.NDArray,
    cell_areas: data_alloc.NDArray,
    geofac_n2s: data_alloc.NDArray,
    c2e2co: data_alloc.NDArray,
    nflatlev: int,
    model_top_height: ta.wpfloat,
    SLEVE_decay_scale_1: ta.wpfloat,
    SLEVE_decay_exponent: ta.wpfloat,
    SLEVE_decay_scale_2: ta.wpfloat,
    SLEVE_minimum_layer_thickness_1: ta.wpfloat,
    SLEVE_minimum_relative_layer_thickness_1: ta.wpfloat,
    SLEVE_minimum_layer_thickness_2: ta.wpfloat,
    SLEVE_minimum_relative_layer_thickness_2: ta.wpfloat,
    lowest_layer_thickness: ta.wpfloat,
    exchange: decomposition.ExchangeRuntime,
) -> data_alloc.NDArray:
    """
    Compute the (Cell, K) vertical coordinate field starting from the
    "flat/uniform/aquaplanet" vertical coordinate.

    Args:
        vct_a: Vertical coordinate with flat topography.
        topography: Topography field.
        cell_areas: Cell areas field.
        geofac_n2s: Coefficients for nabla2 computation.
        grid: Grid object.
        vertical_geometry: Vertical grid object.
        backend: Backend to use for computations.

    Returns:
        The (Cell, K) vertical coordinate field.

    Raises:
        exceptions.InvalidComputationError: If level nflatlev is not flat.
        exceptions.InvalidConfigError: If model top is too low and num_levels > 6.
    """

    vertical_coordinates_on_half_levels = _compute_SLEVE_coordinate_from_vcta_and_topography(
        vct_a=vct_a,
        topography=topography,
        cell_areas=cell_areas,
        geofac_n2s=geofac_n2s,
        c2e2co=c2e2co,
        nflatlev=nflatlev,
        model_top_height=model_top_height,
        SLEVE_decay_scale_1=SLEVE_decay_scale_1,
        SLEVE_decay_exponent=SLEVE_decay_exponent,
        SLEVE_decay_scale_2=SLEVE_decay_scale_2,
        exchange=exchange,
    )

    vertical_coordinate = _check_and_correct_layer_thickness(
        vertical_coordinate=vertical_coordinates_on_half_levels,
        vct_a=vct_a,
        SLEVE_minimum_layer_thickness_1=SLEVE_minimum_layer_thickness_1,
        SLEVE_minimum_relative_layer_thickness_1=SLEVE_minimum_relative_layer_thickness_1,
        SLEVE_minimum_layer_thickness_2=SLEVE_minimum_layer_thickness_2,
        SLEVE_minimum_relative_layer_thickness_2=SLEVE_minimum_relative_layer_thickness_2,
        lowest_layer_thickness=lowest_layer_thickness,
    )

    _check_flatness_of_flat_level(
        vertical_coordinate=vertical_coordinates_on_half_levels,
        vct_a=vct_a,
        nflatlev=nflatlev,
    )

    return vertical_coordinate
