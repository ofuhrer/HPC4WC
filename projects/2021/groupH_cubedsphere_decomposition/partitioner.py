from typing import Tuple, Callable, Iterable, Optional, Union, cast, Sequence
import copy
import abc
import functools
import dataclasses
from . import constants, utils
from .constants import (
    NORTH,
    SOUTH,
    WEST,
    EAST,
    NORTHWEST,
    NORTHEAST,
    SOUTHWEST,
    SOUTHEAST,
)
import numpy as np
from . import boundary as bd
from .quantity import QuantityMetadata

BOUNDARY_CACHE_SIZE = None


__all__ = ["TilePartitioner", "CubedSpherePartitioner", "get_tile_index"]


def get_tile_index(rank: int, total_ranks: int) -> int:
    """
    Returns the zero-indexed tile number, given a rank and total number of ranks.
    """
    if total_ranks % 6 != 0:
        raise ValueError(f"total_ranks {total_ranks} is not evenly divisible by 6")
    ranks_per_tile = total_ranks // 6
    return rank // ranks_per_tile


def get_tile_number(tile_rank: int, total_ranks: int) -> int:
    """Deprecated: use get_tile_index.
    
    Returns the tile number for a given rank and total number of ranks.
    """
    FutureWarning(
        "get_tile_number will be removed in a later version, "
        "use get_tile_index(rank, total_ranks) + 1 instead"
    )
    if total_ranks % 6 != 0:
        raise ValueError(f"total_ranks {total_ranks} is not evenly divisible by 6")
    ranks_per_tile = total_ranks // 6
    return tile_rank // ranks_per_tile + 1


class Partitioner(abc.ABC):
    @abc.abstractmethod
    def global_extent(self, rank_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a full tile representation for the given dimensions.

        Args:
            metadata: quantity metadata

        Returns:
            extent: shape of full tile representation
        """
        pass

    @abc.abstractmethod
    def subtile_slice(
        self,
        rank: int,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
        overlap: bool = False,
    ) -> Tuple[Union[int, slice], ...]:
        """Return the subtile slice of a given rank on an array.

        Global refers to the domain being partitioned. For example, for a partitioning
        of a tile, the tile would be the "global" domain.

        Args:
            rank: the rank of the process
            global_dims: dimensions of the global quantity being partitioned
            global_extent: extent of the global quantity being partitioned
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            subtile_slice: the slice of the global compute domain corresponding
                to the subtile compute domain
        """
        pass

    @abc.abstractmethod
    def subtile_extent(self, global_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a single rank representation for the given dimensions."""
        pass

    @abc.abstractproperty
    def total_ranks(self) -> int:
        pass


class TilePartitioner(Partitioner):
    def __init__(
        self, layout: Tuple[int, int],
    ):
        """Create an object for fv3gfs tile decomposition.
        """
        self.layout = layout

    @classmethod
    def from_namelist(cls, namelist):
        """Initialize a TilePartitioner from a Fortran namelist.

        Args:
            namelist (dict): the Fortran namelist
        """
        return cls(layout=namelist["fv_core_nml"]["layout"])

    def subtile_index(self, rank: int) -> Tuple[int, int, int, int]:
        """Return the (y, x) subtile position of a given rank as an integer number of subtiles."""
        return subtile_index(rank, self.total_ranks, self.layout)

    @property
    def total_ranks(self) -> int:
        return self.layout[0] * self.layout[1]

    def global_extent(self, rank_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a full tile representation for the given dimensions.

        Args:
            metadata: quantity metadata

        Returns:
            extent: shape of full tile representation
        """
        return tile_extent_from_rank_metadata(
            rank_metadata.dims, rank_metadata.extent, self.layout
        )

    def subtile_extent(self, global_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a single rank representation for the given dimensions."""
        return rank_extent_from_tile_metadata(
            global_metadata.dims, global_metadata.extent, self.layout
        )

    def subtile_slice(
        self,
        rank: int,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
        overlap: bool = False,
    ) -> Tuple[slice, ...]:
        """Return the subtile slice of a given rank on an array.

        Global refers to the domain being partitioned. For example, for a partitioning
        of a tile, the tile would be the "global" domain.

        Args:
            rank: the rank of the process
            global_dims: dimensions of the global quantity being partitioned
            global_extent: extent of the global quantity being partitioned
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            subtile_slice: the slice of the global compute domain corresponding
                to the subtile compute domain
        """
        return subtile_slice(
            global_dims,
            global_extent,
            self.layout,
            self.subtile_index(rank),
            overlap=overlap,
        )

    def on_tile_top(self, rank: int) -> bool:
        return on_tile_top(self.subtile_index(rank), self.layout)

    def on_tile_bottom(self, rank: int) -> bool:
        return on_tile_bottom(self.subtile_index(rank))

    def on_tile_left(self, rank: int) -> bool:
        return on_tile_left(self.subtile_index(rank))

    def on_tile_right(self, rank: int) -> bool:
        return on_tile_right(self.subtile_index(rank), self.layout)

    def boundary(self, boundary_type: int, rank: int) -> Optional[bd.SimpleBoundary]:
        """Returns a boundary of the requested type for a given rank.

        Target ranks will be on the same tile as the given rank, wrapping around as
        in a doubly-periodic boundary condition.

        Args:
            boundary_type: the type of boundary
            rank: the processor rank

        Returns:
            boundary
        """
        boundary = copy.copy(self._cached_boundary(boundary_type, rank))
        return boundary

    @functools.lru_cache(maxsize=BOUNDARY_CACHE_SIZE)
    def _cached_boundary(
        self, boundary_type: int, rank: int
    ) -> Optional[bd.SimpleBoundary]:
        boundary = {
            WEST: self._left_edge,
            EAST: self._right_edge,
            NORTH: self._top_edge,
            SOUTH: self._bottom_edge,
            NORTHWEST: self._top_left_corner,
            NORTHEAST: self._top_right_corner,
            SOUTHWEST: self._bottom_left_corner,
            SOUTHEAST: self._bottom_right_corner,
        }[boundary_type](rank)
        return boundary

    def _left_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_left(rank):
            #switched indeces of layout
            to_rank = rank + self.layout[0] - 1
        else:
            to_rank = rank - 1
        return bd.SimpleBoundary(
            boundary_type=constants.WEST,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _right_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_right(rank):
            #switched indeces of layout
            to_rank = rank - self.layout[0] + 1
        else:
            to_rank = rank + 1
        return bd.SimpleBoundary(
            boundary_type=constants.EAST,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _top_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_top(rank):
            #switched indeces of layout
            to_rank = rank - (self.layout[1] - 1) * self.layout[0]
        else:
            to_rank = rank + self.layout[1]
        return bd.SimpleBoundary(
            boundary_type=constants.NORTH,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _bottom_edge(self, rank: int) -> bd.SimpleBoundary:
        if self.on_tile_bottom(rank):
            #switched indeces of layout
            to_rank = rank + (self.layout[1] - 1) * self.layout[0]
        else:
            to_rank = rank - self.layout[1]
        return bd.SimpleBoundary(
            boundary_type=constants.SOUTH,
            from_rank=rank,
            to_rank=to_rank,
            n_clockwise_rotations=0,
        )

    def _top_left_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
        return _get_corner(constants.NORTHWEST, rank, self._left_edge, self._top_edge)

    def _top_right_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
        return _get_corner(constants.NORTHEAST, rank, self._right_edge, self._top_edge)

    def _bottom_left_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
        return _get_corner(
            constants.SOUTHWEST, rank, self._left_edge, self._bottom_edge
        )

    def _bottom_right_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
        return _get_corner(
            constants.SOUTHEAST, rank, self._right_edge, self._bottom_edge
        )

    def fliplr_rank(self, rank: int) -> int:
        return fliplr_subtile_rank(rank, self.layout)

    def rotate_rank(self, rank: int, n_clockwise_rotations: int) -> int:
        return rotate_subtile_rank(rank, self.layout, n_clockwise_rotations)


def _get_corner(
    boundary_type: int,
    rank: int,
    edge_func_1: Callable[[int], bd.Boundary],
    edge_func_2: Callable[[int], bd.Boundary],
):
    edge_1 = edge_func_1(rank)
    edge_2 = edge_func_2(edge_1.to_rank)
    rotations = edge_1.n_clockwise_rotations + edge_2.n_clockwise_rotations
    return bd.SimpleBoundary(
        boundary_type=boundary_type,
        from_rank=rank,
        to_rank=edge_2.to_rank,
        n_clockwise_rotations=rotations,
    )


class CubedSpherePartitioner(Partitioner):
    def __init__(self, tile: TilePartitioner):
        """Create an object for fv3gfs cubed-sphere domain decomposition.
        
        Args:
            tile: partitioner for the cube faces
        """
        if not isinstance(tile, TilePartitioner):
            raise TypeError("tile must be a TilePartitioner")
        self.tile = tile

    @classmethod
    def from_namelist(cls, namelist):
        """Initialize a CubedSpherePartitioner from a Fortran namelist.

        Args:
            namelist (dict): the Fortran namelist
        """
        return cls(TilePartitioner.from_namelist(namelist))

    def _ensure_square_layout(self) -> None:
        if not self.tile.layout[0] == self.tile.layout[1]:
            raise NotImplementedError("currently only square layouts are supported")

    def tile_index(self, rank: int) -> int:
        """Returns the tile index of a given rank"""
        return get_tile_index(rank, self.total_ranks)

    def tile_root_rank(self, rank: int) -> int:
        """Returns the lowest rank on the same tile as a given rank."""
        return self.tile.total_ranks * (rank // self.tile.total_ranks)

    @property
    def layout(self) -> Tuple[int, int]:
        return self.tile.layout

    @property
    def total_ranks(self) -> int:
        """the number of ranks on the cubed sphere"""
        return 6 * self.tile.total_ranks

    def boundary(self, boundary_type: int, rank: int) -> Optional[bd.SimpleBoundary]:
        """Returns a boundary of the requested type for a given rank, or None.

        On tile corners, the boundary across that corner does not exist.

        Args:
            boundary_type: the type of boundary
            rank: the processor rank

        Returns:
            boundary
        """
        boundary = copy.copy(self._cached_boundary(boundary_type, rank))
        return boundary

    @functools.lru_cache(maxsize=BOUNDARY_CACHE_SIZE)
    def _cached_boundary(
        self, boundary_type: int, rank: int
    ) -> Optional[bd.SimpleBoundary]:
        boundary = {
            WEST: self._left_edge,
            EAST: self._right_edge,
            NORTH: self._top_edge,
            SOUTH: self._bottom_edge,
            NORTHWEST: self._top_left_corner,
            NORTHEAST: self._top_right_corner,
            SOUTHWEST: self._bottom_left_corner,
            SOUTHEAST: self._bottom_right_corner,
        }[boundary_type](rank)
        if boundary is not None:
            boundary.to_rank = boundary.to_rank % self.total_ranks
        return boundary

    def _left_edge(self, rank: int) -> bd.SimpleBoundary:
        boundary_list = []
        if self.tile.on_tile_left(rank):
            if is_even(self.tile_index(rank)):
                to_root_rank = self.tile_root_rank((rank - 2 * (self.tile.total_ranks))%self.total_ranks)
                to_ranks_seq = self.lr_to_ranks(rank, to_root_rank)
                for n in range(0,len(to_ranks_seq)):
                    boundary_list.append(bd.SimpleBoundary(
                        boundary_type=constants.WEST,
                        from_rank=rank,
                        to_rank=int(to_ranks_seq[n]%self.total_ranks),
                        n_clockwise_rotations=1,
                    ))
            else:
                boundary_list.append(cast(bd.SimpleBoundary, self.tile.boundary(WEST, rank=rank)))
                boundary_list[0].to_rank -= self.tile.total_ranks
                if boundary_list[0].to_rank > self.total_ranks:
                    boundary_list[0].to_rank = boundary_list[0].to_rank%self.total_ranks
        else:
            boundary_list.append(cast(bd.SimpleBoundary, self.tile.boundary(WEST, rank=rank)))
        return boundary_list

    def _right_edge(self, rank: int) -> bd.SimpleBoundary:
        boundary_list = []
        if self.tile.on_tile_right(rank):
            if not is_even(self.tile_index(rank)):
                to_root_rank = self.tile_root_rank(rank) + (2*self.tile.total_ranks)
                to_ranks_seq = self.lr_to_ranks(rank, to_root_rank)
                for n in range(0,len(to_ranks_seq)):
                    boundary_list.append(bd.SimpleBoundary(
                        boundary_type=constants.EAST,
                        from_rank=rank,
                        to_rank=int(to_ranks_seq[n]%self.total_ranks),
                        n_clockwise_rotations=1,
                    ))
            else:
                boundary_list.append(cast(bd.SimpleBoundary, self.tile.boundary(EAST, rank=rank)))
                boundary_list[0].to_rank += self.tile.total_ranks
                if boundary_list[0].to_rank > self.total_ranks:
                    boundary_list[0].to_rank = boundary_list[0].to_rank%self.total_ranks
        else:
            boundary_list.append(cast(bd.SimpleBoundary, self.tile.boundary(EAST, rank=rank)))
        return boundary_list

    def _top_edge(self, rank: int) -> bd.SimpleBoundary:
        boundary_list = []
        if self.tile.on_tile_top(rank):
            if is_even(self.tile_index(rank)):
                to_root_rank = (self.tile_index(rank) + 2) * self.tile.total_ranks
                to_ranks_seq = self.ul_to_ranks(rank, to_root_rank)
                for n in range(0,len(to_ranks_seq)):
                    boundary_list.append(bd.SimpleBoundary(
                        boundary_type=constants.NORTH,
                        from_rank=rank,
                        to_rank=int(to_ranks_seq[n]%self.total_ranks),
                        n_clockwise_rotations=3,
                    ))
            else:
                boundary_list.append(cast(bd.SimpleBoundary, self.tile.boundary(NORTH, rank)))
                boundary_list[0].to_rank += self.tile.total_ranks 
                if boundary_list[0].to_rank > self.total_ranks:
                    boundary_list[0].to_rank = boundary_list[0].to_rank%self.total_ranks
        else:
            boundary_list.append(cast(bd.SimpleBoundary, self.tile.boundary(NORTH, rank=rank)))
        return boundary_list

    def _bottom_edge(self, rank: int) -> bd.SimpleBoundary:
        boundary_list = []
        if self.tile.on_tile_bottom(rank) and not is_even(self.tile_index(rank)):
            to_root_rank = self.tile_root_rank(rank) + 4*self.tile.total_ranks
            to_root_rank = to_root_rank%self.total_ranks
            to_ranks_seq = self.ul_to_ranks(rank, to_root_rank)
            for n in range(0,len(to_ranks_seq)):
                boundary_list.append(bd.SimpleBoundary(
                    boundary_type=constants.SOUTH,
                    from_rank=rank,
                    to_rank=int(to_ranks_seq[n]%self.total_ranks),
                    n_clockwise_rotations=3,
                ))
        else:
            boundary_list.append(cast(bd.SimpleBoundary, self.tile.boundary(SOUTH, rank=rank)))
            if self.tile.on_tile_bottom(rank):
                boundary_list[0].to_rank -= self.tile.total_ranks
                if boundary_list[0].to_rank < 0:
                    boundary_list[0].to_rank = boundary_list[0].to_rank%self.total_ranks
        return boundary_list

    def _top_left_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
        if self.tile.on_tile_top(rank) and self.tile.on_tile_left(rank):
            corner = None
        else:
            if is_even(self.tile_index(rank)) and on_tile_left(
                self.tile.subtile_index(rank)
            ):
                second_edge = self._left_edge
            else:
                second_edge = self._top_edge
            corner = self._get_corner(
                constants.NORTHWEST, rank, self._left_edge, second_edge
            )
        return corner

    def _top_right_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
        if on_tile_top(self.tile.subtile_index(rank), self.layout) and on_tile_right(
            self.tile.subtile_index(rank), self.layout
        ):
            corner = None
        else:
            if is_even(self.tile_index(rank)) and on_tile_top(
                self.tile.subtile_index(rank), self.layout
            ):
                second_edge = self._bottom_edge
            else:
                second_edge = self._right_edge
            corner = self._get_corner(
                constants.NORTHEAST, rank, self._top_edge, second_edge
            )
        return corner

    def _bottom_left_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
        if on_tile_bottom(self.tile.subtile_index(rank)) and on_tile_left(
            self.tile.subtile_index(rank)
        ):
            corner = None
        else:
            if not is_even(self.tile_index(rank)) and on_tile_bottom(
                self.tile.subtile_index(rank)
            ):
                second_edge = self._top_edge
            else:
                second_edge = self._left_edge
            corner = self._get_corner(
                constants.SOUTHWEST, rank, self._bottom_edge, second_edge
            )
        return corner

    def _bottom_right_corner(self, rank: int) -> Optional[bd.SimpleBoundary]:
        if on_tile_bottom(self.tile.subtile_index(rank)) and on_tile_right(
            self.tile.subtile_index(rank), self.layout
        ):
            corner = None
        else:
            if not is_even(self.tile_index(rank)) and on_tile_bottom(
                self.tile.subtile_index(rank)
            ):
                second_edge = self._bottom_edge
            else:
                second_edge = self._right_edge
            corner = self._get_corner(
                constants.SOUTHEAST, rank, self._bottom_edge, second_edge
            )
        return corner

    def _get_corner(
        self,
        boundary_type: int,
        rank: int,
        edge_func_1: Callable[[int], bd.Boundary],
        edge_func_2: Callable[[int], bd.Boundary],
    ) -> bd.SimpleBoundary:
        edge_1 = edge_func_1(rank)
        edge_2 = edge_func_2(edge_1.to_rank)
        rotations = edge_1.n_clockwise_rotations + edge_2.n_clockwise_rotations
        return bd.SimpleBoundary(
            boundary_type=boundary_type,
            from_rank=rank,
            to_rank=edge_2.to_rank,
            n_clockwise_rotations=rotations,
        )

    def global_extent(self, rank_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a full cube representation for the given dimensions.

        Args:
            metadata: quantity metadata

        Returns:
            extent: shape of full cube representation
        """
        return (6,) + tile_extent_from_rank_metadata(
            rank_metadata.dims, rank_metadata.extent, self.layout
        )

    def subtile_extent(self, cube_metadata: QuantityMetadata) -> Tuple[int, ...]:
        """Return the shape of a single rank representation for the given dimensions."""
        if cube_metadata.dims[0] != constants.TILE_DIM:
            raise NotImplementedError(
                "currently only supports tile dimension {constants.TILE_DIM} as the "
                "first dimension, got dims {cube_metadata.dims}"
            )
        return rank_extent_from_tile_metadata(
            cube_metadata.dims[1:], cube_metadata.extent[1:], self.layout
        )

    def subtile_slice(
        self,
        rank: int,
        global_dims: Sequence[str],
        global_extent: Sequence[int],
        overlap: bool = False,
    ) -> Tuple[Union[int, slice], ...]:
        """Return the subtile slice of a given rank on an array.

        Global refers to the domain being partitioned. For example, for a partitioning
        of a tile, the tile would be the "global" domain.

        Args:
            rank: the rank of the process
            global_dims: dimensions of the global quantity being partitioned
            global_extent: extent of the global quantity being partitioned
            overlap (optional): if True, for interface variables include the part
                of the array shared by adjacent ranks in both ranks. If False, ensure
                only one of those ranks (the greater rank) is assigned the overlapping
                section. Default is False.

        Returns:
            subtile_slice: the tuple slice of the global compute domain corresponding
                to the subtile compute domain
        """
        if global_dims[0] != constants.TILE_DIM:
            raise NotImplementedError(
                "currently only supports tile dimension {constants.TILE_DIM} as the "
                "first dimension, got dims {cube_metadata.dims}"
            )
        i_tile = self.tile_index(rank)
        return (i_tile,) + subtile_slice(
            global_dims[1:],
            global_extent[1:],
            self.layout,
            self.tile.subtile_index(rank),
            overlap=overlap,
        )
    """From here on our own functions of the CubeSpherePartitioner are listed. They are returning the amount of boundaries for each edge and the shared ranks"""
    
    def lr_boundary_seq(self, rank: int):
        """Calculates the sequence of boundaries on an edge of a tile + the amount of to_ranks shared with this rank"""
        rank_position_y = (rank % self.tile.total_ranks) // self.tile.layout[0] # position of rank on left/right boundary in y-axis
        boundary_seq = []
        if self.tile.layout[0] % self.tile.layout[1] == 0 or self.tile.layout[1] % self.tile.layout[0] == 0: # layout is dividable
            if self.tile.layout[0] >= self.tile.layout[1]:
                div = self.tile.layout[0] // self.tile.layout[1]
            else:
                div = 1
            boundary_seq = [div]*self.tile.layout[1]   # adds sequence of repeated multiples (div) in a list.
            Nboundary = boundary_seq[rank_position_y]
        else:
            nfirst = self.tile.layout[0] // self.tile.layout[1] + 1
            boundary_seq.append(nfirst)
            for y in range(1,self.tile.layout[1]): # calculates the rest of the sequence of the boundary by dividing the layout
                num1 = (self.tile.layout[0]*self.tile.layout[1]) - y*self.tile.layout[0]
                num2before = num1//self.tile.layout[1] + 1
                num2after = (num1 - self.tile.layout[0])//self.tile.layout[1] + 1
                numadd = num2before - num2after + 1
                boundary_seq.append(numadd)
            Nboundary = boundary_seq[rank_position_y] # Amount of boundaries created on the edge of a rank
        return Nboundary, boundary_seq

    def ul_boundary_seq(self, rank: int): 
        """Calculates the sequence of boundaries on an edge of a tile + the amount of to_ranks shared with this rank"""
        rank_position_x = (rank % self.tile.total_ranks)//self.tile.layout[0] # position of rank on upper/ower boundary in x-axis
        boundary_seq = []
        if self.tile.layout[0] % self.tile.layout[1] == 0 or self.tile.layout[1] % self.tile.layout[0] == 0: # layout is dividable
            if self.tile.layout[1] >= self.tile.layout[0]:
                div = self.tile.layout[1]//self.tile.layout[0]
            else:
                div = 1
            boundary_seq = [div]*self.tile.layout[0]
            Nboundary = boundary_seq[rank_position_x]
        else:
            nfirst = self.tile.layout[1] // self.tile.layout[0] + 1
            boundary_seq.append(nfirst)
            for x in range(1,self.tile.layout[0]): # calculates the rest of the sequence of the boundary by dividing the layout
                num1 = (self.tile.layout[0]*self.tile.layout[1]) - x*self.tile.layout[1]  
                num2before = num1//self.tile.layout[0] + 1      
                num2after = (num1-self.tile.layout[1])//self.tile.layout[0] + 1   
                numadd = num2before - num2after + 1  
                boundary_seq.append(numadd)
            Nboundary = boundary_seq[rank_position_x] # Amount of boundaries created on the edge of a rank
        return Nboundary, boundary_seq

    def lr_to_ranks(self, rank: int, to_root_rank: int):
        """Returns the to_ranks of the boundaries for a specific rank and its edge"""
        Nboundary, boundary_seq =  self.lr_boundary_seq(rank)
        if self.tile.layout[0] % self.tile.layout[1] == 0 or self.tile.layout[1] % self.tile.layout[0] == 0:
            div = self.tile.layout[1]//self.tile.layout[0]
            boundary_seq = boundary_seq
            if(div >= 1):
                cumsum_boundary_seq = np.cumsum(boundary_seq)//div # cumulated sum of boundary sequence to calculate shared ranks later on
            else:
                cumsum_boundary_seq = np.cumsum(boundary_seq)
        else:
            boundary_seq = [x - 1 for x in boundary_seq]
            cumsum_boundary_seq = np.cumsum(boundary_seq) # a cumulative sum of boundary_seq to know the start and end of the shared ranks 
        rank_position_y = (rank % self.tile.total_ranks) // self.tile.layout[0]
        to_root_rank = to_root_rank
        to_ranks_pot = []
        for x in range(self.tile.layout[0]):
            if self.tile.on_tile_left(rank):
                to_ranks_pot.append(to_root_rank + (self.tile.layout[0]*(self.tile.layout[1]-1)) + x)
            else:
                to_ranks_pot.append(to_root_rank + x) # creates list of all sharable ranks (potential to_ranks) on the respective edge
        to_ranks_pot = np.fliplr([to_ranks_pot, to_ranks_pot, to_ranks_pot])[1] # flips list for orientation purposes
        if(rank_position_y == 0):
            if self.tile.layout[0] % self.tile.layout[1] == 0 or self.tile.layout[1] % self.tile.layout[0] == 0:
                if self.tile.layout[0]//self.tile.layout[1] >= 1:
                    start = 0
                    end = cumsum_boundary_seq[0]
                    to_ranks_seq = to_ranks_pot[start:end]
                else:
                    start = 0
                    div = self.tile.layout[1]//self.tile.layout[0]
                    end = cumsum_boundary_seq[0]//div
                    to_ranks_seq = to_ranks_pot[start:end] 
            else:
                start = 0
                end = cumsum_boundary_seq[0]+1
                to_ranks_seq = to_ranks_pot[start:end]
        else:
            if self.tile.layout[0]%self.tile.layout[1] == 0 or self.tile.layout[1]%self.tile.layout[0] == 0:
                if self.tile.layout[0]/self.tile.layout[1] >= 1:
                    start = cumsum_boundary_seq[rank_position_y-1]
                    end = cumsum_boundary_seq[rank_position_y]
                    to_ranks_seq = to_ranks_pot[start:end]
                else:
                    div = self.tile.layout[1]//self.tile.layout[0]
                    start = rank_position_y//div
                    end = start + 1
                    to_ranks_seq = to_ranks_pot[start:end]
            else:
                start = cumsum_boundary_seq[rank_position_y-1]
                end = cumsum_boundary_seq[rank_position_y] + 1
                to_ranks_seq = to_ranks_pot[start:end]
        return to_ranks_seq

    def ul_to_ranks(self, rank: int, to_root_rank: int):
        """Returns the to_ranks of the boundaries for a specific rank and its edge"""
        Nboundary, boundary_seq =  self.ul_boundary_seq(rank)
        if self.tile.layout[0]%self.tile.layout[1] == 0 or self.tile.layout[1]%self.tile.layout[0] == 0:
            div = self.tile.layout[1]//self.tile.layout[0]
            boundary_seq = boundary_seq
            if(div >= 1):
                cumsum_boundary_seq = np.cumsum(boundary_seq)//div # cumulated sum of boundary sequence to calculate shared ranks later on
            else:
                cumsum_boundary_seq = np.cumsum(boundary_seq)
        else:
            boundary_seq = [x - 1 for x in boundary_seq]
            cumsum_boundary_seq = np.cumsum(boundary_seq) # a cumulative sum of boundary_seq to know the start and end of the shared ranks 
        rank_position_x = (rank % self.tile.total_ranks)%self.tile.layout[0]
        to_root_rank = to_root_rank
        to_ranks_pot = []
        for y in range(self.tile.layout[1]):
            if self.tile.on_tile_top(rank):
                to_ranks_pot.append(to_root_rank + self.tile.layout[0]*y) # creates list of all sharable ranks (potential to_ranks) on the respective edge
            else:
                to_ranks_pot.append(to_root_rank + self.tile.layout[0]*y + self.tile.layout[0]-1)
                
        to_ranks_pot = np.fliplr([to_ranks_pot,to_ranks_pot,to_ranks_pot])[1] # flips list for orientation         
        if(rank_position_x == 0):
            if self.tile.layout[0]%self.tile.layout[1] == 0 or self.tile.layout[1]%self.tile.layout[0] == 0:
                if self.tile.layout[1]//self.tile.layout[0] > 1:
                    div = self.tile.layout[1]//self.tile.layout[0]
                    start = 0
                    end = start + div
                    to_ranks_seq = to_ranks_pot[start:end]
                else:
                    div = self.tile.layout[0]//self.tile.layout[1]
                    start = 0
                    end = start + 1
                    to_ranks_seq = to_ranks_pot[start:end]
            else:
                start = 0
                end = cumsum_boundary_seq[0]+1
                to_ranks_seq = to_ranks_pot[start:end]
        else:
            if self.tile.layout[0]%self.tile.layout[1] == 0 or self.tile.layout[1]%self.tile.layout[0] == 0:
                if self.tile.layout[1]//self.tile.layout[0] > 1:
                    div = self.tile.layout[1]//self.tile.layout[0]
                    start = rank_position_x*div
                    end = start + div
                    to_ranks_seq = to_ranks_pot[start:end]
                else:
                    div = self.tile.layout[0]//self.tile.layout[1]
                    start = rank_position_x//div
                    end = start + 1
                    to_ranks_seq = to_ranks_pot[start:end]
            else:
                start = cumsum_boundary_seq[rank_position_x-1]
                end = cumsum_boundary_seq[rank_position_x] + 1
                to_ranks_seq = to_ranks_pot[start:end]
        return to_ranks_seq


def on_tile_left(subtile_index: Tuple[int, int, int, int]) -> bool:
    return subtile_index[1] == 0


def on_tile_right(subtile_index: Tuple[int, int, int, int], layout: Tuple[int, int]) -> bool:
    return subtile_index[1] == layout[0]-1


def on_tile_top(subtile_index: Tuple[int, int, int, int], layout: Tuple[int, int]) -> bool:
    return subtile_index[2] == layout[1] - 1


def on_tile_bottom(subtile_index: Tuple[int, int, int, int]) -> bool:
    return subtile_index[2] == 0


def rotate_subtile_rank(
    rank: int, layout: Tuple[int, int], n_clockwise_rotations: int
) -> int:
    """Returns the rank position where this rank would be if you rotated the
    tile n_clockwise_rotations times.
    """
    if n_clockwise_rotations == 0:
        to_tile_rank = rank
    elif n_clockwise_rotations == 1:
        total_ranks = layout[0] * layout[1]
        rank_array = np.arange(total_ranks).reshape(layout)
        rotated_rank_array = np.rot90(rank_array)
        to_tile_rank = rank_array[np.where(rotated_rank_array == rank)][0]
    else:
        raise NotImplementedError()
    return to_tile_rank


def transpose_subtile_rank(rank, layout):
    """Returns the rank position where this rank would be if you transposed
    the tile.
    """
    return transform_subtile_rank(np.transpose, rank, layout)


def fliplr_subtile_rank(rank, layout):
    """Returns the rank position where this rank would be if you flipped the
    tile along a vertical axis
    """
    return transform_subtile_rank(np.fliplr, rank, layout)


def flipud_subtile_rank(rank, layout):
    """Returns the rank position where this rank would be if you flipped the
    tile along a horizontal axis
    """
    return transform_subtile_rank(np.flipud, rank, layout)


def transform_subtile_rank(
    transform_func: Callable[[np.ndarray], np.ndarray],
    rank: int,
    layout: Tuple[int, int],
):
    """Returns the rank position where this rank would be if you performed
    a transformation on the tile which strictly moves ranks.
    """
    total_ranks = layout[0] * layout[1]
    rank_array = np.arange(total_ranks).reshape(layout)
    transformed_rank_array = transform_func(rank_array)
    return rank_array[np.where(transformed_rank_array == rank)][0]


def subtile_index(
    rank: int, ranks_per_tile: int, layout: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    within_tile_rank = rank % ranks_per_tile
    j = within_tile_rank // layout[1] 
    i = within_tile_rank % layout[0]
    k = within_tile_rank // layout[0] 
    l = within_tile_rank % layout[1]
    return j, i, k, l


def is_even(value: Union[int, float]) -> bool:
    return value % 2 == 0


def tile_extent_from_rank_metadata(
    dims: Sequence[str], rank_extent: Sequence[int], layout: Tuple[int, int]
) -> Tuple[int, ...]:
    """
    Returns the extent of a tile given data about a single rank, and the tile
    layout.

    Args:
        dims: dimension names
        rank_extent: the extent of one rank
        layout: the (y, x) number of ranks along each tile axis

    Returns:
        tile_extent: the extent of one tile
    """
    layout_factors = np.asarray(
        utils.list_by_dims(dims, layout, non_horizontal_value=1)
    )
    return extent_from_metadata(dims, rank_extent, layout_factors)


def rank_extent_from_tile_metadata(
    dims: Sequence[str], tile_extent: Sequence[int], layout: Tuple[int, int]
) -> Tuple[int, ...]:
    """
    Returns the extent of a rank given data about a tile, and the tile
    layout.

    Args:
        dims: dimension names
        rank_extent: the extent of a tile
        layout: the (y, x) number of ranks along each tile axis

    Returns:
        rank_extent: the extent of one rank
    """
    layout_factors = 1 / np.asarray(
        utils.list_by_dims(dims, layout, non_horizontal_value=1)
    )
    return extent_from_metadata(dims, tile_extent, layout_factors)


def extent_from_metadata(
    dims: Iterable[str], extent: Iterable[int], layout_factors: np.ndarray
) -> Tuple[int, ...]:
    return_extents = []
    for dim, rank_extent, layout_factor in zip(dims, extent, layout_factors):
        if dim in constants.INTERFACE_DIMS:
            add_extent = -1
        else:
            add_extent = 0
        tile_extent = (rank_extent + add_extent) * layout_factor - add_extent
        return_extents.append(int(tile_extent))  # layout_factor is float, need to cast
    return tuple(return_extents)


@dataclasses.dataclass
class _IndexData1D:
    dim: str
    extent: int
    i_subtile: int
    n_ranks: int

    @property
    def base_extent(self):
        return self.extent - self.extent_minus_gridcell_count

    @property
    def extent_minus_gridcell_count(self):
        if self.dim in constants.INTERFACE_DIMS:
            return 1
        else:
            return 0

    @property
    def is_end_index(self):
        return self.i_subtile == self.n_ranks - 1


def _index_generator(dims, tile_extent, subtile_index, horizontal_layout):
    subtile_extent = rank_extent_from_tile_metadata(
        dims, tile_extent, horizontal_layout
    )
    quantity_layout = utils.list_by_dims(
        dims, horizontal_layout, non_horizontal_value=1
    )
    quantity_subtile_index = utils.list_by_dims(
        dims, subtile_index, non_horizontal_value=0
    )
    for dim, extent, i_subtile, n_ranks in zip(
        dims, subtile_extent, quantity_subtile_index, quantity_layout
    ):
        yield _IndexData1D(dim, extent, i_subtile, n_ranks)


def subtile_slice(
    dims: Iterable[str],
    global_extent: Iterable[int],
    layout: Tuple[int, int],
    subtile_index: Tuple[int, int, int, int],
    overlap: bool = False,
) -> Tuple[slice, ...]:
    """
    Returns the slice of data within a tile's computational domain belonging
    to a single rank.

    Args:
        dims: dimension names for each axis
        global_extent: size of the tile or cube's computational domain
        layout: the (y, x) number of ranks along each tile axis
        subtile_index: the (y, x) position of the rank on the tile
        overlap: whether to assign regions which belong to multiple ranks
            to both ranks, or only to the higher rank (default)
    """
    return_list = []
    # discard last index for interface variables, unless you're the last rank
    # done so that only one rank is responsible for the shared interface point
    for index in _index_generator(dims, global_extent, subtile_index, layout):
        start = index.i_subtile * index.base_extent
        if index.is_end_index or overlap:
            end = start + index.extent
        else:
            end = start + index.base_extent
        return_list.append(slice(start, end))
    return tuple(return_list)