import math
import numpy as np

class CubedSpherePartitioner(object):
    """Domain decomposition and distribution of MPI ranks
       on a 6-faced cubed sphere.
    
      Fixed cube tile (facet) numbering:
          |-----|
          |     |
          |  3  ^
          |     x
    |y>---|---<y|-----|y>---|
    x     |     |     x     |
    v  5  ^  1  ^  2  v  4  |
    |     y     y     |     |
    |-----|x>---|x>---|-----| 
          |     |
          ^  6  |
          y     |
          |x>---|
    """

    def __init__(self, comm, domain, num_halo):
        assert len(domain) == 3, \
            "Must specify a three-dimensional domain"
        assert domain[0] > 0 and domain[1] > 0 and domain[2] > 0, \
            "Invalid domain specification (negative size)"
        assert num_halo >= 0, "Number of halo points must be zero or positive"
        assert domain[1] == domain[2], "Must specify cubed sphere faces of quadratic dimensions"

        self.__comm = comm
        self.__num_halo = num_halo

        self.__global_rank = comm.Get_rank()
        self.__num_ranks = comm.Get_size()

        assert self.__num_ranks % 6 == 0, "Number of ranks must be divisible by 6 for cube topology"
        assert self.__num_ranks == 6 or math.sqrt(self.__num_ranks / 6).is_integer(), \
            "Number of ranks per face must be square of an integer"
        
        self.__ranks_per_tile = self.__num_ranks // 6
        self.__ranks_per_axis = int(math.sqrt(self.__ranks_per_tile))

        self.__local_rank = self.__rank_global2local()

        self.__rank2tile, \
        self.__tile2ranks, \
        self.__tile2root = self.__assign_ranks_tiles(self.__num_ranks, self.__ranks_per_tile)

        self.__tile = self.__rank2tile[self.__global_rank] # 1-based tile numbering

        self.__tile_root = self.tile_root()

        self.__tile_neighbors = self.__assign_tile_neighbors()

        self.__tile_comm = self.__split_tile_comm(self.__comm, self.__tile, self.__global_rank)

        self.__rank_grid = self.__assign_rank_grid(self.__tile, self.__tile2ranks, self.__tile_neighbors, self.__ranks_per_axis)

        self.__rank_neighbors = self.__assign_rank_neighbors(self.__global_rank, self.__rank_grid, self.__tile)

        self.__neighbor_halo_rotations = self.__assign_neighbor_halo_rotations(self.__tile,
                                                                               self.__rank2tile,
                                                                               self.__tile_neighbors,
                                                                               self.__rank_neighbors)

        self.__global_shape = [domain[0], domain[1] + 2 * num_halo, domain[2] + 2 * num_halo]

        self.__size = self.__setup_grid()

        assert domain[1] >= self.__size[0] and domain[2] >= self.__size[1], "Domain is too small for number of ranks"
        
        self.__setup_domain(domain, num_halo)


    def comm(self):
        """Returns the MPI communicator used to setup this partitioner"""
        return self.__comm    


    def tile_comm(self):
        """Returns the MPI communicator of the tile of the current MPI worker"""
        return self.__tile_comm


    def num_halo(self):
        """Returns the number of halo points"""
        return self.__num_halo


    def global_rank(self):
        """Returns the global rank of the current MPI worker"""
        return self.__global_rank


    def local_rank(self):
        """Returns the local tile rank of the current MPI worker"""
        return self.__local_rank


    def num_ranks(self):
        """Returns the global number of ranks that have been distributed by this partitioner"""
        return self.__num_ranks


    def tile(self):
        """Returns tile number of the current MPI worker"""
        return self.__tile


    def tile_root(self):
        """Returns the global MPI rank which is root rank of the tile of the current MPI worker"""
        return self.__tile2root[self.__tile]


    def shape(self):
        """Returns the shape of a local field (including halo points)"""
        return self.__shape


    def global_shape(self):
        """Returns the shape of a global tile field (including halo points)"""
        return self.__global_shape


    def size(self):
        """Dimensions of the two-dimensional worker grid"""
        return self.__size    


    def position(self):
        """Position of current rank on two-dimensional worker grid"""
        return self.__rank_to_position(self.__local_rank)


    def left(self):
        """Returns the rank of the left neighbor"""
        return self.__rank_neighbors['L']


    def right(self):
        """Returns the rank of the right neighbor"""
        return self.__rank_neighbors['R']


    def top(self):
        """Returns the rank of the top/up neighbor"""
        return self.__rank_neighbors['U']


    def bottom(self):
        """Returns the rank of the bottom/down neighbor"""
        return self.__rank_neighbors['D']


    def rot_halo_left(self):
        """Returns number of positive 90 degree rotations required to match halo to left neighbor's halo"""
        return self.__neighbor_halo_rotations['L']


    def rot_halo_right(self):
        """Returns number of positive 90 degree rotations required to match halo to right neighbor's halo"""
        return self.__neighbor_halo_rotations['R']


    def rot_halo_top(self):
        """Returns number of positive 90 degree rotations required to match halo to top neighbor's halo"""
        return self.__neighbor_halo_rotations['U']


    def rot_halo_bottom(self):
        """Returns number of positive 90 degree rotations required to match halo to bottom neighbor's halo"""
        return self.__neighbor_halo_rotations['D']

    
    def scatter(self, field, tile_root=None):
        """Scatter a global field from a tile root rank to the tile workers"""
        if tile_root is None:
            tile_root = self.__tile_root
        if self.__global_rank == tile_root:
            assert np.any(field.shape[0] == np.array(self.__global_shape[0])), \
                "Field does not have correct shape"
        assert tile_root in self.__tile2ranks[self.__tile], "Root processor must be a valid tile rank"
        if self.__ranks_per_tile == 1:
            return field
        sendbuf = None
        if self.__global_rank == tile_root:
            sendbuf = np.empty( [self.__ranks_per_tile,] + self.__max_shape, dtype=field.dtype )
            for rank in range(self.__ranks_per_tile):
                j_start, i_start, j_end, i_end = self.__domains[rank]
                sendbuf[rank, :, :j_end-j_start, :i_end-i_start] = field[:, j_start:j_end, i_start:i_end]
        recvbuf = np.empty(self.__max_shape, dtype=field.dtype)
        self.__tile_comm.Scatter(sendbuf, recvbuf, root=0)
        j_start, i_start, j_end, i_end = self.__domain
        return recvbuf[:, :j_end-j_start, :i_end-i_start].copy()
        
    
    def gather(self, field, tile_root=None):
        """Gather a distributed fields from tile workers to a single global field on a tile root rank"""
        if tile_root is None:
            tile_root = self.__tile_root
        assert np.any(field.shape == np.array(self.__shape)), "Field does not have correct shape"
        assert tile_root in self.__tile2ranks[self.__tile] + [-1], "Root processor must be -1 (all) or a valid rank"
        if self.__ranks_per_tile == 1:
            return field
        j_start, i_start, j_end, i_end = self.__domain
        sendbuf = np.empty( self.__max_shape, dtype=field.dtype )
        sendbuf[:, :j_end-j_start, :i_end-i_start] = field
        recvbuf = None
        if self.__global_rank == tile_root or tile_root == -1:
            recvbuf = np.empty( [self.__ranks_per_tile,] + self.__max_shape, dtype=field.dtype )
        if tile_root > -1:
            self.__tile_comm.Gather(sendbuf, recvbuf, root=0)
        else:
            self.__tile_comm.Allgather(sendbuf, recvbuf)
        global_field = None
        if self.__global_rank == tile_root or tile_root == -1:
            global_field = np.empty(self.__global_shape, dtype=field.dtype)
            for rank in range(self.__ranks_per_tile):
                j_start, i_start, j_end, i_end = self.__domains[rank]
                global_field[:, j_start:j_end, i_start:i_end] = recvbuf[rank, :, :j_end-j_start, :i_end-i_start]
        return global_field
                
    
    def compute_domain(self):
        """Return position of subdomain without halo on the global domain"""
        return [self.__domain[0] + self.__num_halo, self.__domain[1] + self.__num_halo, \
                self.__domain[2] - self.__num_halo, self.__domain[3] - self.__num_halo]


    def __rank_global2local(self):
        """Return local tile rank based on global rank, ranks per tile"""
        return self.__global_rank % self.__ranks_per_tile


    def __rank_local2global(self, tile):
        """Return global rank based on local tile rank, tile number (1-based), ranks per tile"""
        return self.__local_rank + self.__ranks_per_tile * (tile - 1)


    def __assign_ranks_tiles(self, num_ranks, ranks_per_tile):
        """Return dictionaries: rank->tile, tile->[ranks], tile->root rank"""
        rank2tile = {i: (i // ranks_per_tile) + 1 for i in range(num_ranks)} # 1-based tile numbering
        tile2ranks = dict()
        for k, v in rank2tile.items():
            tile2ranks.setdefault(v, list()).append(k)
        tile2root = {v: k[0] for v, k in tile2ranks.items()}
        return rank2tile, tile2ranks, tile2root
    

    def __assign_tile_neighbors(self):
        """Return dictionary: tile, direction->neighbor tile, number of relative positive 90 degree coordinate rotations"""
        tile_neighbors = {1: {'U': [3, 1], 'D': [6, 0], 'L': [5, 3], 'R': [2, 0]},
                          2: {'U': [3, 0], 'D': [6, 1], 'L': [1, 0], 'R': [4, 3]},
                          3: {'U': [5, 1], 'D': [2, 0], 'L': [1, 3], 'R': [4, 0]},
                          4: {'U': [5, 0], 'D': [2, 1], 'L': [3, 0], 'R': [6, 3]},
                          5: {'U': [1, 1], 'D': [4, 0], 'L': [3, 3], 'R': [6, 0]},
                          6: {'U': [1, 0], 'D': [4, 1], 'L': [5, 0], 'R': [2, 3]}}
        # Sanity checks - comment out for production on Daint as it only works for Python 3.8 (not on Daint as of Aug20) ---
        # assert tile_neighbors[1]['L'][0] == tile_neighbors[6]['L'][0] == tile_neighbors[4]['U'][0], 'Cube geometry faulty'
        # assert tile_neighbors[6]['D'][0] == tile_neighbors[2]['R'][0] == tile_neighbors[3]['R'][0], 'Cube geometry faulty' 
        # assert tile_neighbors[3]['U'][0] == tile_neighbors[6]['L'][0] == tile_neighbors[1]['L'][0], 'Cube geometry faulty'
        # for i in range(6):
        #     rotation_sum = 0
        #     [rotation_sum := rotation_sum + k[1] for v, k in tile_neighbors[i+1].items()]
        #     assert rotation_sum == 4, 'Cube geometry faulty' 
        # ------------------------------------------------------------------------------------------------------------------
        return tile_neighbors


    def __calculate_rank_grid(self, tile, tile2ranks, ranks_per_axis, rotation=0):
            """Return rotated array containing square grid of tile's global ranks"""
            return np.rot90(np.flipud(np.asarray(tile2ranks[tile]).reshape(ranks_per_axis, -1)), rotation)


    def __assign_rank_grid(self, tile, tile2ranks, tile_neighbors, ranks_per_axis):
        """Return dictionary of arrays containing all neighboring tiles' placements of global ranks
           and whether their bordering coordinate orientation is flipped with regard to center tile"""
        rank_grid = {tile: self.__calculate_rank_grid(tile, tile2ranks, ranks_per_axis)}
        for k, v in tile_neighbors[tile].items():
            rank_grid[k] = self.__calculate_rank_grid(v[0], tile2ranks, ranks_per_axis, v[1])
        return rank_grid


    def __assign_rank_neighbors(self, global_rank, rank_grid, tile):
        """Return dictionary: global rank->global neighbor ranks"""
        up_down_grid = np.vstack((rank_grid['U'], rank_grid[tile], rank_grid['D']))
        left_right_grid = np.hstack((rank_grid['L'], rank_grid[tile], rank_grid['R']))
        y_up_down_grid, x_up_down_grid = np.where(up_down_grid == global_rank)
        y_left_right_grid, x_left_right_grid = np.where(left_right_grid == global_rank)
        return {'U': up_down_grid[(y_up_down_grid-1, x_up_down_grid)].item(),
                'D': up_down_grid[(y_up_down_grid+1, x_up_down_grid)].item(),
                'L': left_right_grid[(y_left_right_grid, x_left_right_grid-1)].item(),
                'R': left_right_grid[(y_left_right_grid, x_left_right_grid+1)].item()}


    def __assign_neighbor_halo_rotations(self, tile, rank2tile, tile_neighbors, rank_neighbors):
        """Return dictionary: Required positive 90 degree halo rotations to match neighbor's halo orientation"""
        neighbor_tiles = tile_neighbors[tile]
        neighbor_tiles_rotations = {v[0]: v[1] for v in neighbor_tiles.values()}
        return {v: 0 if tile == rank2tile[k] else neighbor_tiles_rotations[rank2tile[k]] 
                for v, k in rank_neighbors.items()}


    def __split_tile_comm(self, comm, tile, global_rank):
        """Return local tile communicator split from comm using tile as color"""
        tile_comm = comm.Split(tile, global_rank)
        assert self.__local_rank == tile_comm.Get_rank(), "Check calculated local rank vs. MPI local rank"
        return tile_comm


    def __setup_grid(self):
        """Distribute ranks onto a Cartesian grid of workers"""
        return (self.__ranks_per_axis, self.__ranks_per_axis)
    

    def __setup_domain(self, shape, num_halo):
        """Distribute the points of the computational grid onto the Cartesian grid of workers"""
        assert len(shape) == 3, "Must pass a 3-dimensional shape"
        size_z = shape[0]
        size_y = self.__distribute_to_bins(shape[1], self.__size[0])
        size_x = self.__distribute_to_bins(shape[2], self.__size[1])

        pos_y = self.__cumsum(size_y, initial_value=num_halo)
        pos_x = self.__cumsum(size_x, initial_value=num_halo)

        domains = []
        shapes = []
        for rank in range(self.__ranks_per_tile):
            pos = self.__rank_to_position(rank)
            domains += [[ pos_y[pos[0]] - num_halo, pos_x[pos[1]] - num_halo, \
                            pos_y[pos[0] + 1] + num_halo, pos_x[pos[1] + 1] + num_halo ]]
            shapes += [[ size_z, domains[rank][2] - domains[rank][0], \
                                    domains[rank][3] - domains[rank][1] ]]
        self.__domains, self.__shapes =  domains, shapes
        
        self.__domain, self.__shape = domains[self.__local_rank], shapes[self.__local_rank]
        self.__max_shape = self.__find_max_shape( self.__shapes )


    def __distribute_to_bins(self, number, bins):
        """Distribute a number of elements to a number of bins"""
        assert (number % bins == 0), 'Domain size must be divisible by ranks per coordinate axis'
        n = number // bins
        bin_size = [n] * bins
        return bin_size


    def __cumsum(self, array, initial_value=0):
        """Cumulative sum with an optional initial value (default is zero)"""
        cumsum = [initial_value]
        for i in array:
            cumsum += [ cumsum[-1] + i ]
        return cumsum


    def __find_max_shape(self, shapes):
        max_shape = shapes[0]
        for shape in shapes[1:]:
            max_shape = list(map(max, zip(max_shape, shape)))
        return max_shape
    

    def __rank_to_position(self, local_rank):
        """Find position of rank on worker grid"""
        return ( local_rank // self.__size[1], local_rank % self.__size[1] )


    def __position_to_rank(self, position):
        """Find local rank given a position on the worker grid"""
        if position[0] is None or position[1] is None:
            return None
        else:
            return position[0] * self.__size[1] + position[1]