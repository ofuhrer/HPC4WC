import math
import numpy as np

class Partitioner:
    """2-dimensional domain decomposition of a 3-dimensional computational
       grid among MPI ranks on a communicator."""
    
    
    def __init__(self, comm, domain, num_halo, periodic = (True, True)):
        assert len(domain) == 3, \
            "Must specify a three-dimensional domain"
        assert domain[0] > 0 and domain[1] > 0 and domain[2] > 0, \
            "Invalid domain specification (negative size)"
        assert num_halo >= 0, "Number of halo points must be zero or positive"

        self.__comm = comm
        self.__num_halo = num_halo
        self.__periodic = periodic

        self.__rank = comm.Get_rank()
        self.__num_ranks = comm.Get_size()
        self.__global_shape = [domain[0], domain[1] + 2 * num_halo, domain[2] + 2 * num_halo]

        size = self.__setup_grid()
        assert domain[1] >= size[0] and domain[2] >= size[1], "Domain is too small for number of ranks"
        self.__setup_domain(domain, num_halo)
        
    def comm(self):
        """Returns the MPI communicator use to setup this partitioner"""
        return self.__comm
    
    
    def num_halo(self):
        """Returns the number of halo points"""
        return self.__num_halo
    
    
    def periodic(self, dim=None):
        """Returns the periodicity of individual or all dimensions"""
        if dim is not None:
            return self.__periodic[dim]
        else:
            return self.__periodic

        
    def rank(self):
        """Returns the rank of the current MPI worker"""
        return self.__rank
    
    
    def num_ranks(self):
        """Returns the number of ranks that have been distributed by this partitioner"""
        return self.__num_ranks
    
        
    def shape(self):
        """Returns the shape of a local field (including halo points)"""
        return self.__shape
    
    
    def global_shape(self):
        """Returns the shape of a local field (including halo points)"""
        return self.__global_shape
    

    def grid_size(self):
        """Dimensions of the two-dimensional worker grid"""
        return self.__grid_size
    
    
    def grid_position(self):
        """Position of current rank on two-dimensional worker grid"""
        return self.__rank_to_position(self.__rank)


    def left(self):
        """Returns the rank of the left neighbor"""
        return self.__get_neighbor_rank( [0, -1] )
    
    
    def right(self):
        """Returns the rank of the left neighbor"""
        return self.__get_neighbor_rank( [0, +1] )
    
    
    def top(self):
        """Returns the rank of the left neighbor"""
        return self.__get_neighbor_rank( [+1, 0] )
    
    
    def bottom(self):
        """Returns the rank of the left neighbor"""
        return self.__get_neighbor_rank( [-1, 0] )
    
    
    def scatter(self, field, root=0):
        """Scatter a global field from a root rank to the workers"""
        if self.__rank == root:
            assert np.any(field.shape[0] == np.array(self.__global_shape[0])), \
                "Field does not have correct shape"
        assert 0 <= root < self.__num_ranks, "Root processor must be a valid rank"
        if self.__num_ranks == 1:
            return field
        sendbuf = None
        if self.__rank == root:
            sendbuf = np.empty( [self.__num_ranks,] + self.__max_shape, dtype=field.dtype )
            for rank in range(self.__num_ranks):
                j_start, i_start, j_end, i_end = self.__domains[rank]
                sendbuf[rank, :, :j_end-j_start, :i_end-i_start] = field[:, j_start:j_end, i_start:i_end]
        recvbuf = np.empty(self.__max_shape, dtype=field.dtype)
        self.__comm.Scatter(sendbuf, recvbuf, root=root)
        j_start, i_start, j_end, i_end = self.__domain
        return recvbuf[:, :j_end-j_start, :i_end-i_start].copy()
        
    
    def gather(self, field, root=0):
        """Gather a distributed fields from workers to a single global field on a root rank"""
        assert np.any(field.shape == np.array(self.__shape)), "Field does not have correct shape"
        assert -1 <= root < self.__num_ranks, "Root processor must be -1 (all) or a valid rank"
        if self.__num_ranks == 1:
            return field
        j_start, i_start, j_end, i_end = self.__domain
        sendbuf = np.empty( self.__max_shape, dtype=field.dtype )
        sendbuf[:, :j_end-j_start, :i_end-i_start] = field
        recvbuf = None
        if self.__rank == root or root == -1:
            recvbuf = np.empty( [self.__num_ranks,] + self.__max_shape, dtype=field.dtype )
        if root > -1:
            self.__comm.Gather(sendbuf, recvbuf, root=root)
        else:
            self.__comm.Allgather(sendbuf, recvbuf)
        global_field = None
        if self.__rank == root or root == -1:
            global_field = np.empty(self.__global_shape, dtype=field.dtype)
            for rank in range(self.__num_ranks):
                j_start, i_start, j_end, i_end = self.__domains[rank]
                global_field[:, j_start:j_end, i_start:i_end] = recvbuf[rank, :, :j_end-j_start, :i_end-i_start]
        return global_field
                
    
    def compute_domain(self):
        """Return position of subdomain without halo on the global domain"""
        return [self.__domain[0] + self.__num_halo, self.__domain[1] + self.__num_halo, \
                self.__domain[2] - self.__num_halo, self.__domain[3] - self.__num_halo]


    def __setup_grid(self):
        """Distribute ranks onto a Cartesian grid of workers"""
        for ranks_x in range(math.floor( math.sqrt(self.__num_ranks) ), 0, -1):
            if self.__num_ranks % ranks_x == 0:
                break
        self.__grid_size = (self.__num_ranks // ranks_x, ranks_x)
        return self.__grid_size
        

    def __get_neighbor_rank(self, offset):
        """Get the rank ID of a neighboring rank at a certain offset relative to the current rank"""
        position = self.__rank_to_position(self.__rank)
        pos_y = self.__cyclic_offset(position[0], offset[0], self.__grid_size[0], self.__periodic[0])
        pos_x = self.__cyclic_offset(position[1], offset[1], self.__grid_size[1], self.__periodic[1])
        return self.__position_to_rank( [pos_y, pos_x] )


    def __cyclic_offset(self, position, offset, size, periodic=True):
        """Add offset with cyclic boundary conditions"""
        pos = position + offset
        if periodic:
            while pos < 0:
                pos += size
            while pos > size - 1:
                pos -= size
        return pos if -1 < pos < size else None


    def __setup_domain(self, shape, num_halo):
        """Distribute the points of the computational grid onto the Cartesian grid of workers"""
        assert len(shape) == 3, "Must pass a 3-dimensional shape"
        size_z = shape[0]
        size_y = self.__distribute_to_bins(shape[1], self.__grid_size[0])
        size_x = self.__distribute_to_bins(shape[2], self.__grid_size[1])

        pos_y = self.__cumsum(size_y, initial_value=num_halo)
        pos_x = self.__cumsum(size_x, initial_value=num_halo)

        domains = []
        shapes = []
        for rank in range(self.__num_ranks):
            pos = self.__rank_to_position(rank)
            domains += [[ pos_y[pos[0]] - num_halo, pos_x[pos[1]] - num_halo, \
                          pos_y[pos[0] + 1] + num_halo, pos_x[pos[1] + 1] + num_halo ]]
            shapes += [[ size_z, domains[rank][2] - domains[rank][0], \
                                 domains[rank][3] - domains[rank][1] ]]
        self.__domains, self.__shapes =  domains, shapes
        
        self.__domain, self.__shape = domains[self.__rank], shapes[self.__rank]
        self.__max_shape = self.__find_max_shape( self.__shapes )


    def __distribute_to_bins(self, number, bins):
        """Distribute a number of elements to a number of bins"""
        n = number // bins
        bin_size = [n] * bins
        # make bins in the middle slightly larger
        extend = number - n * bins
        if extend > 0:
            start_extend = bins // 2 - extend // 2
            bin_size[start_extend:start_extend + extend] = \
                [ n + 1 for n in bin_size[start_extend:start_extend + extend] ]
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
    

    def __rank_to_position(self, rank):
        """Find position of rank on worker grid"""
        return ( rank // self.__grid_size[1], rank % self.__grid_size[1] )
    

    def __position_to_rank(self, position):
        """Find rank given a position on the worker grid"""
        if position[0] is None or position[1] is None:
            return None
        else:
            return position[0] * self.__grid_size[1] + position[1]
    

