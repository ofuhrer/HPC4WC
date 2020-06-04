import math
import numpy as np

class Partitioner:
    """2-dimensional domain decomposition of a 3-dimensional computational
       grid among MPI ranks on a communicator."""
    
    
    def __init__(self, comm, domain, num_halo):
        self.comm = comm
        self.num_ranks = comm.Get_size()
        self.rank = comm.Get_rank()
        self.num_halo = num_halo
        self.global_shape = [domain[0], domain[1] + 2 * num_halo, domain[2] + 2 * num_halo]

        self.__setup_grid()
        self.__setup_domain(domain, num_halo)
    

    def shape(self):
        """Returns the shape of a local field (including halo points)"""
        return self.shape
    
    
    def left(self):
        """Returns the rank of the left neighbor"""
        return self.left_neighbor
    
    
    def right(self):
        """Returns the rank of the left neighbor"""
        return self.right_neighbor
    
    
    def top(self):
        """Returns the rank of the left neighbor"""
        return self.top_neighbor
    
    
    def bottom(self):
        """Returns the rank of the left neighbor"""
        return self.bottom_neighbor
    
    
    def scatter(self, field, root=0):
        """Scatter a global field from a root rank to the workers"""
        if self.rank == root:
            assert np.any(field.shape[0] == np.array(self.global_shape[0])), \
                "Field does not have correct shape"
        assert 0 <= root < self.num_ranks, "Root processor must be a valid rank"
        if self.num_ranks == 1:
            return field
        sendbuf = None
        if self.rank == root:
            sendbuf = np.empty( [self.num_ranks,] + self.__max_shape, dtype=field.dtype )
            for rank in range(self.num_ranks):
                j_start, i_start, j_end, i_end = self.__domain[rank]
                sendbuf[rank, :, :j_end-j_start, :i_end-i_start] = field[:, j_start:j_end, i_start:i_end]
        recvbuf = np.empty(self.__max_shape, dtype=field.dtype)
        self.comm.Scatter(sendbuf, recvbuf, root=root)
        j_start, i_start, j_end, i_end = self.domain
        return recvbuf[:, :j_end-j_start, :i_end-i_start].copy()
        
    
    def gather(self, field, root=0):
        """Gather a distributed fields from workers to a single global field on a root rank"""
        assert np.any(field.shape == np.array(self.shape)), "Field does not have correct shape"
        assert -1 <= root < self.num_ranks, "Root processor must be -1 (all) or a valid rank"
        if self.num_ranks == 1:
            return field
        j_start, i_start, j_end, i_end = self.domain
        sendbuf = np.empty( self.__max_shape, dtype=field.dtype )
        sendbuf[:, :j_end-j_start, :i_end-i_start] = field
        recvbuf = None
        if self.rank == root or root == -1:
            recvbuf = np.empty( [self.num_ranks,] + self.__max_shape, dtype=field.dtype )
        if root > -1:
            self.comm.Gather(sendbuf, recvbuf, root=root)
        else:
            self.comm.Allgather(sendbuf, recvbuf)
        global_field = None
        if self.rank == root or root == -1:
            global_field = np.empty(self.global_shape, dtype=field.dtype)
            for rank in range(self.num_ranks):
                j_start, i_start, j_end, i_end = self.__domain[rank]
                global_field[:, j_start:j_end, i_start:i_end] = recvbuf[rank, :, :j_end-j_start, :i_end-i_start]
        return global_field
                
    
    def compute_domain(self):
        """Return position of subdomain without halo on the global domain"""
        return [self.domain[0] + self.num_halo, self.domain[1] + self.num_halo, \
                self.domain[2] - self.num_halo, self.domain[3] - self.num_halo]


    def __setup_grid(self):
        """Distribute ranks onto a Cartesian grid of workers"""
        for ranks_x in range(math.floor( math.sqrt(self.num_ranks) ), 0, -1):
            if self.num_ranks % ranks_x == 0:
                break
        self.size = (self.num_ranks // ranks_x, ranks_x)
        self.position = self.__rank_to_position(self.rank)

        self.left_neighbor = self.__get_neighbor_rank( [0, -1] )
        self.right_neighbor = self.__get_neighbor_rank( [0, +1] )
        self.top_neighbor = self.__get_neighbor_rank( [+1, 0] )
        self.bottom_neighbor = self.__get_neighbor_rank( [-1, 0] )


    def __get_neighbor_rank(self, offset):
        """Get the rank ID of a neighboring rank at a certain offset relative to the current rank"""
        pos_y = self.__cyclic_offset(self.position[0], offset[0], self.size[0])
        pos_x = self.__cyclic_offset(self.position[1], offset[1], self.size[1])
        return self.__position_to_rank( [pos_y, pos_x] )


    def __cyclic_offset(self, position, offset, size):
        """Add offset with cyclic boundary conditions"""
        pos = position + offset
        while pos < 0:
            pos += size
        while pos > size - 1:
            pos -= size
        return pos


    def __setup_domain(self, shape, num_halo):
        """Distribute the points of the computational grid onto the Cartesian grid of workers"""
        assert len(shape) == 3, "Must pass a 3-dimensional shape"
        size_z = shape[0]
        size_y = self.__distribute_to_bins(shape[1], self.size[0])
        size_x = self.__distribute_to_bins(shape[2], self.size[1])

        pos_y = self.__cumsum(size_y, initial_value=num_halo)
        pos_x = self.__cumsum(size_x, initial_value=num_halo)

        compute = []
        domain = []
        shape = []
        for rank in range(self.num_ranks):
            pos = self.__rank_to_position(rank)
            compute += [[ pos_y[pos[0]], pos_x[pos[1]],  pos_y[pos[0] + 1], pos_x[pos[1] + 1] ]]
            domain += [[ compute[rank][0] - num_halo, compute[rank][1] - num_halo, \
                         compute[rank][2] + num_halo, compute[rank][3] + num_halo ]]
            shape += [[ size_z, domain[rank][2] - domain[rank][0], \
                                domain[rank][3] - domain[rank][1] ]]

        self.__domain, self.__shape =  domain, shape
        self.__max_shape = self.__find_max_shape( self.__shape )
        self.domain, self.shape = domain[self.rank], shape[self.rank]


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
        return ( rank // self.size[1], rank % self.size[1] )
    

    def __position_to_rank(self, position):
        """Find rank given a position on the worker grid"""
        return position[0] * self.size[1] + position[1]
    

