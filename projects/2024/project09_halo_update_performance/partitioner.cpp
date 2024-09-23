// Author: Philip Ploner
// Date: 2024-08-24
// 2-dimensional domain decomposition of a 3-dimensional computational grid among MPI ranks on a communicator, implemented in C++
// This code follows the same structure as the partitioner.py file from HPC4WC/day3/


#include <cassert>
#include <vector>
#include <cmath>
#include <memory>
#include <thread>
#include <chrono>
#include <mpi.h>
#include "utils.h"

class Partitioner
{
public:
    Partitioner(MPI_Comm comm, std::vector<int> domain, int num_halo, std::pair<bool, bool> periodic = {true, true})
    {
        assert(domain.size() == 3);
        assert(domain[0] > 0 && domain[1] > 0 && domain[2] > 0);
        assert(num_halo >= 0);

        comm_ = comm;
        num_halo_ = num_halo;
        periodic_ = periodic;

        xsize_ = domain[2] + 2*num_halo;
        ysize_ = domain[1] + 2*num_halo;
        zsize_ = domain[0];

        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &num_ranks_);
        global_shape_ = {domain[0], domain[1] + 2 * num_halo, domain[2] + 2 * num_halo};

        size_ = setup_grid(); //[2,1], domain: [4,20,20]
        assert(domain[1] >= size_[0] && domain[2] >= size_[1]);
        setup_domain(domain, num_halo);
    }

    // Returns the MPI communicator use to setup this partitioner
    MPI_Comm comm() const
    {
        return comm_;
    }

    // Returns the number of halo points
    int num_halo() const
    {
        return num_halo_;
    }

    // Returns the periodicity of individual or all dimensions
    std::pair<bool, bool> periodic() const
    {
        return periodic_;
    }

    // Returns the rank of the current MPI worker
    int rank() const
    {
        return rank_;
    }

    // Returns the number of ranks that have been distributed by this partitioner
    int num_ranks() const
    {
        return num_ranks_;
    }

    // "Returns the shape of a local field (including halo points)
    std::vector<int> shape() const
    {
        return shape_;
    }

    // Returns the shape of a global field (including halo points)
    std::vector<int> global_shape() const
    {
        return global_shape_;
    }

    // Dimensions of the two-dimensional worker grid
    std::vector<int> size() const
    {
        return size_;
    }

    // Position of current rank on two-dimensional worker grid
    std::pair<int, int> position() 
    {
        return rank_to_position(rank_);
    }

    // Get the rank ID of a neighboring rank at a certain offset relative to the current rank
    int get_neighbor_rank(std::pair<int, int> offset) 
    {
        auto position = rank_to_position(rank_);
        auto pos_y = cyclic_offset(position.first, offset.first, size_[0], periodic_.first);
        auto pos_x = cyclic_offset(position.second, offset.second, size_[1], periodic_.second);
        return position_to_rank({pos_y, pos_x});
    }

    // Returns the rank of the left neighbor
    int left() 
    {
        return get_neighbor_rank({0, -1});
    }

    // Returns the rank of the right neighbor
    int right() 
    {
        return get_neighbor_rank({0, 1});
    }

    // Returns the rank of the top neighbor
    int top() 
    {
        return get_neighbor_rank({1, 0});
    }

    // Returns the rank of the bottom neighbor
    int bottom() 
    {
        return get_neighbor_rank({-1, 0});
    }

    // Scatter a global field from a root rank to the workers
    Storage3D<double> scatter(Storage3D<double> &field, int root = 0)
    {
        if (rank_ == root)
        {
            assert(field.zMax() == global_shape_[0]);
        }
        assert(0 <= root && root < num_ranks_);
        if (num_ranks_ == 1)
        {
            return field;
        }
        std::unique_ptr<std::vector<double>> sendbuf;

        int size_of_rank = max_shape_[2] * max_shape_[1] * max_shape_[0];
        if (rank_ == root)
        {
            sendbuf = std::make_unique<std::vector<double>>(num_ranks_ * max_shape_[2] * max_shape_[1] * max_shape_[0], 0.0);
            for (int rank = 0; rank < num_ranks_; ++rank)
            {
                int j_start = domains_[rank][0], i_start=domains_[rank][1], j_end=domains_[rank][2], i_end=domains_[rank][3];
                for (int k = 0; k < max_shape_[0]; ++k)
                {
                    for (int j = j_start; j < j_end; ++j)
                    {
                        for (int i = i_start; i < i_end; ++i)
                        {
                            (*sendbuf)[rank*size_of_rank + (i - i_start) + (j - j_start) * (i_end - i_start) + k * (i_end - i_start) * (j_end - j_start)] = field(i,j,k);
                        }
                    }
                }
            }
        }

        std::vector<double> recvbuf(max_shape_[2] * max_shape_[1] * max_shape_[0], 0.0);
        MPI_Barrier(comm_);
        MPI_Scatter(rank_ == root ? &(*sendbuf)[0] : nullptr, max_shape_[0] * max_shape_[1] * max_shape_[2], 
        MPI_DOUBLE, &recvbuf[0], max_shape_[0] * max_shape_[1] * max_shape_[2], MPI_DOUBLE, root, comm_);
        MPI_Barrier(comm_);


        int j_start=domain_[0], i_start=domain_[1], j_end=domain_[2], i_end=domain_[3];
        static Storage3D<double> recvbuf_3d (max_shape_[2] - 2*num_halo_, max_shape_[1] - 2*num_halo_, max_shape_[0], num_halo_, 0.0);
        for (int k = 0; k < max_shape_[0]; ++k)
        {
            for (int j = 0; j < j_end - j_start; ++j)
            {
                for (int i = 0; i < i_end - i_start; ++i)
                {
                    recvbuf_3d(i,j,k) = recvbuf[i + j * (i_end - i_start) + k * (i_end - i_start) * (j_end - j_start)];
                }
            }
        }

        MPI_Barrier(comm_);
        return recvbuf_3d;
    }

    // Gather a distributed fields from workers to a single global field on a root rank
    std::unique_ptr<Storage3D<double>> gather(Storage3D<double> &field, int root = 0)
    {
        assert(field.zMax() == shape_[0]);
        assert(-1 <= root && root < num_ranks_);
        if (num_ranks_ == 1)
        {
            return std::make_unique<Storage3D<double>>(field);
        }
        int j_start=domain_[0], i_start=domain_[1], j_end=domain_[2], i_end=domain_[3];
        std::vector<double> sendbuf (max_shape_[2] * max_shape_[1] * max_shape_[0], 0.0);
        for (int k = 0; k < shape_[0]; ++k)
        {
            for (int j = j_start; j < j_end; ++j)
            {
                for (int i = i_start; i < i_end; ++i)
                {
                    sendbuf[(i - i_start) + (j - j_start) * (i_end - i_start) + k * (i_end - i_start) * (j_end - j_start)] = field(i - i_start, j - j_start, k);
                }
            }
        }
        MPI_Barrier(comm_);
        std::unique_ptr<std::vector<double>> recvbuf;
        if (rank_ == root || root == -1)
        {
            recvbuf = std::make_unique<std::vector<double>>(num_ranks_ * max_shape_[2] * max_shape_[1] * max_shape_[0], 0.0);
        }

        MPI_Barrier(comm_);
        if (root > -1)
        {
            MPI_Gather(&sendbuf[0], max_shape_[0] * max_shape_[1] * max_shape_[2], MPI_DOUBLE, rank_ == root ? &(*recvbuf)[0] : nullptr, max_shape_[0] * max_shape_[1] * max_shape_[2], MPI_DOUBLE, root, comm_);
        }
        else
        {
            MPI_Allgather(&sendbuf[0], max_shape_[0] * max_shape_[1] * max_shape_[2], MPI_DOUBLE, rank_ == root ? &(*recvbuf)[0] : nullptr, max_shape_[0] * max_shape_[1] * max_shape_[2], MPI_DOUBLE, comm_);
        }
        MPI_Barrier(comm_);
        int size_of_rank = max_shape_[2] * max_shape_[1] * max_shape_[0];
        if (rank_ == root || root == -1)
        {
            std::unique_ptr<Storage3D<double>> global_field = std::make_unique<Storage3D<double>>(global_shape_[2] - 2*num_halo_, global_shape_[1] - 2*num_halo_, global_shape_[0], num_halo_, 0.0);
            for (int rank = 0; rank < num_ranks_; ++rank)
            {
                int j_start = domains_[rank][0], i_start=domains_[rank][1], j_end=domains_[rank][2], i_end=domains_[rank][3];
                for (int k = 0; k < max_shape_[0]; ++k)
                {
                    for (int j = j_start; j < j_end; ++j)
                    {
                        for (int i = i_start; i < i_end; ++i)
                        {
                            (*global_field)(i,j,k) = (*recvbuf)[rank*size_of_rank + (i - i_start) + (j - j_start) * (i_end - i_start) + k * (i_end - i_start) * (j_end - j_start)];
                        }
                    }
                }
            }
            MPI_Barrier(comm_);
            return global_field;
        }
        else
        {
            MPI_Barrier(comm_);
            return nullptr;
        }
    }

    // Return position of subdomain without halo on the global domain
    std::vector<int> compute_domain() const
    {
        return {domain_[0] + num_halo_, domain_[1] + num_halo_, domain_[2] - num_halo_, domain_[3] - num_halo_};
    }

private:
    MPI_Comm comm_;
    int num_halo_;
    int xsize_;
    int ysize_;
    int zsize_;
    std::pair<bool, bool> periodic_;
    int rank_;
    int num_ranks_;
    std::vector<int> global_shape_;
    std::vector<int> size_;
    std::vector<std::vector<int>> domains_;
    std::vector<std::vector<int>> shapes_;
    std::vector<int> domain_;
    std::vector<int> shape_;
    std::vector<int> max_shape_;

    // Distribute ranks onto a Cartesian grid of workers
    std::vector<int> setup_grid()
    {
        int ranks_x_size;
        for (int ranks_x = std::floor(std::sqrt(num_ranks_)); ranks_x > 0; --ranks_x)
        {
            if (num_ranks_ % ranks_x == 0)
            {
                ranks_x_size = ranks_x;
                break;
            }
        }
        return {num_ranks_ / ranks_x_size, ranks_x_size};
    }

    // Add offset with cyclic boundary conditions
    int cyclic_offset(int position, int offset, int size, bool periodic = true)
    {
        int pos = position + offset;
        if (periodic)
        {
            while (pos < 0)
            {
                pos += size;
            }
            while (pos > size - 1)
            {
                pos -= size;
            }
        }
        return -1 < pos && pos < size ? pos : -1;
    }

    // "Distribute the points of the computational grid onto the Cartesian grid of workers
    void setup_domain(std::vector<int> shape, int num_halo)
    {
        int size_z = shape[0];
        auto size_y = distribute_to_bins(shape[1], size_[0]);
        auto size_x = distribute_to_bins(shape[2], size_[1]);

        auto pos_y = cumsum(size_y, num_halo);
        auto pos_x = cumsum(size_x, num_halo);

        std::vector<std::vector<int>> domains;
        std::vector<std::vector<int>> shapes;
        for (int rank = 0; rank < num_ranks_; ++rank)
        {
            auto pos = rank_to_position(rank); 
            domains.push_back({pos_y[pos.first] - num_halo, pos_x[pos.second] - num_halo, pos_y[pos.first + 1] + num_halo, pos_x[pos.second + 1] + num_halo});
            shapes.push_back({size_z, domains[rank][2] - domains[rank][0], domains[rank][3] - domains[rank][1]});
        }
        domains_ = domains;
        shapes_ = shapes;

        domain_ = domains_[rank_]; 
        shape_ = shapes_[rank_];
        max_shape_ = find_max_shape(shapes_); 
    }

    // Distribute a number of elements to a number of bins
    std::vector<int> distribute_to_bins(int number, int bins)
    {
        int n = number / bins;
        std::vector<int> bin_size(bins, n);
        int extend = number - n * bins; 
        if (extend > 0)
        {
            int start_extend = bins / 2 - extend / 2;
            for (int i = start_extend; i < start_extend + extend; ++i)
            {
                bin_size[i] += 1;
            }
        }
        return bin_size;
    }

    // Cumulative sum with an optional initial value (default is zero)
    std::vector<int> cumsum(std::vector<int> array, int initial_value = 0)
    {
        std::vector<int> cumsum = {initial_value};
        for (int i : array)
        {
            cumsum.push_back(cumsum.back() + i); 
        }
        return cumsum;
    }

    // Find largest shape in a list of shapes
    std::vector<int> find_max_shape(const std::vector<std::vector<int>> &shapes)
    {
        if (shapes.empty())
        {
            return {};
        }

        std::vector<int> max_shape = shapes[0];
        for (size_t i = 1; i < shapes.size(); ++i)
        {
            for (size_t j = 0; j < max_shape.size(); ++j)
            {
                max_shape[j] = std::max(max_shape[j], shapes[i][j]);
            }
        }
        return max_shape;
    }

    // Find position of rank on worker grid
    std::pair<int, int> rank_to_position(int rank)
    {
        return {rank / size_[1], rank % size_[1]};
    }

    // Find rank given a position on the worker grid
    int position_to_rank(std::pair<int, int> position)
    {
        if (position.first == -1 || position.second == -1)
        {
            return -1;
        }
        else
        {
            return position.first * size_[1] + position.second;
        }
    }
};
