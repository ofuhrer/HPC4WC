#pragma once
#include <HPC4WC/field.h>

#include <iostream>

#ifndef DISABLE_PARTITIONER

#include <mpi.h>

namespace HPC4WC {

/**
 * @brief Space partitioning using a cartesian grid.
 * 
 * Uses MPI to exchange halo data.
 * 
 * @todo Check all MPI errors
 */
class Partitioner {
public:
    /**
     * @brief Initialize MPI.
     * 
     * @param[in] argc How many entries are in argv.
     * @param[in] argv Entries to pass to MPI.
     * @attention Must be called prior to any other code from this class!
     */
    static void init(int argc, char* argv[]);

    /**
     * @brief Finalize MPI.
     * 
     * @attention Must be called after any other code from this class!
     */
    static void finalize();

    /**
     * @brief Create a new %Partitioner.
     * 
     * Allocates a part of the field on each rank, corresponding to the cartesian grid.
     * On rank 0, this also allocates the global field.
     * 
     * @param[in] ni Size of the field in the i direction.
     * @param[in] nj Size of the field in the j direction.
     * @param[in] nk Size of the field in the k direction.
     * @param[in] num_halo Number of halo points (in i and j direction).
     */
    Partitioner(Field::const_idx_t& ni, Field::const_idx_t& nj, Field::const_idx_t& nk, Field::const_idx_t& num_halo);

    /**
     * @brief Default deconstructor
     */
    ~Partitioner() {}

    /**
     * @brief Get a pointer to the part of the global field belonging to this rank.
     * 
     * Uses the cartesian grid to determine which part is stored on this rank.
     * @return A pointer to the local field (only this rank).
     */
    FieldSPtr getField();

    /**
     * @brief Get a pointer to the full global field.
     * 
     * @return A pointer to the global field (only rank 0). If not on rank 0, return a nullptr.
     * @attention This only returns a valid field on rank 0! Otherwise returns a nullptr.
     */
    FieldSPtr getGlobalField();

    /**
     * @brief Scatter the global field across all ranks.
     * 
     * Using the cartesian grid, distributes the global field from rank 0 to all ranks
     * (where each rank gets the corresponding part of the global field).
     */
    void scatter();

    /**
     * @brief Gather the local fields into the global field on rank 0.
     * 
     * Using the cartesian grid, gathers the local field from all ranks to the global field on rank 0.
     */
    void gather();

    /**
     * @brief Halo Exchange
     * 
     * Applies periodic boundary conditions (halo exchange) in both i and j direction.
     * See PeriodicBoundaryConditions::apply with PeriodicBoundaryConditions::PERIODICITY::BOTH.
     * If only one rank is available, this function will fallback to PeriodicBoundaryConditions.
     */
    void applyPeriodicBoundaryConditions();

    /**
     * @brief Get the current MPI rank.
     * 
     * Small helper to get the MPI rank.
     * @return The rank of the caller.
     */
    const int rank() const;

private:
    /**
     * @brief Get the field size (i and j direction) for a given rank.
     * 
     * Returns the local field size, such that the field can be perfectly distributed (including smaller fields at the boundary).
     * @param[in] rank The rank for which to get the size.
     * @return The field size as a pair, i and j direction.
     */
    std::pair<Field::idx_t, Field::idx_t> getLocalFieldSize(int rank);

    MPI_Comm m_comm; ///< cartesian communicator
    int m_numRanks; ///< max number of ranks
    int m_rank; ///< current rank
    int m_dimSize[2] = {0, 0}; ///< order of ranks in the cartesian grid
    FieldSPtr m_field; ///< local field
    FieldSPtr m_globalField; ///< global field (only on rank 0, else nullptr)
    Field::const_idx_t m_ni, m_nj, m_num_halo;  ///< store global ni, global nj and num_halo
};

}  // namespace HPC4WC

#endif /* DISABLE_PARTITIONER */