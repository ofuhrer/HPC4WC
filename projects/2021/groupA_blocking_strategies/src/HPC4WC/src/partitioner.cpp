#include <HPC4WC/boundary_condition.h>
#include <HPC4WC/partitioner.h>

#include <iostream>

#ifndef DISABLE_PARTITIONER

namespace HPC4WC {

Partitioner::Partitioner(Field::const_idx_t& ni, Field::const_idx_t& nj, Field::const_idx_t& nk, Field::const_idx_t& num_halo)
    : m_ni(ni), m_nj(nj), m_num_halo(num_halo) {
    MPI_Comm_size(MPI_COMM_WORLD, &m_numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

    MPI_Dims_create(m_numRanks, 2, m_dimSize);

    int periods[2] = {1, 1};
    int reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, m_dimSize, periods, reorder, &m_comm);

    auto local_size = getLocalFieldSize(m_rank);

    m_field = std::make_shared<Field>(local_size.first, local_size.second, nk, m_num_halo);

    if (m_rank == 0) {
        m_globalField = std::make_shared<Field>(m_ni, m_nj, nk, m_num_halo);
    }
}

void Partitioner::init(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
}

void Partitioner::finalize() {
    MPI_Finalize();
}

FieldSPtr Partitioner::getField() {
    return m_field;
}

FieldSPtr Partitioner::getGlobalField() {
    return m_globalField;
}

void Partitioner::scatter() {
    if (m_rank == 0) {
        for (Field::idx_t k = 0; k < m_field->num_k(); k++) {
            for (int rank = 1; rank < m_numRanks; rank++) {
                auto localSizes = getLocalFieldSize(rank);
                std::vector<double> buf(localSizes.first * localSizes.second);

                Field::idx_t interior_ni, interior_nj;  // if partitioning of domain is exactly possible, use this indices
                interior_ni = m_ni / m_dimSize[0];
                interior_nj = m_nj / m_dimSize[1];

                int rank_coords[2];
                MPI_Cart_coords(m_comm, rank, 2, rank_coords);

                Field::idx_t start_row = rank_coords[0] * interior_ni;
                Field::idx_t start_col = rank_coords[1] * interior_nj;

                Field::idx_t idx = 0;
                for (Field::idx_t i = start_row; i < start_row + localSizes.first; ++i) {
                    for (Field::idx_t j = start_col; j < start_col + localSizes.second; ++j) {
                        buf[idx++] = m_globalField->operator()(i + m_num_halo, j + m_num_halo, k);
                    }
                }
                // send buf with its values to each rank
                MPI_Send(buf.data(), (int)buf.size(), MPI_DOUBLE, rank, k, m_comm);
            }
        }

        // copy over to fill in small data of rank 0
        auto localSizes = getLocalFieldSize(0);
        std::vector<double> buf(localSizes.first * localSizes.second);
        Field::idx_t interior_ni, interior_nj;  // if partitioning of domain is exactly possible, use this indices
        interior_ni = m_ni / m_dimSize[0];
        interior_nj = m_nj / m_dimSize[1];

        int my_coords[2];
        MPI_Cart_coords(m_comm, m_rank, 2, my_coords);

        Field::idx_t start_row = my_coords[0] * interior_ni;
        Field::idx_t start_col = my_coords[1] * interior_nj;
        for (Field::idx_t k = 0; k < m_field->num_k(); k++) {
            for (Field::idx_t i = start_row; i < start_row + localSizes.first; ++i) {
                for (Field::idx_t j = start_col; j < start_col + localSizes.second; ++j) {
                    m_field->operator()(i + m_num_halo - start_row, j + m_num_halo - start_col, k) =
                        m_globalField->operator()(i + m_num_halo, j + m_num_halo, k);
                }
            }
        }

    } else {
        auto localSizes = getLocalFieldSize(m_rank);
        std::vector<double> buf(localSizes.first * localSizes.second);

        for (Field::idx_t k = 0; k < m_field->num_k(); k++) {
            Field::idx_t idx = 0;
            MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, 0, k, m_comm, MPI_STATUS_IGNORE);
            // unpack buf into k plane
            for (Field::idx_t i = 0; i < localSizes.first; ++i) {
                for (Field::idx_t j = 0; j < localSizes.second; ++j) {
                    m_field->operator()(i + m_num_halo, j + m_num_halo, k) = buf[idx++];
                }
            }
        }
    }
}

void Partitioner::gather() {
    if (m_rank == 0) {
        // recv. from all to corresponding cells
        for (Field::idx_t k = 0; k < m_field->num_k(); k++) {
            for (int rank = 1; rank < m_numRanks; rank++) {
                auto localSizes = getLocalFieldSize(rank);
                std::vector<double> buf(localSizes.first * localSizes.second);

                Field::idx_t interior_ni, interior_nj;  // if partitioning of domain is exactly possible, use this indices
                interior_ni = m_ni / m_dimSize[0];
                interior_nj = m_nj / m_dimSize[1];

                int rank_coords[2];
                MPI_Cart_coords(m_comm, rank, 2, rank_coords);

                Field::idx_t start_row = rank_coords[0] * interior_ni;
                Field::idx_t start_col = rank_coords[1] * interior_nj;

                MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, rank, k, m_comm, MPI_STATUS_IGNORE);

                Field::idx_t idx = 0;
                for (Field::idx_t i = start_row; i < start_row + localSizes.first; ++i) {
                    for (Field::idx_t j = start_col; j < start_col + localSizes.second; ++j) {
                        m_globalField->operator()(i + m_num_halo, j + m_num_halo, k) = buf[idx++];
                    }
                }
            }
        }

        // copy over rank 0
        auto localSizes = getLocalFieldSize(0);
        std::vector<double> buf(localSizes.first * localSizes.second);
        Field::idx_t interior_ni, interior_nj;  // if partitioning of domain is exactly possible, use this indices
        interior_ni = m_ni / m_dimSize[0];
        interior_nj = m_nj / m_dimSize[1];

        int my_coords[2];
        MPI_Cart_coords(m_comm, m_rank, 2, my_coords);
        Field::idx_t start_row = my_coords[0] * interior_ni;
        Field::idx_t start_col = my_coords[1] * interior_nj;
        for (Field::idx_t k = 0; k < m_field->num_k(); k++) {
            for (Field::idx_t i = start_row; i < start_row + localSizes.first; ++i) {
                for (Field::idx_t j = start_col; j < start_col + localSizes.second; ++j) {
                    m_globalField->operator()(i + m_num_halo, j + m_num_halo, k) =
                        m_field->operator()(i + m_num_halo - start_row, j + m_num_halo - start_col, k);
                }
            }
        }

    } else {
        auto localSizes = getLocalFieldSize(m_rank);
        std::vector<double> buf(localSizes.first * localSizes.second);

        for (Field::idx_t k = 0; k < m_field->num_k(); k++) {
            Field::idx_t idx = 0;

            // pack data from k plane
            for (Field::idx_t i = 0; i < localSizes.first; ++i) {
                for (Field::idx_t j = 0; j < localSizes.second; ++j) {
                    buf[idx++] = m_field->operator()(i + m_num_halo, j + m_num_halo, k);
                }
            }

            MPI_Send(buf.data(), (int)buf.size(), MPI_DOUBLE, 0, k, m_comm);
        }
    }
}

void Partitioner::applyPeriodicBoundaryConditions() {
    // if we have only a single rank, we can perform the default periodic boundary conditions
    // and don't have to send data around over the bus.
    if (m_numRanks == 1) {
        PeriodicBoundaryConditions::apply(*m_field.get(), PeriodicBoundaryConditions::PERIODICITY::BOTH);
        return;
    }

    // get neighbours
    int top, bottom, right, left;

    MPI_Cart_shift(m_comm, 0, 1, &top, &bottom);
    MPI_Cart_shift(m_comm, 1, 1, &left, &right);

    // TAGS:
    // to top = 1
    // to bottom = 2
    // to left = 3
    // to right = 4

    Field::const_idx_t f_ni = m_field->num_i();
    Field::const_idx_t f_nj = m_field->num_j();
    Field::const_idx_t f_nk = m_field->num_k();

    std::vector<double> snd_buf_t(m_num_halo * f_nj * f_nk);
    std::vector<double> rcv_buf_t(m_num_halo * f_nj * f_nk);
    std::vector<double> snd_buf_b(m_num_halo * f_nj * f_nk);
    std::vector<double> rcv_buf_b(m_num_halo * f_nj * f_nk);
    std::vector<double> snd_buf_l(m_num_halo * (f_ni + 2 * m_num_halo) * f_nk);
    std::vector<double> rcv_buf_l(m_num_halo * (f_ni + 2 * m_num_halo) * f_nk);
    std::vector<double> snd_buf_r(m_num_halo * (f_ni + 2 * m_num_halo) * f_nk);
    std::vector<double> rcv_buf_r(m_num_halo * (f_ni + 2 * m_num_halo) * f_nk);

    // top halo --> recv from top, send to bottom

    // post Recv.

    MPI_Request request_top;
    MPI_Irecv(rcv_buf_t.data(), (int)rcv_buf_t.size(), MPI_DOUBLE, top, 2, m_comm, &request_top);
    MPI_Request request_bottom;
    MPI_Irecv(rcv_buf_b.data(), (int)rcv_buf_b.size(), MPI_DOUBLE, bottom, 1, m_comm, &request_bottom);
    MPI_Request request_left;
    MPI_Irecv(rcv_buf_l.data(), (int)rcv_buf_l.size(), MPI_DOUBLE, left, 4, m_comm, &request_left);
    MPI_Request request_right;
    MPI_Irecv(rcv_buf_r.data(), (int)rcv_buf_r.size(), MPI_DOUBLE, right, 3, m_comm, &request_right);

    // fill buffer
    Field::idx_t idx = 0;
    for (Field::idx_t k = 0; k < f_nk; k++) {
        for (Field::idx_t i = 0; i < m_num_halo; i++) {
            for (Field::idx_t j = 0; j < f_nj; j++) {
                snd_buf_t[idx++] = m_field->operator()(i + f_ni, j + m_num_halo, k);
            }
        }
    }
    //MPI_Sendrecv(snd_buf_t.data(), (int)snd_buf_t.size(), MPI_DOUBLE, bottom, 2, rcv_buf_t.data(), (int)rcv_buf_t.size(), MPI_DOUBLE, top, 2, m_comm, MPI_STATUS_IGNORE);

    MPI_Send(snd_buf_t.data(), (int)snd_buf_t.size(), MPI_DOUBLE, bottom, 2, m_comm);
    MPI_Wait(&request_top, MPI_STATUS_IGNORE);
    idx = 0;
    for (Field::idx_t k = 0; k < f_nk; k++) {
        for (Field::idx_t i = 0; i < m_num_halo; i++) {
            for (Field::idx_t j = 0; j < f_nj; j++) {
                m_field->operator()(i, j + m_num_halo, k) = rcv_buf_t[idx++];
            }
        }
    }

    // bottom halo --> recv from bottom, send to top
    // fill buffer
    idx = 0;
    for (Field::idx_t k = 0; k < f_nk; k++) {
        for (Field::idx_t i = 0; i < m_num_halo; i++) {
            for (Field::idx_t j = 0; j < f_nj; j++) {
                snd_buf_b[idx++] = m_field->operator()(i + m_num_halo, j + m_num_halo, k);
            }
        }
    }
    //MPI_Sendrecv(snd_buf_b.data(), (int)snd_buf_b.size(), MPI_DOUBLE, top, 1., rcv_buf_b.data(), (int)rcv_buf_b.size(), MPI_DOUBLE, bottom, 1, m_comm, MPI_STATUS_IGNORE);
    MPI_Send(snd_buf_b.data(), (int)snd_buf_b.size(), MPI_DOUBLE, top, 1, m_comm);
    MPI_Wait(&request_bottom, MPI_STATUS_IGNORE);
    idx = 0;
    for (Field::idx_t k = 0; k < f_nk; k++) {
        for (Field::idx_t i = 0; i < m_num_halo; i++) {
            for (Field::idx_t j = 0; j < f_nj; j++) {
                m_field->operator()(i + f_ni + m_num_halo, j + m_num_halo, k) = rcv_buf_b[idx++];
            }
        }
    }

    // left halo --> recv from left, send to right
    idx = 0;
    for (Field::idx_t k = 0; k < f_nk; k++) {
        for (Field::idx_t i = 0; i < f_ni + 2 * m_num_halo; i++) {
            for (Field::idx_t j = 0; j < m_num_halo; j++) {
                snd_buf_l[idx++] = m_field->operator()(i, j + f_nj, k);
            }
        }
    }
    //MPI_Sendrecv(snd_buf_l.data(), (int)snd_buf_l.size(), MPI_DOUBLE, right, 4, rcv_buf_l.data(), (int)rcv_buf_l.size(), MPI_DOUBLE, left, 4, m_comm, MPI_STATUS_IGNORE);
    MPI_Send(snd_buf_l.data(), (int)snd_buf_l.size(), MPI_DOUBLE, right, 4, m_comm);
    MPI_Wait(&request_left, MPI_STATUS_IGNORE);
    idx = 0;
    for (Field::idx_t k = 0; k < f_nk; k++) {
        for (Field::idx_t i = 0; i < f_ni + 2 * m_num_halo; i++) {
            for (Field::idx_t j = 0; j < m_num_halo; j++) {
                m_field->operator()(i, j, k) = rcv_buf_l[idx++];
            }
        }
    }
    // right halo --> recv from right, send to left
    idx = 0;
    for (Field::idx_t k = 0; k < f_nk; k++) {
        for (Field::idx_t i = 0; i < f_ni + 2 * m_num_halo; i++) {
            for (Field::idx_t j = 0; j < m_num_halo; j++) {
                snd_buf_r[idx++] = m_field->operator()(i, j + m_num_halo, k);
            }
        }
    }
    //MPI_Sendrecv(snd_buf_r.data(), (int)snd_buf_r.size(), MPI_DOUBLE, left, 3, rcv_buf_r.data(), (int)rcv_buf_r.size(), MPI_DOUBLE, right, 3, m_comm, MPI_STATUS_IGNORE);
    MPI_Send(snd_buf_r.data(), (int)snd_buf_r.size(), MPI_DOUBLE, left, 3, m_comm);
    MPI_Wait(&request_right, MPI_STATUS_IGNORE);
    idx = 0;
    for (Field::idx_t k = 0; k < f_nk; k++) {
        for (Field::idx_t i = 0; i < f_ni + 2 * m_num_halo; i++) {
            for (Field::idx_t j = 0; j < m_num_halo; j++) {
                m_field->operator()(i, j + m_num_halo + f_nj, k) = rcv_buf_r[idx++];
            }
        }
    }
}

const int Partitioner::rank() const {
    return m_rank;
}

std::pair<Field::idx_t, Field::idx_t> Partitioner::getLocalFieldSize(int rank) {
    int coords[2];
    MPI_Cart_coords(m_comm, rank, 2, coords);

    Field::idx_t local_ni, local_nj;
    local_ni = m_ni / m_dimSize[0];
    local_nj = m_nj / m_dimSize[1];
    if (coords[0] == m_dimSize[0] - 1) {
        local_ni = m_ni - (m_dimSize[0] - 1) * local_ni;
    }
    if (coords[1] == m_dimSize[1] - 1) {
        local_nj = m_nj - (m_dimSize[1] - 1) * local_nj;
    }
    return std::make_pair(local_ni, local_nj);
}

}  // namespace HPC4WC

#endif /* DISABLE_PARTITIONER */