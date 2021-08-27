#include <HPC4WC/field.h>

#include <stdexcept>

namespace HPC4WC {

Field::Field(const_idx_t& ni, const_idx_t& nj, const_idx_t& nk, const_idx_t& num_halo) : m_ni(ni), m_nj(nj), m_nk(nk), m_num_halo(num_halo) {
    if (nk <= 0 || ni <= 0 || nj <= 0 || num_halo < 0) {
        throw std::logic_error("Field(ni, nj, nk, num_halo): some indices are negative or zero.");
    }
    m_data.resize(nk, Eigen::MatrixXd::Zero(ni + 2 * num_halo, nj + 2 * num_halo));
}

Field::Field(const_idx_t& ni, const_idx_t& nj, const_idx_t& nk, const_idx_t& num_halo, const double& value)
    : m_ni(ni), m_nj(nj), m_nk(nk), m_num_halo(num_halo) {
    if (nk <= 0 || ni <= 0 || nj <= 0 || num_halo < 0) {
        throw std::logic_error("Field(ni, nj, nk, num_halo, value): some indices are negative or zero.");
    }
    m_data.resize(nk, Eigen::MatrixXd::Constant(ni + 2 * num_halo, nj + 2 * num_halo, value));
}

double Field::operator()(const_idx_t& i, const_idx_t& j, const_idx_t& k) const {
    if (k < 0 || k >= m_data.size() || i < 0 || i >= m_data[k].rows() || j < 0 || j >= m_data[k].cols()) {
        throw std::out_of_range("double Field::operator()(i, j, k) const: out of bounds.");
    }
    return m_data[k](i, j);
}
double& Field::operator()(const_idx_t& i, const_idx_t& j, const_idx_t& k) {
    if (k < 0 || k >= m_data.size() || i < 0 || i >= m_data[k].rows() || j < 0 || j >= m_data[k].cols()) {
        throw std::out_of_range("double& Field::operator()(i, j, k): out of bounds.");
    }
    return m_data[k](i, j);
}

void Field::setFrom(const Field& other) {
    if (m_nk != other.m_nk || m_ni != other.m_ni || m_nj != other.m_nj || m_num_halo != other.m_num_halo) {
        throw std::logic_error("Field::setFrom(other): Sizes do not match.");
    }
    for (idx_t k = 0; k < m_nk; k++) {
        m_data[k] = other.m_data[k];
    }
}

void Field::setFrom(const Field& other, Field::const_idx_t& offset_i, Field::const_idx_t& offset_j) {
    if (m_nk != other.m_nk) {
        throw std::logic_error("Field::setFrom(other, offset_i, offset_j): k-dimension does not match.");
    }
    for (Field::idx_t k = 0; k < m_nk; k++) {
        setFrom(other.m_data[k].block(m_num_halo, m_num_halo, other.m_ni, other.m_nj), offset_i + m_num_halo, offset_j + m_num_halo, k);
    }
}

void Field::setFrom(const Eigen::MatrixXd& ij_plane_part, const_idx_t& i, const_idx_t& j, const_idx_t& k) {
    if (k < 0 || k >= m_data.size() || i < 0 || i + ij_plane_part.rows() >= m_data[k].rows() || j < 0 || j + ij_plane_part.cols() >= m_data[k].cols()) {
        throw std::out_of_range("Field::setFrom(ij_plane_part, i, j, k): Out of bounds.");
    }
    m_data[k].block(i, j, ij_plane_part.rows(), ij_plane_part.cols()) = ij_plane_part;
}

bool Field::operator==(const Field& other) const {
    if (m_nk != other.m_nk || m_ni != other.m_ni || m_nj != other.m_nj || m_num_halo != other.m_num_halo)
        return false;

    for (idx_t k = 0; k < m_nk; k++) {
        if ((m_data[k] - other.m_data[k]).squaredNorm() > 1e-3)
            return false;
    }
    return true;
}

bool Field::operator!=(const Field& other) const {
    // rely on the other operator implemented.
    return !(*this == other);
}

}  // namespace HPC4WC