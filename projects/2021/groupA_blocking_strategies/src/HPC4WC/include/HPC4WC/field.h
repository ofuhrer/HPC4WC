#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

namespace HPC4WC {

/**
 * @brief A field. Holds data.
 * 
 * Indices start at 0 and go up to nx + 2 * num_halo,
 * this means, the actual field starts at (num_halo, num_halo)
 */
class Field {
public:
    using idx_t = long;               ///< field index type
    using const_idx_t = const idx_t;  ///< field const index type

    /**
     * @brief Create a field.
     * @param[in] ni Number of field nodes in the first direction
     * @param[in] nj Number of field nodes in the second direction
     * @param[in] nk Number of field nodes in the third direction
     * @param[in] num_halo Number of halo points around the field.
     * @throws std::logic_error If an index is negative (num_halo) or smaller or equal to zero (ni, nj, nk).
     */
    Field(const_idx_t& ni, const_idx_t& nj, const_idx_t& nk, const_idx_t& num_halo);

    /**
     * @brief Create a field with a given value.
     * @param[in] ni Number of field nodes in the first direction
     * @param[in] nj Number of field nodes in the second direction
     * @param[in] nk Number of field nodes in the third direction
     * @param[in] num_halo Number of halo points around the field.
     * @param[in] value The value to set the field to.
     * @throws std::logic_error If an index is negative (num_halo) or smaller or equal to zero (ni, nj, nk).
     */
    Field(const_idx_t& ni, const_idx_t& nj, const_idx_t& nk, const_idx_t& num_halo, const double& value);

    /**
     * @brief Creating a field from another is not allowed. (DELETED)
     */
    Field(const Field&) = delete;
    /**
     * @brief Creating a field without a size is not allowed. (DELETED)
     */
    Field() = delete;

    /**
     * @brief Default deconstructor.
     */
    ~Field() {}

    /**
     * @brief Get access to a field variable.
     * 
     * @param[in] i The first index
     * @param[in] j The second index
     * @param[in] k The third index
     * @result The value at the given location.
     * @throws std::out_of_bounds if any index is out of bounds.
     */
    double operator()(const_idx_t& i, const_idx_t& j, const_idx_t& k) const;

    /**
     * @brief Get reference access to a field variable.
     * 
     * @param[in] i The first index
     * @param[in] j The second index
     * @param[in] k The third index
     * @result The value at the given location.
     * @throws std::out_of_bounds if any index is out of bounds.
     */
    double& operator()(const_idx_t& i, const_idx_t& j, const_idx_t& k);

    /**
     * @brief Update a field from another field.
     * @param[in] f The other field which is copied over.
     * @throws Error if the fields do not match.
     * @throws std::logic_error if the two fields have non-matching dimensions.
     */
    void setFrom(const Field& f);

    /**
     * @brief Set part of the field from another, given an offset.
     * 
     * This routine skips halopoints, meaning offset 0/0 places the data in the actual data field, not the halo.
     * Internally relies on Field::setFrom(const Eigen::MatrixXd,const_idx_t&, const_idx_t&, const_idx_t&)
     * and therefore might throw more errors if the offset's are off.
     * 
     * @throws std::logic_error If the k dimension does not match.
     * 
     * @param[in] f The field to get the data from.
     * @param[in] offset_i The offset in i direction.
     * @param[in] offset_j The offset in j direction.
     */
    void setFrom(const Field& f, Field::const_idx_t& offset_i, Field::const_idx_t& offset_j);

    /**
     * @brief Update a field at k level from a given plane part.
     * @param[in] ij_plane_part The other plane part which is copied over.
     * @param[in] i The first index to put the part in this field.
     * @param[in] j The second index to put the part in this field.
     * @param[in] k Where to put the other plane.
     * @throws std::out_of_range if the indices are wrong (out of bounds access)
     */
    void setFrom(const Eigen::MatrixXd& ij_plane_part, const_idx_t& i, const_idx_t& j, const_idx_t& k);

    /**
     * @brief Get number of points in the first direction.
     * @return Number of points.
     */
    const_idx_t num_i() const {
        return m_ni;
    }
    /**
     * @brief Get number of points in the second direction.
     * @return Number of points.
     */
    const_idx_t num_j() const {
        return m_nj;
    }
    /**
     * @brief Get number of points in the third direction.
     * @return Number of points.
     */
    const_idx_t num_k() const {
        return m_nk;
    }
    /**
     * @brief Get number of halo points in each direction.
     * @return Number of points.
     */
    const_idx_t num_halo() const {
        return m_num_halo;
    }

    /**
     * @brief Check two fields for equality.
     * @param[in] other The other field to compare against.
     * @return True if the two fields match.
     * 
     * @attention This does not any runtime checks on the dimensions, simply return false if they do not match.
     */
    bool operator==(const Field& other) const;

    /**
     * @brief Check two fields for inequality.
     * @param[in] other The other field to compare against.
     * @return True if the two fields do not match. 
     * 
     * @attention This does not any runtime checks on the dimensions, simply return false if they do not match.
     */
    bool operator!=(const Field& other) const;

private:
    idx_t m_ni;
    idx_t m_nj;
    idx_t m_nk;
    idx_t m_num_halo;

    std::vector<Eigen::MatrixXd> m_data;
};

using FieldSPtr = std::shared_ptr<Field>;

}  // namespace HPC4WC