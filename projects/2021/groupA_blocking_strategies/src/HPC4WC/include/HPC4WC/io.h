#pragma once
#include <HPC4WC/field.h>

#include <iostream>

namespace HPC4WC {

/**
 * @brief Helper to write fields to a stream.
 */
class IO {
public:
    /**
     * @brief Write a field to a stream.
     * @param[inout] stream The stream to which the file should be written (can be std::cout, or i.e. std::ofstream)
     * @param[in] field The field to write.
     * @param[in] k The last index of the field, which means this writes only a i-j-plane.
     */
    static void write(std::ostream& stream, const Field& field, Field::const_idx_t& k);

    /**
     * @brief Write the full field to a stream.
     * 
     * It will be written as a continuous array, where the i-j-planes are stacked.
     * @param[inout] stream The stream to which the file should be written (can be std::cout, or i.e. std::ofstream)
     * @param[in] field The field to write.
     */
    static void write(std::ostream& stream, const Field& field);
};

}  // namespace HPC4WC