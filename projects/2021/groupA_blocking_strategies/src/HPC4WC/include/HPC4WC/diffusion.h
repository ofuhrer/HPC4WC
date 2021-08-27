#pragma once
#include <HPC4WC/field.h>
namespace HPC4WC {

/**
 * @brief Apply the diffusion equation to a field.
 * 
 * Applies the equation \f$ \frac {\partial f}{\partial t}=\alpha\nabla ^{2}f \f$.
 * It uses an improved laplace operator and doesn't use temporary fields wherever possible.
 * @attention Applies the diffusion equation only to indices i and j (not k).
 */
class Diffusion {
public:
    /**
     * @brief Apply the equation.
     * @param [inout] f The field to apply the diffusion equation to.
     * @param [in] alpha The coefficient in front of the nabla operator.
     */
    static void apply(Field& f, const double& alpha = 1e-3);
};

/**
 * @brief Apply the diffusion equation to a field.
 * 
 * Applies the equation \f$ \frac {\partial f}{\partial t}=\alpha\nabla ^{2}f \f$.
 * It uses a simplified approach but can benefit from blocking.
 * @attention Applies the diffusion equation only to indices i and j (not k).
 */
class SimpleDiffusion {
public:
    /**
     * @brief Apply the equation.
     * @param [inout] f The field to apply the diffusion equation to.
     * @param [in] alpha The coefficient in front of the nabla operator.
     */
    static void apply(Field& f, const double& alpha = 1e-3);

private:
    /**
     * @brief Applies a laplacian operator onto a field.
     * 
     * The laplacian is only applied with a size given by f_out.
     * Furthermore, the offset_i and offset_j values control the starting point,
     * meaning 0,0 will start at the top left block of f_in (without halo points).
     * 
     * @throws std::out_of_range if the offsets are wrong or f_out too big.
     * @throws std::logical_error if the two fields do not match in k / num halo points.
     * 
     * @param[in] f_in The field to perform the laplacian operator on.
     * @param[out] f_out The field where to write the laplacian of f_in.
     * @param[in] offset_i The offset in f_in in i direction.
     * @param[in] offset_j The offset in f_in in j direction.
     */
    static void laplacian(const Field& f_in, Field& f_out, Field::const_idx_t& offset_i = 0, Field::const_idx_t& offset_j = 0);

    /**
     * @brief Perform time integration on a field.
     * 
     * @throws std::logic_error if the fields do not match.
     * 
     * @param[in] f_in The field to integrate over time.
     * @param[out] f_out The output field.
     * @param[in] alpha The alpha value of the integration.
     */
    static void time_integration(const Field& f_in, Field& f_out, const double& alpha);
};

}  // namespace HPC4WC