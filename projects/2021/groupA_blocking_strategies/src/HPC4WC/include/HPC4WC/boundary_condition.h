#pragma once
#include <HPC4WC/field.h>

namespace HPC4WC {
/**
 * @brief Periodic boundary conditions
 * 
 * Values will be copied over.
 * 
 * @attention Works only on i and j indices.
 */
class PeriodicBoundaryConditions {
public:
    enum PERIODICITY {ONLY_I, ONLY_J, BOTH};
    /**
     * @brief Apply periodic boundary conditions on a field.
     * @param[inout] field The field to apply the boundary conditions to.
     * @param[in] periodicity The periodicity to use. Not setting it will set 0 dirichlet-boundary-condition.
     */
    static void apply(Field& field, const PERIODICITY& periodicity=BOTH);
};

/**
 * @brief Dirichlet boundary conditions
 * 
 * Values will be set to a constant.
 */
class DirichletBoundaryConditions {
public:
    /**
     * @brief Apply dirichlet boundary conditions on a field.
     * @param[inout] field The field to apply the boundary conditions to.
     * @param[in] value The value to use.
     */
    static void apply(Field& field, const double& value = 0.);
};

}  // namespace HPC4WC