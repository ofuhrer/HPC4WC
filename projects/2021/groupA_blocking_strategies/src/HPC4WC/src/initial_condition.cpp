#include <HPC4WC/initial_condition.h>

namespace HPC4WC {

void CubeInitialCondition::apply(Field& field) {
    using const_idx_t = Field::const_idx_t;
    const_idx_t num_halo = field.num_halo();
    const_idx_t i_1third = field.num_i() / 3 + num_halo;
    const_idx_t i_2third = 2 * field.num_i() / 3 + num_halo;
    const_idx_t j_1third = field.num_j() / 3 + num_halo;
    const_idx_t j_2third = 2 * field.num_j() / 3 + num_halo;
    const_idx_t k_1third = field.num_k() / 3;
    const_idx_t k_2third = 2 * field.num_k() / 3;

    for (Field::idx_t i = i_1third; i < i_2third; i++) {
        for (Field::idx_t j = j_1third; j < j_2third; j++) {
            for (Field::idx_t k = k_1third; k < k_2third; k++) {
                field(i, j, k) = 1.;
            }
        }
    }
}

void DiagonalInitialCondition::apply(Field& field) {
    Field::const_idx_t num_halo = field.num_halo();
    for (Field::idx_t i = 0; i < field.num_i(); i++) {
        for (Field::idx_t j = 0; j < field.num_j(); j++) {
            for (Field::idx_t k = 0; k < field.num_k(); k++) {
                if ((i + j) % 4 == 0) {
                    field(i + num_halo, j + num_halo, k) = 1.;
                }
            }
        }
    }
}

void XInitialCondition::apply(Field& field, Field::const_idx_t& x_width) {
    Field::const_idx_t num_halo = field.num_halo();
    for (Field::idx_t k = 0; k < field.num_k(); k++) {
        for (Field::idx_t i = 0; i < field.num_i(); i++) {
            for (Field::idx_t j = 0; j < field.num_j(); j++) {
                if (i == j) {
                    field(i + num_halo, j + num_halo, k) = 1.;
                } else if (i == field.num_j() - j - 1) {
                    field(i + num_halo, j + num_halo, k) = 1.;
                } else if (i > j && (i - j) < x_width) {
                    field(i + num_halo, j + num_halo, k) = 1.;
                } else if (j > i && (j - i) < x_width) {
                    field(i + num_halo, j + num_halo, k) = 1.;
                } else if (i > (field.num_j() - j - 1) && (i - (field.num_j() - j - 1)) < x_width) {
                    field(i + num_halo, j + num_halo, k) = 1.;
                } else if ((field.num_j() - j - 1) > i && ((field.num_j() - j - 1) - i) < x_width) {
                    field(i + num_halo, j + num_halo, k) = 1.;
                }
            }
        }
    }
}

}  // namespace HPC4WC