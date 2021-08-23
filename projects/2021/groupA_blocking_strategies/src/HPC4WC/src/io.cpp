#include <HPC4WC/config.h>
#include <HPC4WC/io.h>

namespace HPC4WC {

void IO::write(std::ostream& out, const Field& field, Field::const_idx_t& k) {
    using const_idx_t = Field::const_idx_t;
    const_idx_t num_halo = field.num_halo();
    for (Field::idx_t i = 0; i < field.num_i(); i++) {
        for (Field::idx_t j = 0; j < field.num_j(); j++) {
            out << field(i + num_halo, j + num_halo, k) << " ";
        }
        out << std::endl;
    }
}

void IO::write(std::ostream& out, const Field& field) {
    for (Field::idx_t k = 0; k < field.num_k(); k++) {
        write(out, field, k);
    }
}

}  // namespace HPC4WC