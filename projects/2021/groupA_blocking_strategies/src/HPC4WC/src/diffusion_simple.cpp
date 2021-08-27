#include <HPC4WC/config.h>
#include <HPC4WC/diffusion.h>

namespace HPC4WC {

void SimpleDiffusion::laplacian(const Field& f_in, Field& f_out, Field::const_idx_t& offset_i, Field::const_idx_t& offset_j) {
    if (f_out.num_k() != f_in.num_k() || f_out.num_halo() != f_in.num_halo()) {
        throw std::logic_error("SimpleDiffusion::laplacian: in/out fields do not have the same number of k/halopoints.");
    }
    if (f_out.num_i() > f_in.num_i() + f_in.num_halo() || f_out.num_j() > f_in.num_j() + f_in.num_halo()) {
        throw std::out_of_range("SimpleDiffusion::laplacian: The output field is too big.");
    }

    // Check for offset out of bounds
    if (f_out.num_i() + offset_i > f_in.num_i() + f_in.num_halo() || f_out.num_j() + offset_j > f_in.num_j() + f_in.num_halo()) {
        throw std::out_of_range("SimpleDiffusion::laplacian: Offset is out of range.");
    }
    if (offset_i < 0 && -offset_i < f_in.num_halo() - 1 || offset_j < 0 && -offset_j < f_in.num_halo() - 1) {
        throw std::out_of_range("SimpleDiffusion::laplacian: Offset is out of range (too negative).");
    }

    for (Field::idx_t k = 0; k < f_in.num_k(); k++) {
        for (Field::idx_t i_out = f_out.num_halo(); i_out < f_out.num_i() + f_out.num_halo(); i_out++) {
            for (Field::idx_t j_out = f_out.num_halo(); j_out < f_out.num_j() + f_out.num_halo(); j_out++) {
                f_out(i_out, j_out, k) = -4. * f_in(i_out + offset_i, j_out + offset_j, k) + f_in(i_out - 1 + offset_i, j_out + offset_j, k) +
                                         f_in(i_out + 1 + offset_i, j_out + offset_j, k) + f_in(i_out + offset_i, j_out - 1 + offset_j, k) +
                                         f_in(i_out + offset_i, j_out + 1 + offset_j, k);
            }
        }
    }
}

void SimpleDiffusion::time_integration(const Field& f_in, Field& f_out, const double& alpha) {
    if (f_in.num_halo() != f_out.num_halo() || f_in.num_i() != f_out.num_i() || f_in.num_j() != f_out.num_j() || f_in.num_k() != f_out.num_k()) {
        throw std::logic_error("Field::setFrom(other): Sizes do not match.");
    }
    for (Field::idx_t k = 0; k < f_in.num_k(); k++) {
        for (Field::idx_t i_out = f_out.num_halo(); i_out < f_out.num_i() + f_out.num_halo(); i_out++) {
            for (Field::idx_t j_out = f_out.num_halo(); j_out < f_out.num_j() + f_out.num_halo(); j_out++) {
                f_out(i_out, j_out, k) = f_out(i_out, j_out, k) - alpha * f_in(i_out, j_out, k);
            }
        }
    }
}

void SimpleDiffusion::apply(Field& f, const double& alpha) {
    Field::idx_t block_i, block_j;

    if (Config::BLOCK_I && Config::BLOCK_J) {
        block_i = Config::BLOCK_SIZE_I;
        block_j = Config::BLOCK_SIZE_J;
    } else if (Config::BLOCK_I) {
        block_i = Config::BLOCK_SIZE_I;
        block_j = f.num_j();
    } else if (Config::BLOCK_J) {
        block_i = f.num_i();
        block_j = Config::BLOCK_SIZE_J;
    } else {
        block_i = f.num_i();
        block_j = f.num_j();
    }

    if (f.num_i() % block_i != 0 || f.num_j() % block_j != 0) {
        throw std::logic_error("Block size does not match the field given.");
    }

    Field tmp1 = Field(block_i + 2, block_j + 2, f.num_k(), f.num_halo());
    Field tmp2 = Field(block_i, block_j, f.num_k(), f.num_halo());

    Field tmp3 = Field(f.num_i(), f.num_j(), f.num_k(), f.num_halo());

    for (Field::idx_t block_i_start = 0; block_i_start < f.num_i(); block_i_start += block_i) {
        for (Field::idx_t block_j_start = 0; block_j_start < f.num_j(); block_j_start += block_j) {
            laplacian(f, tmp1, block_i_start - 1, block_j_start - 1);
            laplacian(tmp1, tmp2, 1, 1);

            // copy data over to a big tmp field, where we "intermediate" store the results.
            tmp3.setFrom(tmp2, block_i_start, block_j_start);
        }
    }

    // finally do time integration from the big tmp field back to f.
    time_integration(tmp3, f, alpha);
}

}  // namespace HPC4WC