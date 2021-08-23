#include <HPC4WC/config.h>
#include <HPC4WC/diffusion.h>
#include <omp.h>

namespace HPC4WC {

#define STENCIL(i, j, k)                                                                                                                                    \
    a1* f(i, j - 2, k) + a2* f(i - 1, j - 1, k) + a8* f(i, j - 1, k) + a2* f(i + 1, j - 1, k) + a1* f(i - 2, j, k) + a8* f(i - 1, j, k) + a20* f(i, j, k) + \
        a8* f(i + 1, j, k) + a1* f(i + 2, j, k) + a2* f(i - 1, j + 1, k) + a8* f(i, j + 1, k) + a2* f(i + 1, j + 1, k) + a1* f(i, j + 2, k)

void Diffusion::apply(Field& f, const double& alpha) {
    const double a1 = -1. * alpha;
    const double a2 = -2. * alpha;
    const double a8 = 8. * alpha;
    const double a20 = 1. - 20. * alpha;

    if (Config::BLOCK_I && Config::BLOCK_J) {
        // Blocking in both axis

        Eigen::MatrixXd tmp(Config::BLOCK_SIZE_I, Config::BLOCK_SIZE_J);

        Eigen::MatrixXd row_halo_old(f.num_halo(), f.num_j());  // top-bottom
        Eigen::MatrixXd row_halo_current(f.num_halo(), f.num_j());
        Eigen::MatrixXd col_halo(Config::BLOCK_SIZE_I - f.num_halo(), f.num_halo());  // left-right

#pragma omp parallel for shared(f) firstprivate(tmp, row_halo_current, row_halo_old, col_halo, a1, a2, a8, a20) default(none) \
    num_threads(Config::OPENMP_NUM_THREADS)

        for (Field::idx_t k = 0; k < f.num_k(); k++) {
            Field::idx_t block_i;
            Field::idx_t block_j;
            for (block_i = 0; block_i < f.num_i() - Config::BLOCK_SIZE_I + 1; block_i += Config::BLOCK_SIZE_I) {
                row_halo_old = row_halo_current;
                for (block_j = 0; block_j < f.num_j() - Config::BLOCK_SIZE_J + 1; block_j += Config::BLOCK_SIZE_J) {
                    for (Field::idx_t i = 0; i < Config::BLOCK_SIZE_I; i++) {
                        for (Field::idx_t j = 0; j < Config::BLOCK_SIZE_J; j++) {
                            tmp(i, j) = STENCIL(block_i + i + f.num_halo(), block_j + j + f.num_halo(), k);
                        }
                    }

                    if (block_j > 0) {
                        // write old col_halo
                        f.setFrom(col_halo, block_i + f.num_halo(), block_j, k);
                    }

                    row_halo_current.block(0, block_j, f.num_halo(), Config::BLOCK_SIZE_J) =
                        tmp.block(Config::BLOCK_SIZE_I - f.num_halo(), 0, f.num_halo(), Config::BLOCK_SIZE_J);  // bottom rows

                    col_halo = tmp.block(0, Config::BLOCK_SIZE_J - f.num_halo(), Config::BLOCK_SIZE_I - f.num_halo(), f.num_halo());  // right cols
                    f.setFrom(tmp.block(0, 0, Config::BLOCK_SIZE_I - f.num_halo(), Config::BLOCK_SIZE_J - f.num_halo()), block_i + f.num_halo(),
                              block_j + f.num_halo(), k);
                }

                // do the last j entries (column)
                if (block_j < f.num_j()) {
                    Field::const_idx_t num_cols = f.num_j() - block_j;
                    for (Field::idx_t j = block_j; j < f.num_j(); j++) {
                        for (Field::idx_t i = 0; i < Config::BLOCK_SIZE_I; i++) {
                            tmp(i, j - block_j) = STENCIL(i + block_i + f.num_halo(), j + f.num_halo(), k);
                        }
                    }

                    row_halo_current.block(0, block_j, f.num_halo(), num_cols) = tmp.block(Config::BLOCK_SIZE_I - f.num_halo(), 0, f.num_halo(), num_cols);

                    f.setFrom(tmp.block(0, 0, Config::BLOCK_SIZE_I - f.num_halo(), num_cols), block_i + f.num_halo(), block_j + f.num_halo(), k);
                }

                if (block_i > 0) {
                    // now we write tmp2_copy
                    f.setFrom(row_halo_old, block_i, f.num_halo(), k);
                }

                // do final col_halo
                f.setFrom(col_halo, block_i + f.num_halo(), block_j, k);
            }

            // do last rows (no blocking here in j direction, therefore we need a new array!)
            if (block_i < f.num_i()) {
                Eigen::MatrixXd tmp_(f.num_i() - block_i, f.num_j());

                for (Field::idx_t i = block_i; i < f.num_i(); i++) {
                    for (Field::idx_t j = 0; j < f.num_j(); j++) {
                        tmp_(i - block_i, j) = STENCIL(i + f.num_halo(), j + f.num_halo(), k);
                    }
                }
                f.setFrom(tmp_, block_i + f.num_halo(), +f.num_halo(), k);
            }

            // write final tmp2
            f.setFrom(row_halo_current, block_i, f.num_halo(), k);
        }
    } else if (Config::BLOCK_I) {
        Eigen::MatrixXd tmp(Config::BLOCK_SIZE_I, f.num_j());
        Eigen::MatrixXd row_halo(f.num_halo(), f.num_j());

#pragma omp parallel for shared(f) firstprivate(tmp, row_halo, a1, a2, a8, a20) default(none) num_threads(Config::OPENMP_NUM_THREADS)
        for (Field::idx_t k = 0; k < f.num_k(); k++) {
            // i blocking
            Field::idx_t block_i;
            for (block_i = 0; block_i < f.num_i() - Config::BLOCK_SIZE_I + 1; block_i += Config::BLOCK_SIZE_I) {
                for (Field::idx_t i = 0; i < Config::BLOCK_SIZE_I; i++) {
                    for (Field::idx_t j = 0; j < f.num_j(); j++) {
                        tmp(i, j) = STENCIL(block_i + i + f.num_halo(), j + f.num_halo(), k);
                    }
                }
                // write old row_halo
                if (block_i > 0) {
                    f.setFrom(row_halo, block_i, f.num_halo(), k);
                }

                row_halo = tmp.block(Config::BLOCK_SIZE_I - f.num_halo(), 0, f.num_halo(), f.num_j());
                f.setFrom(tmp.block(0, 0, Config::BLOCK_SIZE_I - f.num_halo(), f.num_j()), block_i + f.num_halo(), f.num_halo(), k);
            }

            // Do residual rows using simple loop
            if (block_i < f.num_i()) {
                Field::const_idx_t final_rows = f.num_i() - block_i;
                // do final rows
                for (Field::idx_t i = block_i; i < f.num_i(); i++) {
                    for (Field::idx_t j = 0; j < f.num_j(); j++) {
                        tmp(i - block_i, j) = STENCIL(i + f.num_halo(), j + f.num_halo(), k);
                    }
                }
                f.setFrom(tmp.block(0, 0, final_rows, f.num_j()), block_i + f.num_halo(), f.num_halo(), k);
            }

            // write final row_halo
            f.setFrom(row_halo, block_i, f.num_halo(), k);
        }
    } else if (Config::BLOCK_J) {
        Eigen::MatrixXd tmp(f.num_i(), Config::BLOCK_SIZE_J);
        Eigen::MatrixXd col_halo(f.num_i(), f.num_halo());

#pragma omp parallel for shared(f) firstprivate(tmp, col_halo, a1, a2, a8, a20) default(none) num_threads(Config::OPENMP_NUM_THREADS)
        for (Field::idx_t k = 0; k < f.num_k(); k++) {
            // i blocking
            Field::idx_t block_j;
            for (block_j = 0; block_j < f.num_j() - Config::BLOCK_SIZE_J + 1; block_j += Config::BLOCK_SIZE_J) {
                for (Field::idx_t j = 0; j < Config::BLOCK_SIZE_J; j++) {
                    for (Field::idx_t i = 0; i < f.num_i(); i++) {
                        tmp(i, j) = STENCIL(i + f.num_halo(), block_j + j + f.num_halo(), k);
                    }
                }
                // write old col_halo
                if (block_j > 0) {
                    f.setFrom(col_halo, f.num_halo(), block_j, k);
                }

                col_halo = tmp.block(0, Config::BLOCK_SIZE_J - f.num_halo(), f.num_i(), f.num_halo());
                f.setFrom(tmp.block(0, 0, f.num_i(), Config::BLOCK_SIZE_J - f.num_halo()), f.num_halo(), block_j + f.num_halo(), k);
            }

            // Do residual rows using simple loop
            if (block_j < f.num_j()) {
                Field::const_idx_t final_cols = f.num_j() - block_j;
                // do final rows
                for (Field::idx_t j = block_j; j < f.num_j(); j++) {
                    for (Field::idx_t i = 0; i < f.num_i(); i++) {
                        tmp(i, j - block_j) = STENCIL(i + f.num_halo(), j + f.num_halo(), k);
                    }
                }
                f.setFrom(tmp.block(0, 0, f.num_i(), final_cols), f.num_halo(), block_j + f.num_halo(), k);
            }

            // write final col_halo
            f.setFrom(col_halo, f.num_halo(), block_j, k);
        }
    } else {
        Eigen::MatrixXd tmp(f.num_i(), f.num_j());

#pragma omp parallel for shared(f) firstprivate(tmp, a1, a2, a8, a20) default(none) num_threads(Config::OPENMP_NUM_THREADS)
        for (Field::idx_t k = 0; k < f.num_k(); k++) {
            for (Field::idx_t i = 0; i < f.num_i(); i++) {
                for (Field::idx_t j = 0; j < f.num_j(); j++) {
                    tmp(i, j) = STENCIL(i + f.num_halo(), j + f.num_halo(), k);
                }
            }
            f.setFrom(tmp, f.num_halo(), f.num_halo(), k);
        }
    }
}

}  // namespace HPC4WC