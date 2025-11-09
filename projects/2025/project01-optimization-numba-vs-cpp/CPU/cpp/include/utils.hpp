#ifndef UTILS_HPP
#define UTILS_HPP

#include "macros.hpp"

inline void compute_x_coeff(double dx, double dy, double etaA, double etaB, double eta1, double eta2,
                            double &vx1, double &vx2, double &vx3, double &vx4, double &vx5,
                            double &vy1, double &vy2, double &vy3, double &vy4) {
    vx1 = 2.0 * etaA / (dx * dx);
    vx2 = eta1 / (dy * dy);
    vx3 = -(eta1 + eta2) / (dy * dy) - 2.0 * (etaA + etaB) / (dx * dx);
    vx4 = eta2 / (dy * dy);
    vx5 = 2.0 * etaB / (dx * dx);
    vy1 = eta1 / (dx * dy);
    vy2 = -eta2 / (dx * dy);
    vy3 = -eta1 / (dx * dy);
    vy4 = eta2 / (dx * dy);
}

inline void compute_y_coeff(double dx, double dy, double etaA, double etaB, double eta1, double eta2,
                            double &vy1, double &vy2, double &vy3, double &vy4, double &vy5,
                            double &vx1, double &vx2, double &vx3, double &vx4) {
    vy1 = eta1 / (dx * dx);
    vy2 = 2.0 * etaA / (dy * dy);
    vy3 = -2.0 * (etaA + etaB) / (dy * dy) - (eta1 + eta2) / (dx * dx);
    vy4 = 2.0 * etaB / (dy * dy);
    vy5 = eta2 / (dx * dx);
    vx1 = eta1 / (dx * dy);
    vx2 = -eta1 / (dx * dy);
    vx3 = -eta2 / (dx * dy);
    vx4 = eta2 / (dx * dy);
}

inline void apply_vx_BC(double* vx, int nx1, int ny1, double BC) {
    for (int j = 0; j < nx1; ++j) {
        AT(vx, 0, j, nx1)          = -BC * AT(vx, 1, j, nx1);
        AT(vx, ny1 - 1, j, nx1)    = -BC * AT(vx, ny1 - 2, j, nx1);
    }
    for (int i = 0; i < ny1; ++i) {
        AT(vx, i, 0, nx1)          = 0.0;
        AT(vx, i, nx1 - 2, nx1)    = 0.0;
        AT(vx, i, nx1 - 1, nx1)    = 0.0;
    }
}

inline void apply_vy_BC(double* vy, int nx1, int ny1, double BC) {
    for (int i = 0; i < ny1; ++i) {
        AT(vy, i, 0, nx1)          = -BC * AT(vy, i, 1, nx1);
        AT(vy, i, nx1 - 1, nx1)    = -BC * AT(vy, i, nx1 - 2, nx1);
    }
    for (int j = 0; j < nx1; ++j) {
        AT(vy, 0, j, nx1)          = 0.0;
        AT(vy, ny1 - 2, j, nx1)    = 0.0;
        AT(vy, ny1 - 1, j, nx1)    = 0.0;
    }
}


#endif // UTILS_HPP