#ifndef INTERPOLATION
#define INTERPOLATION
#include <algorithm>
#include <Eigen/Dense>
#include <vector> 
#include <iostream>


size_t get_Index_Insert(const std::vector<double>& array, double insertValue) 
{
    return std::find_if(array.cbegin(), array.cend(), [insertValue](double aValue) { return aValue > insertValue;}) - array.begin();
}

std::vector<double> linspace(double first, double last, size_t len) 
{
    std::vector<double> result(len);
    double step = (last-first) / (len - 1);
    for (size_t i=0; i<len; i++) { result[i] = first + i*step; }
    return result;
}

std::vector<std::vector<double>> dlinspace(size_t nx, size_t ny, double x0, double y0, double lx, double ly) 
{
    std::vector<std::vector<double>> result(nx * ny, std::vector<double>(2, 0));
    std::vector<double> x = linspace(x0, x0 + lx, nx);
    std::vector<double> y = linspace(y0, y0 + ly, ny);
    for (size_t i = 0; i < nx; i++)
    {
        for (size_t j = 0; j < ny; j++)
        {
            result[j + ny * i][0] = x[i];
            result[j + ny * i][1] = y[j];
        }
        
    }
    
    return result;
}


Eigen::VectorXd interpolate(std::size_t nx, std::size_t ny, const double x0, 
                            const double lx, const double y0, const double ly,
                            Eigen::MatrixXd const& centers, Eigen::MatrixXd const& valmat)
{
    // Coordinates of data points.
    double sx = lx / nx, sy = ly / ny;
    std::vector<double> x = linspace(x0 + 0.5 * sx, x0 + lx - 0.5 * sx, nx);
    std::vector<double> y = linspace(y0 + 0.5 * sy, y0 + ly - 0.5 * sy, ny);

    // Data values.
    Eigen::VectorXd result = Eigen::VectorXd::Zero(centers.rows());
    std::vector<double> val(valmat.data(), valmat.data() + valmat.size());
    double xk, yk, x1, x2, y1, y2, f11, f12, f21, f22, dx, dy = 0;
    size_t x2i, y2i = 0;
    bool setf11 = false;
    for (size_t k = 0; k < centers.rows(); k++)
    {
        setf11 = true;
        xk = centers(k, 0);
        yk = centers(k, 1);
        x2i = std::min(get_Index_Insert(x, xk), nx - 1); // sometimes returns 100
        y2i = std::min(get_Index_Insert(y, yk), ny - 1); // sometimes returns 100
        dx = (x2i == 0 || x2i == nx - 1) ? 0.5 * sx : sx;
        dy = (y2i == 0 || y2i == ny - 1) ? 0.5 * sy : sy;
        f22 = valmat(x2i, y2i);
        x2 = x[x2i];
        y2 = y[y2i];
        if (x2i == 0) {
            x1 = x0;
            f12 = 0;
            f11 = 0;
            setf11 = false;
        }
        else {
            x1 = x[x2i - 1];
            f12 = valmat(x2i - 1, y2i);
        }
        if (y2i == 0) {
            y1 = y0;
            f21 = 0;
            f11 = 0;
            setf11 = false;
        }
        else {
            y1 = y[y2i - 1];
            f21 = valmat(x2i, y2i - 1);
        }
        if (setf11) {
            f11 = valmat(x2i - 1, y2i - 1);
        }
        result(k) = f11 / dx / dy * (x2 - xk) * (y2 - yk) 
                  + f21 / dx / dy * (xk - x1) * (y2 - yk) 
                  + f12 / dx / dy * (x2 - xk) * (yk - y1)
                  + f22 / dx / dy * (xk - x1) * (yk - y1);
    }
    // Get interpolated value at arbitrary location.
    return result;
}

Eigen::VectorXd interpolate(std::size_t nx, std::size_t ny, std::shared_ptr<Mesh> mesh, Eigen::MatrixXd const& valmat)
{
    Eigen::VectorXd result = Eigen::VectorXd::Zero(nx * ny);
    size_t k = 0;
    for (size_t i = 0; i < nx; i++)
    {
        for (size_t j = 0; j < ny; j++)
        {
            k = mesh->get_inv_gilbert(i, j);
            result(k) = valmat(i, j);
        }   
    }
    return result;
}


Eigen::VectorXd interpolate(const double x0, const double lx, 
                            const double y0, const double ly,
                            Eigen::MatrixXd const& valmat, 
                            std::shared_ptr<Mesh> meshptr)
{
    Eigen::MatrixXd centers = meshptr->get_centers();
    size_t nx = valmat.rows();
    size_t ny = valmat.cols();
    if (meshptr->get_has_inv_gilbert()) {
        return interpolate(nx, ny, meshptr, valmat);
    }
    else {
        return interpolate(nx, ny, x0, lx, y0, ly, centers, valmat);
    }
}

#endif // INTERPOLATION
