#include <array>
#include <Eigen/Core>

using scalar_t = double;
using array_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
using bounds_t = std::array<std::size_t, 2>;

inline std::array<std::size_t, 6> get_bounds(bounds_t const& i, bounds_t const& j, bounds_t const& k) {
    return {i[0], i[1], j[0], j[1], k[0], k[1]};
}
