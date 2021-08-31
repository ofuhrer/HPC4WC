#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <array>

namespace np = boost::python::numpy;

using scalar_t = double;
using array_t = np::ndarray;
using bounds_t = boost::python::list;

inline std::array<std::size_t, 6> get_bounds(bounds_t const& i, bounds_t const& j, bounds_t const& k) {
    const std::size_t start_i = boost::python::extract<std::size_t>(i[0]);
    const std::size_t end_i = boost::python::extract<std::size_t>(i[1]);
    const std::size_t start_j = boost::python::extract<std::size_t>(j[0]);
    const std::size_t end_j = boost::python::extract<std::size_t>(j[1]);
    const std::size_t start_k = boost::python::extract<std::size_t>(k[0]);
    const std::size_t end_k = boost::python::extract<std::size_t>(k[1]);

    return {start_i, end_i, start_j, end_j, start_k, end_k};
}
