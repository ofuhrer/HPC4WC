#include <boost/python.hpp>
#include <string>
// #include <Eigen/Dense>

std::string generated() { return "HPC4WC"; }

BOOST_PYTHON_MODULE(dslgen) {
	using namespace boost::python;
	def("generated", generated);
}
