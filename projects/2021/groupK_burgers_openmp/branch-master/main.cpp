#include <Eigen/Dense>
#include <ProgramOptions.hxx>
#include <concepts>
#include <array>
#include <cassert>
#include <cmath>
#include <numbers>
#include <type_traits>
#include <limits>
#include <algorithm>
#include <utility>
#include <memory>
#include <chrono>
#include <sstream>
#include <iostream>
#include <iomanip>

using Scalar = double;
using TemplateParamScalar =
#if __cpp_nontype_template_args >= 201911
	Scalar
#else
	long int
#endif
;
// using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template<typename T>
concept Tensoroid = true; // CBA
template<typename T>
concept Matrixoid = requires(T const& m) {
	{ m(0, 0) } -> std::convertible_to<Scalar>;
	{ m.rows() } -> std::integral;
	{ m.cols() } -> std::integral;
	// more stuff but CBA
};
template<typename T>
concept Vectoroid = requires(T const& v) {
	{ v[0] } -> std::convertible_to<Scalar>;
	// more stuff but CBA
};

namespace Detail {
	template<typename T>
	struct IsEigenSequence : std::false_type {};
	template<typename FirstType, typename SizeType, typename IncrType>
	struct IsEigenSequence<Eigen::ArithmeticSequence<FirstType, SizeType, IncrType>> : std::true_type {};
}
template<typename T>
concept Sequenceoid = Detail::IsEigenSequence<T>::value;

template<typename T>
[[nodiscard]] constexpr T sq(T const& x) {
	return x * x;
}
[[nodiscard]] constexpr auto absi(std::signed_integral auto x) noexcept {
	assert(x != std::numeric_limits<decltype(x)>::min());
	return x < 0 ? -x : x;
}
[[nodiscard]] constexpr auto absi(std::unsigned_integral auto x) noexcept {
	return x;
}
[[nodiscard]] constexpr unsigned exp2i(unsigned x) noexcept {
	assert(x < std::numeric_limits<unsigned>::digits);
	return 1<<x;
}
inline constexpr Scalar pi = std::numbers::pi_v<Scalar>;

template<std::invocable Invocable>
struct ScopeGuard {
	Invocable invocable;
	[[nodiscard]] explicit constexpr ScopeGuard(Invocable&& invocable) noexcept(std::is_nothrow_move_constructible_v<Invocable>) : invocable(std::move(invocable)) {}
	[[nodiscard]] explicit constexpr ScopeGuard(Invocable const& invocable) noexcept(std::is_nothrow_copy_constructible_v<Invocable>) : invocable(invocable) {}
	constexpr ~ScopeGuard() noexcept(std::is_nothrow_invocable_v<Invocable>) { invocable(); }
};

template<Tensoroid Tensor, unsigned ranks, unsigned slicedRank, std::integral IndexType = int>
struct Rank1Slice {
	Tensor* tensor;
	std::array<IndexType, ranks> position;

	[[nodiscard]] constexpr Rank1Slice(Tensor& tensor, std::convertible_to<IndexType> auto const&... indices)
	: tensor(std::addressof(tensor)), position{ indices... } {
		static_assert(sizeof...(indices) == ranks);
	}

	// not multi-thread friendly
	[[nodiscard]] constexpr decltype(auto) operator[](IndexType i) const {
		assert(tensor);
		const_cast<Rank1Slice*>(this)->position[slicedRank] += i;
		const ScopeGuard restore([this, i]{ const_cast<Rank1Slice*>(this)->position[slicedRank] -= i; });
		return std::apply(*tensor, position);
	}
};
template<unsigned slicedRank>
[[nodiscard]] constexpr auto rank1Slice(Tensoroid auto& tensor, std::integral auto const&... indices) {
	return Rank1Slice<std::remove_reference_t<decltype(tensor)>, sizeof...(indices), slicedRank, std::common_type_t<std::remove_cvref_t<decltype(indices)>...>>(tensor, indices...);
}

template<template<typename, typename> typename F, typename I, typename... Args>
struct FoldR;
template<template<typename, typename> typename F, typename I, typename T>
struct FoldR<F, I, T> : std::type_identity<typename F<T, I>::type> {};
template<template<typename, typename> typename F, typename I, typename T, typename U, typename... Tail>
struct FoldR<F, I, T, U, Tail...> : std::type_identity<typename F<T, typename FoldR<F, I, U, Tail...>::type>::type> {};

template<std::integral auto index_, TemplateParamScalar factor_>
struct IndexCoeffPair {
	static constexpr auto index = index_;
	static constexpr auto factor = factor_;
};
template<typename T>
struct IsIndexCoeffPair : std::false_type {};
template<std::integral auto index, TemplateParamScalar factor>
struct IsIndexCoeffPair<IndexCoeffPair<index, factor>> : std::true_type {};
template<typename T>
concept IndexCoeff = IsIndexCoeffPair<T>::value;

template<typename Lhs, typename Rhs>
struct IndexMin : std::type_identity<std::conditional_t<(Lhs::index < Rhs::index), Lhs, Rhs>> {};
template<typename Lhs, typename Rhs>
struct IndexMax : std::type_identity<std::conditional_t<(Lhs::index > Rhs::index), Lhs, Rhs>> {};

enum class StencilType { asymmetric, symmetric, antisymmetric };
template<StencilType type, IndexCoeff... Pairs>
struct Stencil /* 1D only */ {
	static constexpr auto min_index = FoldR<IndexMin, IndexCoeffPair<std::numeric_limits<int>::max(), {}>, Pairs...>::type::index;
	static constexpr auto max_index = FoldR<IndexMax, IndexCoeffPair<std::numeric_limits<int>::min(), {}>, Pairs...>::type::index;

	static constexpr auto left   = type == StencilType::asymmetric ? min_index : -std::max(absi(min_index), max_index);
	static constexpr auto right  = type == StencilType::asymmetric ? max_index :  std::max(absi(min_index), max_index);
	static constexpr auto margin = sizeof...(Pairs) > 0 ? std::max(absi(left), right) : 0;

	[[nodiscard]] static constexpr auto apply(Vectoroid auto&& vector) {
		switch(type) {
		case StencilType::asymmetric:
			return ((Pairs::factor * vector[Pairs::index]) + ...);
		case StencilType::symmetric:
			return ((Pairs::factor * (Pairs::index ? vector[Pairs::index] + vector[-Pairs::index] : vector[Pairs::index])) + ...);
		case StencilType::antisymmetric:
			return ((Pairs::factor * (Pairs::index ? vector[Pairs::index] - vector[-Pairs::index] : vector[Pairs::index])) + ...);
		};
	}
};

class VectorField2 {
	Matrix uv[2];

public:
	[[nodiscard]] VectorField2() = default;
	[[nodiscard]] explicit VectorField2(Eigen::Index rows, Eigen::Index cols)
	: uv{ Matrix(rows, cols), Matrix(rows, cols) } {
	}
	[[nodiscard]] explicit VectorField2(Eigen::Index rows, Eigen::Index cols, Scalar fill)
	: uv{ Matrix::Constant(rows, cols, fill), Matrix::Constant(rows, cols, fill) } {
	}

	[[nodiscard]] constexpr Matrix const& operator[](int i) const noexcept { assert(0 <= i && i < 2); return uv[i]; }
	[[nodiscard]] constexpr Matrix      & operator[](int i)       noexcept { assert(0 <= i && i < 2); return uv[i]; }

	[[nodiscard]] constexpr Eigen::Index rows() const noexcept { return uv[0].rows(); }
	[[nodiscard]] constexpr Eigen::Index cols() const noexcept { return uv[0].cols(); }
};

using AdvectionStencil1 = Stencil<StencilType::antisymmetric,
	IndexCoeffPair<1, TemplateParamScalar( 45)>,
	IndexCoeffPair<2, TemplateParamScalar(- 9)>,
	IndexCoeffPair<3, TemplateParamScalar(  1)>>;
using AdvectionStencil2 = Stencil<StencilType::symmetric,
	IndexCoeffPair<0, TemplateParamScalar(-20)>,
	IndexCoeffPair<1, TemplateParamScalar( 15)>,
	IndexCoeffPair<2, TemplateParamScalar(- 6)>,
	IndexCoeffPair<3, TemplateParamScalar(  1)>>;

using DiffusionStencil = Stencil<StencilType::symmetric,
	IndexCoeffPair<0, TemplateParamScalar(-30)>,
	IndexCoeffPair<1, TemplateParamScalar( 16)>,
	IndexCoeffPair<2, TemplateParamScalar(- 1)>>;

constexpr int margin = std::max({ AdvectionStencil1::margin, AdvectionStencil2::margin, DiffusionStencil::margin });

[[nodiscard]] constexpr Scalar advection(Scalar d, Scalar w, Vectoroid auto const& phiSlice) {
	return
		(           w  * AdvectionStencil1::apply(phiSlice)
		 - std::abs(w) * AdvectionStencil2::apply(phiSlice)) / (60 * d);
}
template<std::integral auto advectRank>
void advection(Vectoroid auto const& d, Matrixoid auto const& w, Matrixoid auto const& phi, Matrixoid auto& out, std::invocable<Scalar&, Scalar> auto assignment) {
	const int nx = w.cols(), ny = w.rows();
	assert(phi.cols() == nx && phi.rows() == ny);
	assert(out.cols() == nx && out.rows() == ny);
	// #pragma omp parallel for schedule(static)
	for(int x = margin; x < nx - margin; ++x)
		for(int y = margin; y < ny - margin; ++y)
			assignment(out(y, x), advection(d[advectRank], w(y, x), rank1Slice<advectRank>(phi, y, x)));
}
void advection(Vectoroid auto const& d, VectorField2 const& uv, VectorField2& adv) {
	advection<0>(d, uv[0], uv[0], adv[0], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs  = rhs; });
	advection<1>(d, uv[1], uv[0], adv[0], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs += rhs; });

	advection<0>(d, uv[0], uv[1], adv[1], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs  = rhs; });
	advection<1>(d, uv[1], uv[1], adv[1], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs += rhs; });
}

[[nodiscard]] constexpr Scalar diffusion(Scalar d, Vectoroid auto const& phiSlice) {
	return 1 / (12 * sq(d)) * DiffusionStencil::apply(phiSlice);
}
template<std::integral auto diffuseRank>
void diffusion(Vectoroid auto const& d, Matrixoid auto const& phi, Matrixoid auto& out, std::invocable<Scalar&, Scalar> auto assignment) {
	const int nx = phi.cols(), ny = phi.rows();
	assert(out.cols() == nx && out.rows() == ny);
	// #pragma omp parallel for schedule(static)
	for(int x = margin; x < nx - margin; ++x)
		for(int y = margin; y < ny - margin; ++y)
			assignment(out(y, x), diffusion(d[diffuseRank], rank1Slice<diffuseRank>(phi, y, x)));
}
void diffusion(Vectoroid auto const& d, VectorField2 const& uv, VectorField2& diff) {
	diffusion<0>(d, uv[0], diff[0], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs  = rhs; });
	diffusion<1>(d, uv[0], diff[0], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs += rhs; });

	diffusion<0>(d, uv[1], diff[1], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs  = rhs; });
	diffusion<1>(d, uv[1], diff[1], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs += rhs; });
}

void rk_stage(
	Scalar mu,
	VectorField2 const& in_now,
	VectorField2 const& in_tmp,
	VectorField2& out,
	VectorField2& adv,
	VectorField2& diff,
	Scalar dt,
	Vectoroid auto const& d
) {
	advection(d, in_tmp, adv);
	diffusion(d, in_tmp, diff);
	out[0] = in_now[0] + dt * (-adv[0] + mu * diff[0]);
	out[1] = in_now[1] + dt * (-adv[1] + mu * diff[1]);
}

enum UseCase { zhao, hopf_cole };

[[nodiscard]] constexpr VectorField2 solution_factory(UseCase useCase, Scalar mu, Scalar t, Matrixoid auto const& x, Matrixoid auto const& y, Sequenceoid auto slice_x, Sequenceoid auto slice_y) {
	const int mi = slice_x.size() ? slice_x[slice_x.size() - 1] - slice_x[0] + 1 : 0;
	const int mj = slice_y.size() ? slice_y[slice_y.size() - 1] - slice_y[0] + 1 : 0;

	const Matrixoid auto x2d = x(slice_x).replicate(1, mj);
	const Matrixoid auto y2d = y(slice_y).transpose().replicate(mi, 1);

	const int nx = x2d.cols(), ny = x2d.rows();
	assert(y2d.cols() == nx && y2d.rows() == ny);

	using std::exp;
	VectorField2 result;
	switch(useCase) {
	case zhao:
		result[0] = -4 * mu * pi * exp(-5 * sq(pi) * mu * t) * (2 * pi * x2d).array().cos() * (pi * y2d).array().sin() / (2 + exp(-5 * sq(pi) * mu * t) * (2 * pi * x2d).array().sin() * (pi * y2d).array().sin());
		result[1] = -2 * mu * pi * exp(-5 * sq(pi) * mu * t) * (2 * pi * x2d).array().sin() * (pi * y2d).array().cos() / (2 + exp(-5 * sq(pi) * mu * t) * (2 * pi * x2d).array().sin() * (pi * y2d).array().sin());
		break;
	case hopf_cole:
		result[0] = 0.75 - 1 / (4 * (1 + (-Matrix::Constant(x2d.rows(), x2d.cols(), t) - 4 * x2d + 4 * y2d).array().exp() / (32 * mu)));
		result[1] = 0.75 + 1 / (4 * (1 + (-Matrix::Constant(x2d.rows(), x2d.cols(), t) - 4 * x2d + 4 * y2d).array().exp() / (32 * mu)));
		break;
	default:
		assert(false);
	}
	return result;
}
[[nodiscard]] constexpr decltype(auto) solution_factory(UseCase useCase, Scalar mu, Scalar t, Matrixoid auto const& x, Matrixoid auto const& y) {
	return solution_factory(useCase, mu, t, x, y, Eigen::seq(0, x.size() - 1), Eigen::seq(0, y.size() - 1));
}

[[nodiscard]] constexpr VectorField2 initial_solution(UseCase useCase, Scalar mu, Matrixoid auto const& x, Matrixoid auto const& y) {
	return solution_factory(useCase, mu, 0, x, y);
}

void enforce_boundary_conditions(UseCase useCase, Scalar mu, Scalar t, Matrixoid auto const& x, Matrixoid auto const& y, VectorField2& uv) {
	const int nx = x.size(), ny = y.size();

	{
		auto slice_x = Eigen::seq(0, margin - 1);
		auto slice_y = Eigen::seq(0, ny - 1);
		VectorField2 uv_ex = solution_factory(useCase, mu, t, x, y, slice_x, slice_y);
		uv[0](slice_x, slice_y) = std::move(uv_ex[0]);
		uv[1](slice_x, slice_y) = std::move(uv_ex[1]);
	}
	{
		auto slice_x = Eigen::seq(nx - margin, nx - 1);
		auto slice_y = Eigen::seq(0, ny - 1);
		VectorField2 uv_ex = solution_factory(useCase, mu, t, x, y, slice_x, slice_y);
		uv[0](slice_x, slice_y) = std::move(uv_ex[0]);
		uv[1](slice_x, slice_y) = std::move(uv_ex[1]);
	}
	{
		auto slice_x = Eigen::seq(margin, nx - margin - 1);
		auto slice_y = Eigen::seq(0, margin - 1);
		VectorField2 uv_ex = solution_factory(useCase, mu, t, x, y, slice_x, slice_y);
		uv[0](slice_x, slice_y) = std::move(uv_ex[0]);
		uv[1](slice_x, slice_y) = std::move(uv_ex[1]);
	}
	{
		auto slice_x = Eigen::seq(margin, nx - margin - 1);
		auto slice_y = Eigen::seq(ny - margin, ny - 1);
		VectorField2 uv_ex = solution_factory(useCase, mu, t, x, y, slice_x, slice_y);
		uv[0](slice_x, slice_y) = std::move(uv_ex[0]);
		uv[1](slice_x, slice_y) = std::move(uv_ex[1]);
	}
}

// TODO: only a makeshift because my stdlib doesn't implement this
template<typename CharT, typename Traits, typename Rep, typename Period>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, std::chrono::duration<Rep, Period> const& d) {
	std::basic_ostringstream<CharT, Traits> s;
	s.flags(os.flags());
	s.imbue(os.getloc());
	s.precision(os.precision());
	s << d.count();
	     if constexpr(std::is_same_v<typename Period::type, std::nano    >) s << "ns";
	else if constexpr(std::is_same_v<typename Period::type, std::micro   >) s << "µs";
	else if constexpr(std::is_same_v<typename Period::type, std::milli   >) s << "ms";
	else if constexpr(std::is_same_v<typename Period::type, std::ratio<1>>) s << "s";
	else {
		s << '[' << Period::type::num;
		if constexpr(Period::type::den != 1)
			s << '/' << Period::type::den;
		s << ']';
	}
	return os << s.str();
}

template<typename T>
struct WelfordVariance {
	std::size_t n = 0;
	T mean{};
	T m{};

	constexpr void update(T const& x) {
		++n;
		const T old_mean = mean;
		mean = ((n - 1) * old_mean + x) / n;
		m += (x - old_mean) * (x - mean);
	}

	[[nodiscard]] constexpr T variance() { return m / n; }
	[[nodiscard]] constexpr T sample_variance() { return m / (n - 1); }
};

struct BenchmarkReport {
	using Duration = std::chrono::duration<double, std::milli>;

	Duration best;
	Duration average;
	Duration stddev;
	unsigned runs;
};
std::ostream& operator<<(std::ostream& stream, BenchmarkReport const& report) {
	stream << "runtime: " << report.average << " ± " << report.stddev << "; best: " << report.best << "; n=" << report.runs << '\n';
	return stream;
}

template<std::invocable Invocable, typename Clock>
[[nodiscard]] BenchmarkReport benchmark(Invocable&& invocable, unsigned streak, Clock clock) {
	assert(streak > 0);
	using std::sqrt;
	BenchmarkReport result;
	WelfordVariance<BenchmarkReport::Duration::rep> variance;
	result.best = BenchmarkReport::Duration(std::numeric_limits<double>::max());
	for(unsigned current_streak = 0; current_streak < streak; ) {
		const auto start = clock.now();
		invocable();
		const auto end = clock.now();
		const auto duration = std::chrono::duration_cast<BenchmarkReport::Duration>(end - start);
		if(duration < result.best) {
			result.best = duration;
			current_streak = 0;
		} else {
			++current_streak;
		}
		variance.update(duration.count());
	}
	result.average = BenchmarkReport::Duration(variance.mean);
	result.stddev = BenchmarkReport::Duration(sqrt(variance.sample_variance()));
	result.runs = variance.n;
	return result;
}
template<std::invocable Invocable>
[[nodiscard]] BenchmarkReport benchmark(Invocable&& invocable, unsigned streak = 100) {
	if constexpr(std::chrono::high_resolution_clock::is_steady)
		return benchmark(std::forward<Invocable>(invocable), streak, std::chrono::high_resolution_clock{});
	else
		return benchmark(std::forward<Invocable>(invocable), streak, std::chrono::steady_clock{});
}

int main(int argc, char** argv)
try {
	unsigned int sideLength = 21;
	unsigned int iterationsOption = -1;
	unsigned int streak = 100;

	po::parser parser;
	auto& help = parser["help"]
		.abbreviation('?')
		.description("print this help screen");
	parser["sideLength"]
		.abbreviation('l')
		.description("set the grid side length (default: 21)")
		.bind(sideLength);
	parser["iterations"]
		.abbreviation('i')
		.description("set the grid side length (default: (sideLength-1)^2)")
		.bind(iterationsOption);
	parser["streak"]
		.abbreviation('s')
		.description("set the length of a streak required to escape the benchmarking algorithm (default: 100)")
		.bind(streak);

	if(!parser(argc, argv))
		return EXIT_FAILURE;

	if(help.was_set()) {
		std::cout << parser << '\n';
		return EXIT_SUCCESS;
	}

	if(sideLength < 1)
		throw std::range_error("sideLength shall be >= 1");

	if(streak < 1)
		throw std::range_error("streak shall be >= 1");

	// use case
	constexpr UseCase useCase = zhao;

	// diffusion coefficient
	const Scalar mu = 0.1;

	// time
	const Scalar cfl = 1;
	const Scalar timestep = cfl / sq(sideLength - 1);
	const int n_iter = iterationsOption != -1 ? iterationsOption : sq(sideLength - 1);

	// output
	const int print_period = 0;

	// grid
	const int nx = sideLength;
	const Vector x = Vector::LinSpaced(nx, 0, 1);
	const Scalar dx = 1.0 / (nx - 1);
	const int ny = sideLength;
	const Vector y = Vector::LinSpaced(ny, 0, 1);
	const Scalar dy = 1.0 / (ny - 1);

	// vector fields
	VectorField2 uv_now(ny, nx);
	VectorField2 uv_new(ny, nx);
	VectorField2 adv(ny, nx, 0);
	VectorField2 diff(ny, nx, 0);

	const auto rk_fractions = { 1.0 / 3, 1.0 / 2, 1.0 };

	const auto single_threaded_task = [&]{
		uv_now = VectorField2(ny, nx, 0);
		uv_new = initial_solution(useCase, mu, x, y);

		Scalar t = 0;
		for(int i = 0; i < n_iter; ++i) {
			uv_now = uv_new;

			for(const double rk_fraction : rk_fractions) {
				const Scalar dt = rk_fraction * timestep;
				rk_stage(mu, uv_now, uv_new, uv_new, adv, diff, dt, Vector2{ dx, dy });
				enforce_boundary_conditions(useCase, mu, t + dt, x, y, uv_new);
			}

			t += timestep;
			if(print_period > 0 && ((i+1) % print_period == 0 || i+1 == n_iter)) {
				VectorField2 uv_ex  = solution_factory(useCase, mu, t, x, y);
				const Scalar err_u = (uv_new[0].block(margin, margin, ny - 2 * margin, nx - 2 * margin) - uv_ex[0].block(margin, margin, ny - 2 * margin, nx - 2 * margin)).norm() * std::sqrt(dx * dy);
				const Scalar err_v = (uv_new[1].block(margin, margin, ny - 2 * margin, nx - 2 * margin) - uv_ex[1].block(margin, margin, ny - 2 * margin, nx - 2 * margin)).norm() * std::sqrt(dx * dy);
				std::cout << "Iteration " << std::right << std::setfill(' ') << std::setw(6) << i + 1;
				std::cout << std::scientific << std::setprecision(4);
				std::cout << ": ||u - uex|| = " << err_u << " m/s, ||v - vex|| = " << err_v << " m/s\n";
				std::cout << std::defaultfloat;
			}
		}
	};
	std::cout << "Single-threaded benchmark:\n" << benchmark(single_threaded_task, streak);
} catch(std::exception const& exception) {
	std::cerr << po::red << "exception";
	std::cerr << ": " << exception.what() << '\n';
	return EXIT_FAILURE;
}
