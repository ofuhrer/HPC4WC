#include <Eigen/Dense>
#include <upcxx/upcxx.hpp>

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
#include <vector>
#include <sstream>

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
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
concept VectorField = requires(T const& m) {
	{ m[0](0, 0) } -> std::convertible_to<Scalar>;
	{ m[0] } -> Matrixoid;
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

enum UseCase { zhao, hopf_cole };

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

[[nodiscard]] constexpr VectorField2 solution_factory(UseCase useCase, Scalar mu, Scalar t, Matrixoid auto const& x, Matrixoid auto const& y, Sequenceoid auto slice_x, Sequenceoid auto slice_y) {
	const int mi = slice_x.size() ? slice_x[slice_x.size() - 1] - slice_x[0] + 1 : 0;
	const int mj = slice_y.size() ? slice_y[slice_y.size() - 1] - slice_y[0] + 1 : 0;

	const Matrixoid auto x2d = x(slice_x).transpose().replicate(mj, 1);
	const Matrixoid auto y2d = y(slice_y).replicate(1, mi);

	const int nx = x2d.cols(), ny = x2d.rows();
    assert(nx == slice_x.size());
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

Vector padVector(Vectoroid auto input, int targetLength, Scalar paddingValue){
    assert(targetLength >= input.rows());
    Vector padded(targetLength);
    padded << input, paddingValue * Vector::Ones(targetLength - input.rows());
    return padded;
}

struct DistributedConfig {

    explicit DistributedConfig(int32_t globalWidth, int32_t globalHeight, int32_t ranksWidth, int32_t ranksHeight, int32_t rankIndex, int32_t margin) :
        RANKS_TOTAL(ranksWidth * ranksHeight),
        RANKS_WIDTH(ranksWidth),
        RANKS_HEIGHT(ranksHeight),
        RANK_INDEX(rankIndex),
        RANK_X(rankIndex % ranksWidth),
        RANK_Y(rankIndex / ranksWidth),
        GLOBAL_WIDTH(globalWidth),
        GLOBAL_HEIGHT(globalHeight),
        LOCAL_WIDTH((globalWidth + ranksWidth - 1) / ranksWidth),
        LOCAL_HEIGHT((globalHeight + ranksHeight - 1) / ranksHeight),
        STORED_WIDTH(LOCAL_WIDTH + 2 * margin),
        STORED_HEIGHT(LOCAL_HEIGHT + 2 * margin),
        MARGIN(margin),
        DX(1.0 / (globalWidth - 1)),
        DY(1.0 / (globalHeight - 1)),
        GLOBAL_X(Vector::LinSpaced(globalWidth, 0, 1)),
        GLOBAL_Y(Vector::LinSpaced(globalHeight, 0, 1)),
        LOCAL_X(padVector(GLOBAL_X, LOCAL_WIDTH * RANKS_WIDTH, 1.0).segment(RANK_X * LOCAL_WIDTH, LOCAL_WIDTH)),
        LOCAL_Y(padVector(GLOBAL_Y, LOCAL_HEIGHT * RANKS_HEIGHT, 1.0).segment(RANK_Y * LOCAL_HEIGHT, LOCAL_HEIGHT))
        {
            USED_WIDTH = LOCAL_WIDTH;
            if (RANK_X == RANKS_WIDTH - 1) {
                USED_WIDTH = GLOBAL_WIDTH - RANK_X * LOCAL_WIDTH;
            }
            USED_HEIGHT = LOCAL_HEIGHT;
            if (RANK_Y == RANKS_HEIGHT - 1) {
                USED_HEIGHT = GLOBAL_HEIGHT - RANK_Y * LOCAL_HEIGHT;
            }
            if (USED_WIDTH < MARGIN || USED_HEIGHT < MARGIN) {
                std::cerr << "Invalid segmentation " << USED_WIDTH << "x" << USED_HEIGHT << "\n";
                std::abort();
            }
        }

    // Number of rank matrix blocks
    int32_t RANKS_TOTAL;

    // Number of rank matrix blocks per global matrix width
    int32_t RANKS_WIDTH;

    // Number of rank matrix blocks per global matrix height
    int32_t RANKS_HEIGHT;

    // Index of local rank
    int32_t RANK_INDEX;

    // X coordinate of local rank's matrix block
    int32_t RANK_X;

    // Y coordinate of local rank's matrix block
    int32_t RANK_Y;

    // Width of entire matrix that is being distributed
    int32_t GLOBAL_WIDTH;

    // Height of entire matrix that is being distributed
    int32_t GLOBAL_HEIGHT;

    // Width of matrix block that is local to this rank
    int32_t LOCAL_WIDTH;

    // Height of matrix block that is local to this rank
    int32_t LOCAL_HEIGHT;

    // Width of matrix block that is used on this rank, i.e. without the extra cells on the right border
    int32_t USED_WIDTH;

    // Height of matrix block that is used on this rank, i.e. without the extra cells on the bottom border
    int32_t USED_HEIGHT;

    // Width of matrix that is local to this rank including margins
    int32_t STORED_WIDTH;

    // Height of matrix that is local to this rank including margins
    int32_t STORED_HEIGHT;

    // Margin around each local matrix for communication
    int32_t MARGIN;

    // X coordinate spacing
	Scalar DX;

    // Y coordinate spacing
	Scalar DY;

    // X coordinates of entire global matrix
	Vector GLOBAL_X;

    // Y coordinates of entire global matrix
	Vector GLOBAL_Y;

    // X coordinates of local matrix block
	Vector LOCAL_X;

    // Y coordinates of local matrix block
	Vector LOCAL_Y;

};

class DistributedVectorField2 {
    DistributedConfig cfg;
    std::reference_wrapper<std::vector<std::array<upcxx::global_ptr<Scalar>, 2>>> allMatrices;
    std::vector<Eigen::Map<Matrix, Eigen::Aligned128>> uv;

public:
	[[nodiscard]] explicit DistributedVectorField2(const VectorField2& o, std::vector<std::array<upcxx::global_ptr<Scalar>, 2>>& allMatrices, DistributedConfig config) :
        cfg(std::move(config)),
        allMatrices(allMatrices) {

        for (Eigen::Index dim = 0; dim < 2; dim++) {
            assert(cfg.GLOBAL_WIDTH == o[dim].cols() && cfg.GLOBAL_HEIGHT == o[dim].rows());
            allMatrices[cfg.RANK_INDEX][dim] = upcxx::allocate<Scalar>(cfg.STORED_WIDTH * cfg.STORED_HEIGHT, 256);
            uv.emplace_back(Eigen::Map<Matrix, Eigen::Aligned128>(allMatrices[cfg.RANK_INDEX][dim].local(), cfg.STORED_HEIGHT, cfg.STORED_WIDTH));
            for (Eigen::Index y = cfg.LOCAL_HEIGHT * cfg.RANK_Y; y < cfg.GLOBAL_HEIGHT && y < cfg.LOCAL_HEIGHT * (cfg.RANK_Y + 1); ++y) {
                std::copy(o[dim].data() + y * cfg.GLOBAL_WIDTH + cfg.RANK_X * cfg.LOCAL_WIDTH,
                          o[dim].data() + y * cfg.GLOBAL_WIDTH + std::min(cfg.GLOBAL_WIDTH, (cfg.RANK_X + 1) * cfg.LOCAL_WIDTH),
                          uv[dim].data() + cfg.MARGIN + (y - cfg.RANK_Y * cfg.LOCAL_HEIGHT + cfg.MARGIN) * cfg.STORED_WIDTH);
            }
        }
	}

    /**
     * @return if x, y are in bounds
     */
    [[nodiscard]] bool inBounds(Eigen::Index x, Eigen::Index y) {
        if (x < 0) return false;
        if (x >= cfg.RANKS_WIDTH) return false;
        if (y < 0) return false;
        if (y >= cfg.RANKS_HEIGHT) return false;
        return true;
    }

    /**
     * Synchronize borders with a neighbour given an offset
     * @return future for synchronization
     */
    [[nodiscard]] upcxx::future<> synchronizeRight(UseCase useCase, Scalar mu, Scalar t) {
        Eigen::Index otherX = cfg.RANK_X + 1;
        Eigen::Index otherY = cfg.RANK_Y;
        upcxx::future<> fut = upcxx::make_future();

        if (inBounds(otherX, otherY)) {
            for (Eigen::Index dim = 0; dim < 2; dim++) {
                upcxx::global_ptr<Scalar> otherStart = allMatrices.get()[otherY * cfg.RANKS_WIDTH + otherX][dim] + cfg.STORED_WIDTH * cfg.MARGIN + cfg.MARGIN;
                auto localStart = uv[dim].data() + cfg.USED_WIDTH + cfg.MARGIN + cfg.STORED_WIDTH * cfg.MARGIN;
                fut = upcxx::when_all(fut, upcxx::rget_strided<2>(otherStart,
                                                                  {{sizeof(Scalar), sizeof(Scalar) * cfg.STORED_WIDTH}},
                                                                  localStart,
                                                                  {{sizeof(Scalar), sizeof(Scalar) * cfg.STORED_WIDTH}}, {{cfg.MARGIN, cfg.USED_HEIGHT}}));
            }
        } else {

            auto slice_x = Eigen::seq(cfg.USED_WIDTH - margin, cfg.USED_WIDTH - 1);
            auto slice_y = Eigen::seq(0, cfg.USED_HEIGHT - 1);
            VectorField2 uv_ex = solution_factory(useCase, mu, t, cfg.LOCAL_X, cfg.LOCAL_Y, slice_x, slice_y);
            uv[0].block(cfg.MARGIN, cfg.MARGIN, cfg.USED_HEIGHT, cfg.USED_WIDTH)(slice_y, slice_x) = std::move(uv_ex[0]);
            uv[1].block(cfg.MARGIN, cfg.MARGIN, cfg.USED_HEIGHT, cfg.USED_WIDTH)(slice_y, slice_x) = std::move(uv_ex[1]);
        }
        return fut;
    }

    [[nodiscard]] upcxx::future<> synchronizeLeft(UseCase useCase, Scalar mu, Scalar t) {
        Eigen::Index otherX = cfg.RANK_X - 1;
        Eigen::Index otherY = cfg.RANK_Y;
        upcxx::future<> fut = upcxx::make_future();

        if (inBounds(otherX, otherY)) {

            for (Eigen::Index dim = 0; dim < 2; dim++) {
                upcxx::global_ptr<Scalar> otherStart = allMatrices.get()[otherY * cfg.RANKS_WIDTH + otherX][dim] + cfg.LOCAL_WIDTH + cfg.STORED_WIDTH * cfg.MARGIN;
                auto localStart = uv[dim].data() + cfg.STORED_WIDTH * cfg.MARGIN;
                fut = upcxx::when_all(fut, upcxx::rget_strided<2>(otherStart,
                                                                  {{sizeof(Scalar), sizeof(Scalar) * cfg.STORED_WIDTH}},
                                                                  localStart,
                                                                  {{sizeof(Scalar), sizeof(Scalar) * cfg.STORED_WIDTH}}, {{cfg.MARGIN, cfg.USED_HEIGHT}}));
            }
        } else {

            auto slice_x = Eigen::seq(0, cfg.MARGIN - 1);
            auto slice_y = Eigen::seq(0, cfg.USED_HEIGHT - 1);
            VectorField2 uv_ex = solution_factory(useCase, mu, t, cfg.LOCAL_X, cfg.LOCAL_Y, slice_x, slice_y);
            uv[0].block(cfg.MARGIN, cfg.MARGIN, cfg.USED_HEIGHT, cfg.USED_WIDTH)(slice_y, slice_x) = std::move(uv_ex[0]);
            uv[1].block(cfg.MARGIN, cfg.MARGIN, cfg.USED_HEIGHT, cfg.USED_WIDTH)(slice_y, slice_x) = std::move(uv_ex[1]);
        }
        return fut;
    }

    [[nodiscard]] upcxx::future<> synchronizeUp(UseCase useCase, Scalar mu, Scalar t) {
        Eigen::Index otherX = cfg.RANK_X;
        Eigen::Index otherY = cfg.RANK_Y - 1;
        upcxx::future<> fut = upcxx::make_future();
        if (inBounds(otherX, otherY)) {

            for (Eigen::Index dim = 0; dim < 2; dim++) {
                upcxx::global_ptr<Scalar> otherStart = allMatrices.get()[otherY * cfg.RANKS_WIDTH + otherX][dim] + cfg.MARGIN + cfg.STORED_WIDTH * cfg.LOCAL_HEIGHT;
                auto localStart = uv[dim].data() + cfg.MARGIN;
                fut = upcxx::when_all(fut, upcxx::rget_strided<2>(otherStart,
                                                                  {{sizeof(Scalar), sizeof(Scalar) * cfg.STORED_WIDTH}},
                                                                  localStart,
                                                                  {{sizeof(Scalar), sizeof(Scalar) * cfg.STORED_WIDTH}}, {{cfg.USED_WIDTH, cfg.MARGIN}}));
            }
        } else {

            auto slice_x = Eigen::seq(0, cfg.USED_WIDTH - 1);
            auto slice_y = Eigen::seq(0, cfg.MARGIN - 1);
            VectorField2 uv_ex = solution_factory(useCase, mu, t, cfg.LOCAL_X, cfg.LOCAL_Y, slice_x, slice_y);
            uv[0].block(cfg.MARGIN, cfg.MARGIN, cfg.USED_HEIGHT, cfg.USED_WIDTH)(slice_y, slice_x) = std::move(uv_ex[0]);
            uv[1].block(cfg.MARGIN, cfg.MARGIN, cfg.USED_HEIGHT, cfg.USED_WIDTH)(slice_y, slice_x) = std::move(uv_ex[1]);
        }
        return fut;
    }

    [[nodiscard]] upcxx::future<> synchronizeDown(UseCase useCase, Scalar mu, Scalar t) {
        Eigen::Index otherX = cfg.RANK_X;
        Eigen::Index otherY = cfg.RANK_Y + 1;
        upcxx::future<> fut = upcxx::make_future();
        if (inBounds(otherX, otherY)) {

            for (Eigen::Index dim = 0; dim < 2; dim++) {
                upcxx::global_ptr<Scalar> otherStart = allMatrices.get()[otherY * cfg.RANKS_WIDTH + otherX][dim] + cfg.MARGIN + cfg.MARGIN * cfg.STORED_WIDTH;
                auto localStart = uv[dim].data() + cfg.MARGIN + cfg.STORED_WIDTH * (cfg.MARGIN + cfg.USED_HEIGHT);
                fut = upcxx::when_all(fut, upcxx::rget_strided<2>(otherStart,
                                                                  {{sizeof(Scalar), sizeof(Scalar) * cfg.STORED_WIDTH}},
                                                                  localStart,
                                                                  {{sizeof(Scalar), sizeof(Scalar) * cfg.STORED_WIDTH}}, {{cfg.USED_WIDTH, cfg.MARGIN}}));
            }
        } else {

            auto slice_x = Eigen::seq(0, cfg.USED_WIDTH - 1);
            auto slice_y = Eigen::seq(cfg.USED_HEIGHT - cfg.MARGIN, cfg.USED_HEIGHT - 1);
            VectorField2 uv_ex = solution_factory(useCase, mu, t, cfg.LOCAL_X, cfg.LOCAL_Y, slice_x, slice_y);
            uv[0].block(cfg.MARGIN, cfg.MARGIN, cfg.USED_HEIGHT, cfg.USED_WIDTH)(slice_y, slice_x) = std::move(uv_ex[0]);
            uv[1].block(cfg.MARGIN, cfg.MARGIN, cfg.USED_HEIGHT, cfg.USED_WIDTH)(slice_y, slice_x) = std::move(uv_ex[1]);
        }
        return fut;
    }

	[[nodiscard]] Eigen::Map<Matrix, Eigen::Aligned128> const& operator[](int i) const noexcept { assert(0 <= i && i < 2); return uv[i]; }
	[[nodiscard]] Eigen::Map<Matrix, Eigen::Aligned128>      & operator[](int i)       noexcept { assert(0 <= i && i < 2); return uv[i]; }

	[[nodiscard]] Eigen::Index rows() const noexcept { return uv[0].rows(); }
	[[nodiscard]] Eigen::Index cols() const noexcept { return uv[0].cols(); }
};

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
	for(int x = margin; x < nx - margin; ++x)
		for(int y = margin; y < ny - margin; ++y)
			assignment(out(y, x), advection(d[advectRank], w(y, x), rank1Slice<advectRank>(phi, y, x)));
}
void advection(Vectoroid auto const& d, VectorField auto const& uv, VectorField auto& adv) {
	advection<1>(d, uv[0], uv[0], adv[0], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs  = rhs; });
	advection<0>(d, uv[1], uv[0], adv[0], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs += rhs; });

	advection<1>(d, uv[0], uv[1], adv[1], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs  = rhs; });
	advection<0>(d, uv[1], uv[1], adv[1], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs += rhs; });
}

[[nodiscard]] constexpr Scalar diffusion(Scalar d, Vectoroid auto const& phiSlice) {
	return 1 / (12 * sq(d)) * DiffusionStencil::apply(phiSlice);
}
template<std::integral auto diffuseRank>
void diffusion(Vectoroid auto const& d, Matrixoid auto const& phi, Matrixoid auto& out, std::invocable<Scalar&, Scalar> auto assignment) {
	const int nx = phi.cols(), ny = phi.rows();
	assert(out.cols() == nx && out.rows() == ny);
	for(int x = margin; x < nx - margin; ++x)
		for(int y = margin; y < ny - margin; ++y)
			assignment(out(y, x), diffusion(d[diffuseRank], rank1Slice<diffuseRank>(phi, y, x)));
}
void diffusion(Vectoroid auto const& d, VectorField auto const& uv, VectorField auto& diff) {
	diffusion<1>(d, uv[0], diff[0], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs  = rhs; });
	diffusion<0>(d, uv[0], diff[0], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs += rhs; });

	diffusion<1>(d, uv[1], diff[1], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs  = rhs; });
	diffusion<0>(d, uv[1], diff[1], [](Scalar& lhs, Scalar rhs) constexpr noexcept { lhs += rhs; });
}

void rk_stage(
	Scalar mu,
	VectorField auto const& in_now,
	VectorField auto const& in_tmp,
	VectorField auto& out,
	VectorField auto& adv,
	VectorField auto& diff,
	Scalar dt,
	Vectoroid auto const& d
) {
	advection(d, in_tmp, adv);
	diffusion(d, in_tmp, diff);
	out[0] = in_now[0] + dt * (-adv[0] + mu * diff[0]);
	out[1] = in_now[1] + dt * (-adv[1] + mu * diff[1]);
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
	Duration variance;
	unsigned runs;
};
std::ostream& operator<<(std::ostream& stream, BenchmarkReport const& report) {
	stream << "runtime: " << report.average << " ± " << std::sqrt(report.variance.count()) << "ms" << "; best: " << report.best << "; n=" << report.runs << '\n';
	return stream;
}

template<std::invocable Invocable, typename Clock>
[[nodiscard]] BenchmarkReport benchmark(Invocable&& invocable, unsigned streak, Clock clock) {
	assert(streak > 0);
	BenchmarkReport result;
	WelfordVariance<BenchmarkReport::Duration::rep> variance;
	result.best = BenchmarkReport::Duration(std::numeric_limits<double>::max());
	for(unsigned current_streak = 0; current_streak < streak; ) {
        upcxx::barrier();
		const auto start = clock.now();
		invocable();
		const auto end = clock.now();
		auto duration = std::chrono::duration_cast<BenchmarkReport::Duration>(end - start);
        upcxx::barrier();
        duration = upcxx::broadcast(duration, 0).wait();
		if(duration < result.best) {
			result.best = duration;
			current_streak = 0;
		} else {
			++current_streak;
		}
		variance.update(duration.count());
	}
	result.average = BenchmarkReport::Duration(variance.mean);
	result.variance = BenchmarkReport::Duration(variance.sample_variance());
	result.runs = variance.n;
	return result;
}
template<std::invocable Invocable>
[[nodiscard]] BenchmarkReport benchmark(Invocable&& invocable, unsigned streak = 20) {
	if constexpr(std::chrono::high_resolution_clock::is_steady)
		return benchmark(std::forward<Invocable>(invocable), streak, std::chrono::high_resolution_clock{});
	else
		return benchmark(std::forward<Invocable>(invocable), streak, std::chrono::steady_clock{});
}

int main(int argc, char **argv) {

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <grid factor> <ranks width> <ranks height> <num iterations multiplier>" << std::endl;
        return 2;
    }
	// use case
	constexpr UseCase useCase = zhao;

	// diffusion coefficient
	const Scalar mu = 0.1;

	// grid
	const int factor = std::stol(argv[1]);
	const uint32_t coreWidth = std::stol(argv[2]);
	const uint32_t coreHeight = std::stol(argv[3]);
	upcxx::init();
	const uint32_t coreIndex = upcxx::rank_me();
	const DistributedConfig cfg(factor, factor, coreWidth, coreHeight, coreIndex, margin);

	if (coreWidth * coreHeight != upcxx::rank_n()) {
		std::cerr << "Using invalid number of ranks\n";
		return 1;
	}

	// time
	const Scalar cfl = 1;
	const Scalar timestep = cfl / sq(cfg.GLOBAL_WIDTH-1);
	const Scalar n_iter = sq(cfg.GLOBAL_WIDTH-1) * std::stol(argv[4]) / 1000;
	// output
	const int print_period = 0;
	// vector fields

	const auto syncMatrices = [](std::vector<std::array<upcxx::global_ptr<Scalar>, 2>>& mats) {
		for (uint32_t i = 0; i < mats.size(); i++) {
			mats[i] = upcxx::broadcast(mats[i], i).wait();
		}
	};

    const auto single_threaded_task = [&]{

        std::vector<std::array<upcxx::global_ptr<Scalar>, 2>> allMatricesNow(cfg.RANKS_TOTAL);
        DistributedVectorField2 uv_now(VectorField2(cfg.GLOBAL_HEIGHT, cfg.GLOBAL_WIDTH, 0), allMatricesNow, cfg);
        syncMatrices(allMatricesNow);

        std::vector<std::array<upcxx::global_ptr<Scalar>, 2>> allMatricesNew(cfg.RANKS_TOTAL);
        DistributedVectorField2 uv_new(initial_solution(useCase, mu, cfg.GLOBAL_X, cfg.GLOBAL_Y), allMatricesNew, cfg);
        syncMatrices(allMatricesNew);

        upcxx::barrier();
        upcxx::when_all(uv_new.synchronizeRight(useCase, mu, 0), uv_new.synchronizeLeft(useCase, mu, 0), uv_new.synchronizeUp(useCase, mu, 0), uv_new.synchronizeDown(useCase, mu, 0)).wait();
        upcxx::barrier();

        VectorField2 adv(uv_now[0].rows(), uv_now[0].cols(), 0);
        VectorField2 diff(uv_now[0].rows(), uv_now[0].cols(), 0);

        const auto rk_fractions = { 1.0 / 3, 1.0 / 2, 1.0 };
        Scalar t = 0;

        for(int i = 0; i < n_iter; ++i) {
            uv_now = uv_new;

            for(const double rk_fraction : rk_fractions) {
                const Scalar dt = rk_fraction * timestep;
                rk_stage(mu, uv_now, uv_new, uv_new, adv, diff, dt, Vector2{ cfg.DX, cfg.DY });
                upcxx::barrier();
                upcxx::when_all(uv_new.synchronizeRight(useCase, mu, t + dt), uv_new.synchronizeLeft(useCase, mu, t + dt), uv_new.synchronizeUp(useCase, mu, t + dt), uv_new.synchronizeDown(useCase, mu, t + dt)).wait();
                upcxx::barrier();
            }

            t += timestep;
            if(print_period > 0 && ((i+1) % print_period == 0 || i+1 == n_iter)) {
                VectorField2 uv_ex  = solution_factory(useCase, mu, t, cfg.LOCAL_X.segment(0, cfg.USED_WIDTH), cfg.LOCAL_Y.segment(0, cfg.USED_HEIGHT));
                const Scalar local_err_u = (uv_new[0].block(cfg.MARGIN, cfg.MARGIN, cfg.USED_HEIGHT, cfg.USED_WIDTH) - uv_ex[0]).squaredNorm() * cfg.DX * cfg.DY;
                const Scalar local_err_v = (uv_new[1].block(cfg.MARGIN, cfg.MARGIN, cfg.USED_HEIGHT, cfg.USED_WIDTH) - uv_ex[1]).squaredNorm() * cfg.DX * cfg.DY;
                const Scalar err_u = std::sqrt(upcxx::reduce_all(local_err_u, std::plus<Scalar>()).wait());
                const Scalar err_v = std::sqrt(upcxx::reduce_all(local_err_v, std::plus<Scalar>()).wait());
                upcxx::barrier();
                if (cfg.RANK_INDEX == 0) {
                    std::cout << "Iteration " << std::right << std::setfill(' ') << std::setw(6) << i + 1;
                    std::cout << std::scientific << std::setprecision(4);
                    std::cout << ": ||u - uex|| = " << err_u << " m/s, ||v - vex|| = " << err_v << " m/s\n";
                    std::cout << std::defaultfloat;
                }
            }
        }
    };
    if (upcxx::rank_me() == 0) {
        std::cout << "Time: " << benchmark(single_threaded_task);
    } else {
        benchmark(single_threaded_task);
    }

    upcxx::finalize();
}
