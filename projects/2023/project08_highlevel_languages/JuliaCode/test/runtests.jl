using JuliaCode
using Test

# Global constants for update_halo testing
const test_arr_update_halo = reshape(collect(1:16), (4,4,1))
const test_res_update_halo = [11 7 11 7; 10 6 10 6; 11 7 11 7; 10 6 10 6;;;]

# Global constants for laplacian testing
const test_arr_laplacian = [0. 0. 0. 0. 0. 0.; 0. 0. 0. 0. 0. 0.; 0. 0. 1. 1. 0. 0.; 0. 0. 1. 1. 0. 0.; 0. 0. 0. 0. 0. 0.; 0. 0. 0. 0. 0. 0.;;;]
const test_res_laplacian = [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 -2.0 -2.0 0.0 0.0; 0.0 0.0 -2.0 -2.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0;;;]

# Global constants for apply_diffusion testing
const test_arr_apply_diffusion = [0. 0. 0. 0. 0. 0. 0. 0.; 0. 0. 0. 0. 0. 0. 0. 0.; 0. 0. 0. 0. 0. 0. 0. 0.; 0. 0. 0. 1. 1. 0. 0. 0.; 0. 0. 0. 1. 1. 0. 0. 0.; 0. 0. 0. 0. 0. 0. 0. 0.; 0. 0. 0. 0. 0. 0. 0. 0.; 0. 0. 0. 0. 0. 0. 0. 0.;;;]
const test_res_apply_diffusion = [0.94 0.04 0.04 0.94 0.94 0.04 0.04 0.94; 0.04 -0.02 -0.02 0.04 0.04 -0.02 -0.02 0.04; 0.04 -0.02 -0.02 0.04 0.04 -0.02 -0.02 0.04; 0.94 0.04 0.04 0.94 0.94 0.04 0.04 0.94; 0.94 0.04 0.04 0.94 0.94 0.04 0.04 0.94; 0.04 -0.02 -0.02 0.04 0.04 -0.02 -0.02 0.04; 0.04 -0.02 -0.02 0.04 0.04 -0.02 -0.02 0.04; 0.94 0.04 0.04 0.94 0.94 0.04 0.04 0.94;;;]

@testset "stencil2d_naive" begin
    # Test update_halo
    res_update_halo = copy(test_arr_update_halo)
    JuliaCode.Stencil2d_naive.update_halo(res_update_halo, 1)
    @test res_update_halo == test_res_update_halo

    # Test laplacian
    res_laplacian = similar(test_arr_laplacian)
    JuliaCode.Stencil2d_naive.laplacian(test_arr_laplacian, res_laplacian, 2, 0)
    @test res_laplacian ≈ test_res_laplacian

    # Test apply_diffusion
    res_apply_diffusion = similar(test_arr_apply_diffusion)
    JuliaCode.Stencil2d_naive.apply_diffusion(test_arr_apply_diffusion, res_apply_diffusion, 0.01, 1, 2)
    @test res_apply_diffusion ≈ test_res_apply_diffusion
end


@testset "stencil2d_vectorized" begin
    # Test update_halo
    res_update_halo = copy(test_arr_update_halo)
    JuliaCode.Stencil2d_vectorized.update_halo(res_update_halo, 1)
    @test res_update_halo == test_res_update_halo

    # Test laplacian
    res_laplacian = similar(test_arr_laplacian)
    JuliaCode.Stencil2d_vectorized.laplacian(test_arr_laplacian, res_laplacian, 2, 0)
    @test res_laplacian ≈ test_res_laplacian

    # Test apply_diffusion
    res_apply_diffusion = similar(test_arr_apply_diffusion)
    JuliaCode.Stencil2d_vectorized.apply_diffusion(test_arr_apply_diffusion, res_apply_diffusion, 0.01, 1, 2)
    @test res_apply_diffusion ≈ test_res_apply_diffusion
end

@testset "stencil2d_for_vectorized" begin
    # Test update_halo
    res_update_halo = copy(test_arr_update_halo)
    JuliaCode.Stencil2d_for_vectorized.update_halo(res_update_halo, 1)
    @test res_update_halo == test_res_update_halo

    # Test laplacian
    res_laplacian = similar(test_arr_laplacian)
    JuliaCode.Stencil2d_for_vectorized.laplacian(test_arr_laplacian, res_laplacian, 2, 0)
    @test res_laplacian ≈ test_res_laplacian

    # Test apply_diffusion
    res_apply_diffusion = similar(test_arr_apply_diffusion)
    JuliaCode.Stencil2d_for_vectorized.apply_diffusion(test_arr_apply_diffusion, res_apply_diffusion, 0.01, 1, 2)
    @test res_apply_diffusion ≈ test_res_apply_diffusion
end

@testset "stencil2d_multithreaded" begin
    # Test update_halo
    res_update_halo = copy(test_arr_update_halo)
    JuliaCode.Stencil2d_multithreaded.update_halo(res_update_halo, 1)
    @test res_update_halo == test_res_update_halo

    # Test laplacian
    res_laplacian = similar(test_arr_laplacian)
    JuliaCode.Stencil2d_multithreaded.laplacian(test_arr_laplacian, res_laplacian, 2, 0)
    @test res_laplacian ≈ test_res_laplacian

    # Test apply_diffusion
    res_apply_diffusion = similar(test_arr_apply_diffusion)
    JuliaCode.Stencil2d_multithreaded.apply_diffusion(test_arr_apply_diffusion, res_apply_diffusion, 0.01, 1, 2)
    @test res_apply_diffusion ≈ test_res_apply_diffusion
end
