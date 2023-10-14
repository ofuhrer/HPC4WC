module JuliaCode

# How to run this module:
# 1. Be in the project directory (hpcwc-performance-comparison)
# 2. Start Julia with "julia --project=JuliaCode"
# 3. Import package via "using JuliaCode"
# 4. Call "JuliaCode.run()"

# Auto-adapts changes in the code to the loaded package
using BenchmarkTools, CSV, CUDA, DataFrames, Gadfly, LoopVectorization, Revise 

# Include relevant versions of the program
include("plotting_facilities.jl")
include("input_loader.jl")
include("stencil2d_naive.jl")
include("stencil2d_vectorized.jl")
include("stencil2d_for_vectorized.jl")
include("stencil2d_multithreaded.jl")

using .PlottingFacilities
using .InputLoader
using .Stencil2d_naive
using .Stencil2d_vectorized
using .Stencil2d_for_vectorized
using .Stencil2d_multithreaded

# Functions that are used elsewhere need to be exported!
export run

function bench_func(func, dims, gpu=false)
    # Warm-up / precompilation
    a = ones(8,8,8)
    b = copy(a)
    if gpu
        a = cu(a)
        b = cu(b)
    end
    func(a, b, 1.0, 1, 2)

    # Concrete benchmark
    n = nrow(dims)
    timings = zeros(n)
    for idx in 1:n
        # Measure time and store it in an array
        if gpu
            timings[idx] = @belapsed (CUDA.@sync $func(in, out, $dims[$idx, 4], $dims[$idx, 5], 2)) setup=((in, out) = (cu(generate_initial_array($dims[$idx, 1:3])), cu(generate_initial_array($dims[$idx, 1:3]))))
        else
            timings[idx] = @belapsed $func(in, out, $dims[$idx, 4], $dims[$idx, 5], 2) setup=((in, out) = (generate_initial_array($dims[$idx, 1:3]), generate_initial_array($dims[$idx, 1:3])))
        end
    end
    return timings
end

# Main function
function run()
    # Determine if nvidia gpu is available (including CUDA)
    gpu = CUDA.functional()

    # Pre-Benchmark tasks
    input_info = load_input()

    # Define functions to benchmark
    funcs_to_bench = [Stencil2d_naive.apply_diffusion, Stencil2d_vectorized.apply_diffusion, Stencil2d_for_vectorized.apply_diffusion, Stencil2d_multithreaded.apply_diffusion]

    # Allocate timing results array
    timing_results = zeros((nrow(input_info), length(funcs_to_bench) + gpu))

    # Benchmarking
    for bench_idx in 1:length(funcs_to_bench)
        println("Benchmark #$(bench_idx)")
        timing_results[:, bench_idx] .= bench_func(funcs_to_bench[bench_idx], input_info)
    end

    # GPU Benchmarking (only one, last benchmark)
    if gpu
        println("GPU Benchmark")
        timing_results[:, end] .= bench_func(Stencil2d_vectorized.apply_diffusion, input_info, gpu)
    end

    # Create a nice DataFrame output in milliseconds
    timings_df = DataFrame(timing_results .*1000.0, :auto)

    if gpu
        rename!(timings_df, [:naive, :vectorized, :for_vectorized, :multithreaded, :cuda])
    else
        rename!(timings_df, [:naive, :vectorized, :for_vectorized, :multithreaded])
    end

    # Save CSV
    CSV.write("universal_output/jl_benchmarks.csv", timings_df)

    #return plot(log10.(timings_df), x=Row.index, y = Col.value, color = Col.index)
end

end # module JuliaCode

#JuliaCode.run()