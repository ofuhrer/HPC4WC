module InputLoader

using CSV, DataFrames

export load_input, generate_initial_array

function load_input()
    return CSV.read("./universal_input/input_dimensions.csv", DataFrame, header=true)
end

function generate_initial_array(dim_df, num_halo = 2)
    nx = floor(Int,dim_df[1] + 2 * num_halo)
    ny = floor(Int,dim_df[2] + 2 * num_halo)
    nz = floor(Int,dim_df[3])

    arr = zeros(Float64, nx, ny, nz)
    arr[(1 + nx ÷ 4):(3 * nx ÷ 4), (1 + ny ÷ 4):(3 * ny ÷ 4), (1 + nz ÷ 4):(3 * nz ÷ 4)] .= 1.0
    return arr
end

end #module