using Serialization
using Printf
using CairoMakie

input_dir = "docs/frames/manual"
output_dir = joinpath(input_dir, "plots")

for i in 1:length(ARGS)
    if ARGS[i] == "--input"
        global input_dir = ARGS[i + 1]
    elseif ARGS[i] == "--output"
        global output_dir = ARGS[i + 1]
    end
end

files = sort([
    joinpath(input_dir, f)
    for f in readdir(input_dir)
    if startswith(f, "array_frame_") && endswith(f, ".jls")
])

isempty(files) && error("No array_frame_*.jls files found in $input_dir")

mkpath(output_dir)

last_z = nothing

for (iframe, file) in enumerate(files)
    frame = deserialize(file)
    h = Array(frame.h)

    if hasproperty(frame, :z)
        global last_z = Array(frame.z)
    end

    z = last_z === nothing ? zeros(size(h)) : last_z
    eta = h .+ z

    fig = Figure(size = (600, 200))
    ax = Axis(fig[1, 1];
        title = "free surface, frame $iframe",
        xlabel = "x index",
        ylabel = "y index",
        aspect = DataAspect(),
    )

    hm = heatmap!(ax, eta; colormap = :viridis)
    Colorbar(fig[1, 2], hm,
        label = "eta = h + z")
    
    out = joinpath(output_dir, @sprintf("frame_%06d.png", iframe))
    save(out, fig)
    println("saved ", out)
end
