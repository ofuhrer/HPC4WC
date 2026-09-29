using CairoMakie

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

input_path = joinpath(PROJECT_ROOT, "output", "walltimes_16_ranks.csv")
output_path = nothing

i = 1
while i <= length(ARGS)
    if ARGS[i] == "--input"
        i < length(ARGS) || error("--input requires a path")
        global input_path = ARGS[i + 1]
        global i += 2
    elseif ARGS[i] == "--output"
        i < length(ARGS) || error("--output requires a path")
        global output_path = ARGS[i + 1]
        global i += 2
    else
        error("Unknown argument: $(ARGS[i])")
    end
end

function find_input_files(path)
    if isfile(path)
        endswith(lowercase(path), ".csv") || error("Input file must be a CSV: $path")
        return [path]
    elseif isdir(path)
        files = filter(readdir(path; join=true)) do file
            name = lowercase(basename(file))
            isfile(file) && endswith(name, ".csv") && startswith(name, "walltimes")
        end
        sort!(files)
        isempty(files) && error("No walltimes*.csv files found in input directory: $path")
        return files
    end

    error("Input path not found: $path")
end

function parse_topology(value, input_file)
    parts = split(lowercase(strip(value)), 'x')
    length(parts) == 2 || error("Invalid topology '$value' in $input_file")
    dims = Tuple(parse.(Int, parts))
    all(>(0), dims) || error("Topology dimensions must be positive in $input_file")
    return dims
end

function read_decompositions(input_file)
    lines = filter(!isempty, strip.(readlines(input_file)))
    isempty(lines) && error("CSV is empty: $input_file")

    header = strip.(split(first(lines), ','))
    columns = Dict(name => i for (i, name) in enumerate(header))
    required = ["nprocs", "topology", "nx_global", "ny_global", "nx_local", "ny_local"]
    missing_columns = filter(name -> !haskey(columns, name), required)
    isempty(missing_columns) ||
        error("Missing columns in $input_file: $(join(missing_columns, ", "))")

    decompositions = NamedTuple[]
    for (row_index, line) in enumerate(lines[2:end])
        line_number = row_index + 1
        fields = strip.(split(line, ','))
        length(fields) == length(header) ||
            error("Malformed CSV row $line_number in $input_file")

        dims = parse_topology(fields[columns["topology"]], input_file)
        nprocs = parse(Int, fields[columns["nprocs"]])
        global_with_halo = (
            parse(Int, fields[columns["nx_global"]]),
            parse(Int, fields[columns["ny_global"]]),
        )
        local_with_halo = (
            parse(Int, fields[columns["nx_local"]]),
            parse(Int, fields[columns["ny_local"]]),
        )
        minimum(global_with_halo) > 2 || error("Global grid must include an interior in $input_file")
        minimum(local_with_halo) > 2 || error("Local grid must include an interior in $input_file")

        global_size = global_with_halo .- 2
        local_size = local_with_halo .- 2
        prod(dims) == nprocs ||
            error("Topology $(dims[1])x$(dims[2]) has $(prod(dims)) cells but nprocs=$nprocs")
        global_size == local_size .* dims || error(
            "Inconsistent grid sizes for topology $(dims[1])x$(dims[2]): " *
            "global interior $global_size != local interior $local_size .* topology $dims"
        )

        push!(decompositions, (;
            topology = "$(dims[1])x$(dims[2])",
            dims,
            nprocs,
            global_size,
            local_size,
            input_file,
        ))
    end
    return decompositions
end

function unique_decompositions(input_files)
    by_topology = Dict{String, NamedTuple}()
    for input_file in input_files, item in read_decompositions(input_file)
        if haskey(by_topology, item.topology)
            previous = by_topology[item.topology]
            same_grid = item.nprocs == previous.nprocs &&
                        item.global_size == previous.global_size &&
                        item.local_size == previous.local_size
            same_grid || error(
                "Topology $(item.topology) has conflicting grid sizes in " *
                "$(previous.input_file) and $(item.input_file)"
            )
        else
            by_topology[item.topology] = item
        end
    end
    isempty(by_topology) && error("No benchmark rows found in: $(join(input_files, ", "))")
    return sort!(collect(values(by_topology)); by=item -> item.dims)
end

function plot_decomposition!(fig, row, item; show_x_decorations=true)
    px, py = item.dims
    nx, ny = item.global_size
    local_nx, local_ny = item.local_size

    ax = Axis(fig[row, 1];
        title = item.topology,
        xlabel = "global interior x index",
        ylabel = "global interior y index",
        aspect = DataAspect(),
        width = 900,
        height = 260,
        titlegap = 2,
        xlabelpadding = 2,
        ylabelpadding = 2,
        xticklabelpad = 2,
        yticklabelpad = 2,
        xticklabelsvisible = show_x_decorations,
        xticksvisible = show_x_decorations,
        xlabelvisible = show_x_decorations,
    )

    # MPI.Cart_create uses row-major Cartesian rank ordering: the last
    # coordinate varies fastest. This matrix contains one value per MPI rank,
    # regardless of how many cells are in the actual simulation grid.
    rank_grid = [Float32(ix * py + iy) for ix in 0:(px - 1), iy in 0:(py - 1)]
    x_edges = range(0.5, nx + 0.5; length=px + 1)
    y_edges = range(0.5, ny + 0.5; length=py + 1)
    heatmap!(ax, x_edges, y_edges, rank_grid;
        colormap=cgrad(:tab10, item.nprocs, categorical=true),
    )

    for ix in 1:(px - 1)
        x = 0.5 + ix * local_nx
        lines!(ax, [x, x], [0.5, ny + 0.5]; color=:black, linewidth=2)
    end
    for iy in 1:(py - 1)
        y = 0.5 + iy * local_ny
        lines!(ax, [0.5, nx + 0.5], [y, y]; color=:black, linewidth=2)
    end

    label_fontsize = clamp(64 / maximum(item.dims), 7, 20)
    for ix in 0:(px - 1), iy in 0:(py - 1)
        rank = ix * py + iy
        x = 0.5 + (ix + 0.5) * local_nx
        y = 0.5 + (iy + 0.5) * local_ny
        text!(ax, x, y;
            text=string(rank),
            align=(:center, :center),
            color=:white,
            fontsize=label_fontsize,
            font=:bold,
        )
    end

    xlims!(ax, 0.5, nx + 0.5)
    ylims!(ax, 0.5, ny + 0.5)
end

input_files = find_input_files(input_path)
decompositions = unique_decompositions(input_files)

fig = Figure(figure_padding=0)
Label(fig[0, 1], "MPI domain decompositions";
    fontsize=24,
    font=:bold,
    padding=(0, 0, 0, 0),
)

for (row, item) in enumerate(decompositions)
    plot_decomposition!(fig, row, item;
        show_x_decorations=row == length(decompositions),
    )
end
rowgap!(fig.layout, 0)
resize_to_layout!(fig)

output_base = if output_path === nothing
    output_dir = isdir(input_path) ? input_path : dirname(input_path)
    joinpath(output_dir, "domain_decompositions")
elseif lowercase(splitext(output_path)[2]) in [".pdf", ".png", ".svg"]
    first(splitext(output_path))
else
    joinpath(output_path, "domain_decompositions")
end

mkpath(dirname(output_base))
output_files = [output_base * ".png", output_base * ".pdf"]
for output_file in output_files
    save(output_file, fig)
    println("saved ", output_file)
end
