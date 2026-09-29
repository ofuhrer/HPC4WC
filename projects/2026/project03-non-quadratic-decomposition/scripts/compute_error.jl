#!/usr/bin/env julia
using Serialization, LinearAlgebra, Printf

function find_last_jls(outdir)
    if !isdir(outdir)
        error("Outdir not found: $outdir")
    end
    files = filter(x->endswith(x, ".jls"), readdir(outdir, join=true))
    isempty(files) && error("No .jls files in $outdir")
    sort(files)[end]
end

function load_h(path)
    io = open(path, "r")
    obj = deserialize(io)
    close(io)
    if isa(obj, NamedTuple) && hasproperty(obj, :h)
        return Array(obj.h)
    elseif isa(obj, Tuple) && length(obj) >= 1 && hasproperty(obj[1], :h)
        return Array(obj[1].h)
    elseif obj isa AbstractArray
        return Array(obj)
    else
        try
            return Array(obj.h)
        catch
            error("Cannot extract h from $path (got $(typeof(obj)))")
        end
    end
end

if length(ARGS) < 2
    println("Usage: compute_error.jl <manual_outdir> <baseline_outdir>")
    exit(1)
end

manual_out = ARGS[1]
baseline_out = ARGS[2]

mf = find_last_jls(manual_out)
bf = find_last_jls(baseline_out)
println("manual frame: ", mf)
println("baseline frame: ", bf)

h_ref = load_h(mf)
h_tgt = load_h(bf)
if size(h_ref) != size(h_tgt)
    error("size mismatch: $(size(h_ref)) vs $(size(h_tgt))")
end
rel_err = norm(vec(h_tgt .- h_ref)) / (norm(vec(h_ref)) + eps())
@printf("REL_ERR,%.6e\n", rel_err)
exit(0)
