#!/usr/bin/env julia
using Serialization, LinearAlgebra, Printf

function load_h(path::AbstractString)
    open(path, "r") do io
        obj = deserialize(io)
        if isa(obj, AbstractArray)
            return Array(obj)
        elseif isa(obj, Tuple) && length(obj) >= 1 && hasproperty(obj[1], :h)
            return Array(obj[1].h)
        elseif isa(obj, NamedTuple) && hasproperty(obj, :h)
            return Array(obj.h)
        elseif isa(obj, Dict) && haskey(obj, :h)
            return Array(obj[:h])
        else
            try
                return Array(obj.h)
            catch
                error("Could not extract `h` from serialized object in $path")
            end
        end
    end
end

if length(ARGS) < 2
    println("Usage: test_topologies.jl <manual_h_path> <baseline_h_path> [topology]")
    exit(1)
end

manual_path = ARGS[1]
baseline_path = ARGS[2]
topology = length(ARGS) >= 3 ? ARGS[3] : ""

h_ref = load_h(manual_path)
h_tgt = load_h(baseline_path)

if size(h_ref) != size(h_tgt)
    error("Shape mismatch between manual and baseline results: $(size(h_ref)) vs $(size(h_tgt))")
end

rel_err = norm(vec(h_tgt .- h_ref)) / (norm(vec(h_ref)) + eps())
if topology != ""
    @printf("Topology %6s  rel_err=%.6e\n", topology, rel_err)
else
    @printf("rel_err=%.6e\n", rel_err)
end

println("Manual path: ", manual_path)
println("Baseline path: ", baseline_path)
println("Done.")
