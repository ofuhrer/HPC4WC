#!/bin/bash

# ==============================================================================
# PARAMETER SETTINGS
# Adjust your simulation parameters centrally for all runs here:
# ==============================================================================
NX=402
NY=802
NT=20
PROCS=8
BENCHDIR="docs/benchmark/benchmark_test.csv"

# ==============================================================================
# BENCHMARK LOOP
# We pass the Bash variables as arguments to Julia's inline script
# ==============================================================================
julia --project -e '
using MPI

# Read Bash variables into Julia
nx      = ARGS[1]
ny      = ARGS[2]
nt      = ARGS[3]
procs   = parse(Int, ARGS[4])
benchdir = ARGS[5]

# Topologies in the comma-separated format expected by manual.jl
topologies = ["1,8", "2,4", "4,2", "8,1"]

for topo in topologies
    println("\n==================================================")
    println(" Computing topology: ", topo, " (with ", procs, " processes)")
    println("==================================================")

    run(`$(MPI.mpiexec()) -n $procs julia --project src/manual.jl --topo $topo --nx $nx --ny $ny --nt $nt --benchmark --benchdir $benchdir`)

    println("Topology ", topo, " finished.\n")
end
println("All benchmarks completed successfully!")
' "$NX" "$NY" "$NT" "$PROCS" "$BENCHDIR"