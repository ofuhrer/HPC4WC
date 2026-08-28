#!/bin/bash

# ==============================================================================
# PARAMETER SETTINGS
# Adjust your parameters centrally for all visualization runs here:
# ==============================================================================
NX=1026
NY=258
NT=200
PROCS=4
TOPO="4,1"              # single topology for manual.jl (comma-separated)
OUTROOT="docs/frames"   # base output directory for all frames/plots

# ==============================================================================
# VIZ CHECK: MANUAL
# manual.jl accepts --topo and writes array_frame_*.jls into --outdir
# ==============================================================================
julia --project -e '
using MPI

nx      = ARGS[1]
ny      = ARGS[2]
nt      = ARGS[3]
procs   = parse(Int, ARGS[4])
outroot = ARGS[5]
topo    = ARGS[6]

outdir = joinpath(outroot, "manual", topo)
println("\n==============================================")
println(" MANUAL viz, topology: ", topo, " (", procs, " procs)")
println("==============================================")

run(`$(MPI.mpiexec()) -n $procs julia --project src/manual.jl --topo $topo --nx $nx --ny $ny --nt $nt --viz --outdir $outdir`)

run(`julia --project scripts/visualize_arrays.jl --input $outdir --output $outdir/plots`)

println("Manual topology ", topo, " done.\n")
' "$NX" "$NY" "$NT" "$PROCS" "$OUTROOT" "$TOPO"

# ==============================================================================
# VIZ CHECK: BASELINE
# baseline.jl has NO --topo flag: MPI.Dims_create! picks a compact topology
# automatically. Use a small nt so the run finishes quickly.
# ==============================================================================
julia --project -e '
using MPI

nx      = ARGS[1]
ny      = ARGS[2]
nt      = ARGS[3]
procs   = parse(Int, ARGS[4])
outroot = ARGS[5]

outdir = joinpath(outroot, "baseline")
println("\n==============================================")
println(" BASELINE viz (", procs, " procs, topology auto)")
println("==============================================")

run(addenv(`$(MPI.mpiexec()) -n $procs julia --project src/baseline.jl --nx $nx --ny $ny --nt $nt --viz --outdir $outdir`, "USE_GPU"=>"false"))

run(`julia --project scripts/visualize_arrays.jl --input $outdir --output $outdir/plots`)

println("Baseline viz done.\n")
' "$NX" "$NY" "$NT" "$PROCS" "$OUTROOT"

echo "=============================================="
echo " All --viz checks finished."
echo " Plots written under: $OUTROOT/<manual|baseline>/*/plots/"
echo "=============================================="