#!/bin/bash

# ==============================================================================
# PARAMETER SETTINGS
# Adjust centrally for all validation runs here:
# ==============================================================================
NX=1026
NY=258
PROCS=4
TOPO="4,1"              # topology for manual.jl (comma-separated)
OUTROOT="docs/validation"

MANUAL_DIR="$OUTROOT/manual"
BASELINE_DIR="$OUTROOT/baseline"

# ==============================================================================
# TEST RUN: MANUAL
# manual.jl --test saves the initial h as test_h.jls (no timesteps, fast)
# ==============================================================================
julia --project -e '
using MPI

nx      = ARGS[1]
ny      = ARGS[2]
procs   = parse(Int, ARGS[3])
topo    = ARGS[4]
outdir  = ARGS[5]

mkpath(outdir)
println("\n==============================================")
println(" MANUAL --test, topology: ", topo, " (", procs, " procs)")
println("==============================================")

run(`$(MPI.mpiexec()) -n $procs julia --project src/manual.jl --topo $topo --nx $nx --ny $ny --test --outdir $outdir --test-file $outdir/test_h.jls`)

println("Manual test done.\n")
' "$NX" "$NY" "$PROCS" "$TOPO" "$MANUAL_DIR"

# ==============================================================================
# TEST RUN: BASELINE
# baseline.jl --test writes the same initial h; force CPU via USE_GPU=false
# ==============================================================================
julia --project -e '
using MPI

nx      = ARGS[1]
ny      = ARGS[2]
procs   = parse(Int, ARGS[3])
outdir  = ARGS[4]

mkpath(outdir)
println("\n==============================================")
println(" BASELINE --test (", procs, " procs, topology auto, CPU)")
println("==============================================")

run(addenv(`$(MPI.mpiexec()) -n $procs julia --project src/baseline.jl --nx $nx --ny $ny --test --outdir $outdir --test-file $outdir/test_h.jls`, "USE_GPU"=>"false"))

println("Baseline test done.\n")
' "$NX" "$NY" "$PROCS" "$BASELINE_DIR"

# ==============================================================================
# COMPARISON
# compute_error.jl compares the last .jls in each dir and prints REL_ERR
# ==============================================================================
echo "=============================================="
echo " Comparing manual vs baseline initial state"
echo "=============================================="
julia --project scripts/compute_error.jl "$MANUAL_DIR" "$BASELINE_DIR"

echo "=============================================="
echo " Validation finished."
echo "=============================================="