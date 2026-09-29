#!/bin/bash

# ==============================================================================
# PARAMETER-EINSTELLUNGEN
# Hier kannst du deine Simulationen für alle Durchläufe zentral anpassen:
# ==============================================================================
NX=402
NY=802
NT=5
PROCS=8

# ==============================================================================
# BENCHMARK-SCHLEIFE
# Wir übergeben die Bash-Variablen als Argumente an Julias Inline-Skript
# ==============================================================================
julia --project -e '
using MPI

# Bash-Variablen in Julia einlesen
nx = ARGS[1]
ny = ARGS[2]
nt = ARGS[3]
procs = parse(Int, ARGS[4])

topologies = ["1x8", "2x4", "4x2", "8x1"]

for topo in topologies
    println("\n==================================================")
    println(" Berechne Topologie: ", topo, " (mit ", procs, " Prozessen)")
    println("==================================================")
    
    # Exakt dein funktionierender Befehl mit dynamischen Variablen
    run(`$(MPI.mpiexec()) -n $procs julia --project src/manual_halo_timing.jl --topology $topo --nx $nx --ny $ny --nt $nt --benchmark`)
    
    println("Topologie ", topo, " abgeschlossen.\n")
end
println("Alle Benchmarks erfolgreich beendet!")
' "$NX" "$NY" "$NT" "$PROCS"