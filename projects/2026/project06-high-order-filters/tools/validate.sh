#!/usr/bin/env bash
#
# Runs the 14-check validation suite. Must pass before any benchmark.
# Each check echoes its own name; README.md states what they establish.
#
# Usage: validate.sh [launcher]   launcher defaults to "mpirun -n"
#
# Authors: Stefanie Boersig <stefanie.boersig@env.ethz.ch>
#          Boaz Ko <boazko@student.ethz.ch>
#          Ben Bullinger <ben.bullinger@inf.ethz.ch>
set -euo pipefail

cd "$(dirname "$0")/../src"

LAUNCH="${1:-mpirun -n}"
NX=32; NY=32; NZ=8; ITER=32
RUNARGS="--nx $NX --ny $NY --nz $NZ --num_iter $ITER"

make -s VERSION=base >/dev/null
make -s VERSION=limiter >/dev/null
make -s test_sg.x >/dev/null

echo "== partitioner scatter/gather roundtrip"
rt_out=$($LAUNCH 4 ./test_sg.x)
echo "$rt_out"
grep -q " 0 interior mismatches" <<< "$rt_out"

run() { # run <nranks> <binary> <extra args> <output tag>
    $LAUNCH "$1" "./$2" $RUNARGS $3 >/dev/null
    mv out_field.dat "out_$4.dat"
}

echo "== reference: 1 rank, halo 2 (base)"
run 1 stencil2d-base.x "" ref

echo "== MPI correctness: 4 ranks, halo 2 (base)"
run 4 stencil2d-base.x "" mpi4
python3 ../tools/read_field.py out_ref.dat out_mpi4.dat

for H in 3 4; do
    echo "== halo parameterization: 4 ranks, halo $H (base)"
    run 4 stencil2d-base.x "--num_halo $H" "halo$H"
    python3 ../tools/read_field.py out_ref.dat "out_halo$H.dat"
done

echo "== flux form, limiter OFF vs base (SP roundoff tolerance)"
run 4 stencil2d-limiter.x "" fluxoff
python3 ../tools/read_field.py out_ref.dat out_fluxoff.dat --rtol 1e-5 --atol 1e-6

echo "== unlimited 4th order must over/undershoot (test sanity)"
python3 ../tools/check_monotonic.py in_field.dat out_fluxoff.dat --expect-violation

echo "== limiter ON: monotonicity"
run 4 stencil2d-limiter.x "--limiter" limited
python3 ../tools/check_monotonic.py in_field.dat out_limited.dat

# ---- order-generic filter (WP3) ----
make -s VERSION=filter >/dev/null

echo "== filter --order 4 (alpha matched) vs dedicated limiter version: bit-exact"
run 4 stencil2d-filter.x "--order 4 --alpha 0.03125 --limiter" f4lim
python3 ../tools/read_field.py out_limited.dat out_f4lim.dat

echo "== filter --order 4 unlimited vs base (SP roundoff)"
run 4 stencil2d-filter.x "--order 4 --alpha 0.03125" f4off
python3 ../tools/read_field.py out_ref.dat out_f4off.dat --rtol 1e-5 --atol 1e-6

echo "== monotonicity with limiter for all orders (auto halo = n/2)"
for N in 2 4 6 8; do
    run 4 stencil2d-filter.x "--order $N --limiter" "f${N}lim"
    printf 'order %s: ' "$N"
    python3 ../tools/check_monotonic.py in_field.dat "out_f${N}lim.dat" | tail -1
done

echo "== order 2 is inherently monotone: limiter must be (almost) inactive"
$LAUNCH 4 ./stencil2d-filter.x $RUNARGS --order 2 --limiter 2>/dev/null \
    | grep limited_fraction
mv out_field.dat out_scratch.dat

echo "== damping factors vs analytic lambda(k)^N (single-mode IC, all orders)"
MODEARGS="--nx 64 --ny 64 --nz 4 --num_iter 8 --ic mode --kx 4 --ky 3"
for N in 2 4 6 8; do
    ALPHA=$(python3 -c "print(1.0/8**($N//2))")
    $LAUNCH 4 ./stencil2d-filter.x $MODEARGS --order $N >/dev/null
    python3 ../tools/check_amplification.py in_field.dat out_field.dat $N $ALPHA 4 3 8
done

# ---- optimized variants (WP4) ----
# Every layout must reproduce the naive filter exactly, in every limiter
# mode. The branch-free form is a different expression of the same test,
# so it too must be bit-exact, not merely close.
make -s VERSION=filter-kblock >/dev/null
make -s VERSION=filter-inline >/dev/null

echo "== optimized variants vs naive filter, bit-exact, all limiter modes"
for N in 4 8; do
    run 4 stencil2d-filter.x "--order $N" "nv${N}off"
    run 4 stencil2d-filter.x "--order $N --limiter" "nv${N}lim"
    for V in kblock inline; do
        printf 'order %s, %s, limiter off:        ' "$N" "$V"
        run 4 "stencil2d-filter-$V.x" "--order $N" "op${N}off"
        python3 ../tools/read_field.py "out_nv${N}off.dat" "out_op${N}off.dat" | tail -1

        printf 'order %s, %s, limiter on:         ' "$N" "$V"
        run 4 "stencil2d-filter-$V.x" "--order $N --limiter" "op${N}lim"
        python3 ../tools/read_field.py "out_nv${N}lim.dat" "out_op${N}lim.dat" | tail -1

        printf 'order %s, %s, limiter branchfree: ' "$N" "$V"
        run 4 "stencil2d-filter-$V.x" "--order $N --limiter --branchfree" "op${N}bf"
        python3 ../tools/read_field.py "out_nv${N}lim.dat" "out_op${N}bf.dat" | tail -1
    done
done

# ---- Zalesak flux correction (forced limits) ----
make -s VERSION=filter-zalesak >/dev/null

echo "== Zalesak, C forced to 1, vs unlimited filter (SP roundoff)"
# C == 1 restores the full corrective flux, so the scheme reduces to the
# unlimited order-n filter. The tolerance is for the reordered summation
# (phi* then correction), not for a difference in the result.
run 4 stencil2d-filter-zalesak.x "--order 4 --cone" zcone
python3 ../tools/read_field.py out_nv4off.dat out_zcone.dat --rtol 1e-5 --atol 1e-6

echo "== Zalesak, C forced to 0, vs order-2 filter: bit-exact"
# C == 0 discards the correction entirely, leaving the second-order flux.
run 4 stencil2d-filter.x "--order 2 --alpha 0.125" f2ref
run 4 stencil2d-filter-zalesak.x "--order 4 --czero --alpha 0.125" zczero
python3 ../tools/read_field.py out_f2ref.dat out_zczero.dat

echo "== Zalesak monotonicity, orders 4 and 6"
for N in 4 6; do
    run 4 stencil2d-filter-zalesak.x "--order $N" "z${N}"
    printf 'order %s: ' "$N"
    python3 ../tools/check_monotonic.py in_field.dat "out_z${N}.dat" | tail -1
done

rm -f out_*.dat in_field.dat
echo "== all validations passed"
