#!/usr/bin/env bash
# Strong- and weak-scaling sweep for the default stencil2d-mpi.F90 baseline
# (up/down then left/right non-blocking halo exchange -- corners folded into
# the left/right phase, no separate corner messages).
set -euo pipefail

source "$(dirname "$0")/lib_experiment.sh"
ORIGINAL_ARGS=("$@")

mode="strong"
ranks_csv=""
nz=64
num_iter=1024
repeats=3
results_dir=""
local_n=128
nx=1024
ny=1024
build=0
time_limit=""

while [ $# -gt 0 ]; do
    case "$1" in
        --mode)        mode="$2"; shift 2 ;;
        --ranks)       ranks_csv="$2"; shift 2 ;;
        --nx)          nx="$2"; shift 2 ;;
        --ny)          ny="$2"; shift 2 ;;
        --nz)          nz="$2"; shift 2 ;;
        --num_iter)    num_iter="$2"; shift 2 ;;
        --local-n)     local_n="$2"; shift 2 ;;
        --repeats)     repeats="$2"; shift 2 ;;
        --results-dir) results_dir="$2"; shift 2 ;;
        --time-limit)  time_limit="$2"; shift 2 ;;
        --build)       build=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$mode" in
    strong)
        : "${ranks_csv:=1,2,4,8,16,32}"
        : "${results_dir:=results/corners/results_mpi_leftright_strong_1024}"
        : "${time_limit:=15}"
        ;;
    weak)
        : "${ranks_csv:=1,2,4,8,16,32,64}"
        : "${results_dir:=results/corners/results_mpi_leftright_weak_128}"
        : "${time_limit:=10}"
        ;;
    *) echo "--mode must be strong or weak" >&2; exit 2 ;;
esac

binary="./stencil2d-mpi.x"
if [ "$build" -eq 1 ]; then
    make VERSION=mpi
fi
[ -x "$binary" ] || { echo "Missing executable $binary" >&2; exit 2; }

rank_grid_x() {
    local n=$1 i
    for (( i = $(awk -v n="$n" 'BEGIN{print int(sqrt(n))}'); i >= 1; i-- )); do
        if [ $((n % i)) -eq 0 ]; then echo "$i"; return; fi
    done
    echo 1
}

IFS=',' read -r -a RANK_COUNTS <<< "$ranks_csv"
total=$(( ${#RANK_COUNTS[@]} * repeats ))
mkdir -p "$results_dir"
experiment_init "$results_dir" "${ORIGINAL_ARGS[@]}"

status=0
run_count=0
for ranks in "${RANK_COUNTS[@]}"; do
    nodes=$(((ranks + 3) / 4))
    tasks_per_node=$ranks
    [ "$tasks_per_node" -gt 4 ] && tasks_per_node=4

    if [ "$mode" = "weak" ]; then
        px=$(rank_grid_x "$ranks")
        py=$((ranks / px))
        run_nx=$((local_n * px))
        run_ny=$((local_n * py))
        geom_args=(--local-nx "$local_n" --local-ny "$local_n")
        echo "--- ${ranks} ranks: ${px}x${py} rank grid, global ${run_nx}x${run_ny} (${local_n}x${local_n} per rank) ---"
    else
        run_nx=$nx
        run_ny=$ny
        geom_args=()
    fi

    for rep in $(seq 1 "$repeats"); do
        run_count=$((run_count + 1))
        echo "=== [${run_count}/${total}] mpi_leftright ${mode} N=${nodes} n=${ranks} rep=${rep} ==="
        run_case --label mpi_leftright --binary "$binary" \
            --results-dir "$results_dir" --mode "$mode" --rep "$rep" \
            --nodes "$nodes" --ranks "$ranks" --cores 1 \
            --nx "$run_nx" --ny "$run_ny" --nz "$nz" --num_iter "$num_iter" \
            "${geom_args[@]}" \
            --env "OMP_NUM_THREADS=1" \
            --srun-extra "--ntasks-per-node=${tasks_per_node} --gpus-per-task=1 --cpu-bind=cores -t ${time_limit}" \
            || status=1
    done
done

echo "Done. ${run_count} runs written to ${results_dir}/"
exit "$status"
