#!/usr/bin/env bash
# Strong- and weak-scaling sweeps across every validated communication strategy.
#
# The six halo strategies share one executable and are selected at runtime
# with --comm_strategy. The three corner variants and comm_comp each use a
# separate executable. comm_comp reports total runtime only (6 columns);
# the other strategies also report communication/computation timers, with
# additional detailed timers in the corner variants. The analyzer supports
# these mixed schemas.
#
# Placement is one rank per GH200 (four ranks per node), matching the
# placement used for validation.
set -euo pipefail

source "$(dirname "$0")/lib_experiment.sh"
ORIGINAL_ARGS=("$@")

mode="strong"
ranks_csv=""
nz=64
num_iter=1024
repeats=3
results_dir=""
local_n=128          # weak mode: grid points per rank, per dimension
nx=1024              # strong mode: fixed global grid
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
        : "${results_dir:=results_strong_1024_all_strategies}"
        : "${time_limit:=15}"
        ;;
    weak)
        : "${ranks_csv:=1,2,4,8,16,32,64}"
        : "${results_dir:=results_weak_128_all_strategies}"
        : "${time_limit:=10}"
        ;;
    *) echo "--mode must be strong or weak" >&2; exit 2 ;;
esac

# label:binary:app-extra:min_ranks
#
# brokencycles and sendrecv_evenodd use separate blocking MPI_Send/MPI_Recv
# calls (not the combined MPI_Sendrecv, not non-blocking Isend/Irecv). Under
# periodic boundaries, whenever a rank-grid dimension is 1, a rank is its own
# neighbor along that axis -- the blocking send then waits on a matching
# receive that the same process cannot post until the send returns, so the
# process deadlocks. That happens at 1 rank always (both dims collapse) and
# at 2 ranks (grid is 1x2). From 4 ranks up no dimension is 1, so both are
# safe; their min_ranks is set to 4 and lower rank counts are skipped rather
# than run (skipping is data we don't have, not data recorded as zero).
STRATEGIES_ALL=(
    "comm_comp:./stencil2d-comm_comp.x::1"
    "corners:./stencil2d-mpi-corners.x::1"
    "pipelined:./stencil2d-mpi-corners-pipelined.x::1"
    "waitany:./stencil2d-mpi-corners-waitany.x::1"
    "brokencycles:./stencil2d-comm_strats_timer.x:--comm_strategy brokencycles:4"
    "sendrecv_evenodd:./stencil2d-comm_strats_timer.x:--comm_strategy sendrecv_evenodd:4"
    "sendrecv:./stencil2d-comm_strats_timer.x:--comm_strategy sendrecv:1"
    "irecvsend:./stencil2d-comm_strats_timer.x:--comm_strategy irecvsend:1"
    "isendrecv:./stencil2d-comm_strats_timer.x:--comm_strategy isendrecv:1"
    "irecvisend:./stencil2d-comm_strats_timer.x:--comm_strategy irecvisend:1"
)

# STRATEGIES_FILTER (comma-separated labels) restricts to a subset of the labels above (default: all).
if [ -n "${STRATEGIES_FILTER:-}" ]; then
    IFS=',' read -r -a _want <<< "$STRATEGIES_FILTER"
    STRATEGIES=()
    for item in "${STRATEGIES_ALL[@]}"; do
        label=$(echo "$item" | cut -d: -f1)
        for w in "${_want[@]}"; do
            [ "$label" = "$w" ] && STRATEGIES+=("$item")
        done
    done
else
    STRATEGIES=("${STRATEGIES_ALL[@]}")
fi

if [ "$build" -eq 1 ]; then
    for version in mpi-corners mpi-corners-pipelined mpi-corners-waitany comm_strats_timer comm_comp; do
        make VERSION="$version"
    done
fi
for item in "${STRATEGIES[@]}"; do
    binary=$(echo "$item" | cut -d: -f2)
    [ -x "$binary" ] || { echo "Missing executable $binary" >&2; exit 2; }
done

# Squarest rank grid: the partitioner picks size_x as the largest divisor of
# the rank count that is <= sqrt(ranks), and maps size_x onto nx. Mirroring
# that here is what keeps every rank's subdomain exactly local_n x local_n.
rank_grid_x() {
    local n=$1 i
    for (( i = $(awk -v n="$n" 'BEGIN{print int(sqrt(n))}'); i >= 1; i-- )); do
        if [ $((n % i)) -eq 0 ]; then echo "$i"; return; fi
    done
    echo 1
}

IFS=',' read -r -a RANK_COUNTS <<< "$ranks_csv"
total=$(( ${#RANK_COUNTS[@]} * repeats * ${#STRATEGIES[@]} ))
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
        for item in "${STRATEGIES[@]}"; do
            label=$(echo "$item" | cut -d: -f1)
            binary=$(echo "$item" | cut -d: -f2)
            extra=$(echo "$item" | cut -d: -f3)
            min_ranks=$(echo "$item" | cut -d: -f4)
            run_count=$((run_count + 1))
            if [ "$ranks" -lt "$min_ranks" ]; then
                echo "=== [${run_count}/${total}] ${label} ${mode} N=${nodes} n=${ranks} rep=${rep} -- SKIPPED (deadlocks below ${min_ranks} ranks, no data) ==="
                continue
            fi
            echo "=== [${run_count}/${total}] ${label} ${mode} N=${nodes} n=${ranks} rep=${rep} ==="
            run_case --label "$label" --binary "$binary" \
                --results-dir "$results_dir" --mode "$mode" --rep "$rep" \
                --nodes "$nodes" --ranks "$ranks" --cores 1 \
                --nx "$run_nx" --ny "$run_ny" --nz "$nz" --num_iter "$num_iter" \
                "${geom_args[@]}" \
                --env "OMP_NUM_THREADS=1" \
                --srun-extra "--ntasks-per-node=${tasks_per_node} --gpus-per-task=1 --cpu-bind=cores -t ${time_limit}" \
                ${extra:+--app-extra "$extra"} \
                || status=1
        done
    done
done

echo "Done. ${run_count} runs written to ${results_dir}/"
exit "$status"
