#!/usr/bin/env bash
# Validate the six runtime strategies, mpi_leftright baseline, three corner
# variants, and comm_comp against one serial reference field, preserving
# commands, logs, and fields.
set -euo pipefail

# Resolve executables, the Python environment, and output paths from the repo root.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

out_dir=${1:-results/validation}
# compare_fields_f90.py needs numpy and click, which live in .venv. That venv
# points into /user-environment, so this script must be run inside the uenv:
#   uenv run prgenv-gnu/26.3:v1 --view=default -- ./validation/validate_all_strategies.sh
# Falling back to a bare python3 makes every comparison fail with
# ModuleNotFoundError, which the summary reports as MISMATCH.
python_cmd=${PYTHON_CMD:-./.venv/bin/python}
nx=128
ny=128
nz=4
num_iter=1024

mkdir -p "$out_dir/fields" "$out_dir/logs"

git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
git_status=$(git status --porcelain --untracked-files=all 2>/dev/null || true)
{
    echo "validation_start = $(date -Is)"
    echo "git_commit = $git_commit"
    echo "git_dirty = $([ -n "$git_status" ] && echo 1 || echo 0)"
    echo "grid = ${nx}x${ny}x${nz}"
    echo "num_iter = $num_iter"
    echo "parallel_layout = 16 ranks, 4 nodes, 4 ranks/node, 1 GH200/rank"
} > "$out_dir/manifest.txt"
printf '%s\n' "$git_status" > "$out_dir/git_status.txt"
git diff HEAD -- > "$out_dir/code_state.diff"

# CWP metadata propagation can lag very briefly after an srun exits.
wait_for_field() {
    local attempt
    for attempt in $(seq 1 40); do
        [ -f out_field.dat ] && return 0
        sleep 0.25
    done
    echo "Timed out waiting for out_field.dat" >&2
    return 1
}

serial_cmd=(srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default \
    -N 1 -n 1 -c 1 --ntasks-per-node=1 --gpus-per-task=1 --cpu-bind=cores \
    ./stencil2d-orig.x --nx "$nx" --ny "$ny" --nz "$nz" --num_iter "$num_iter")
printf '%q ' "${serial_cmd[@]}" > "$out_dir/logs/serial.command"
printf '\n' >> "$out_dir/logs/serial.command"
[ ! -f out_field.dat ] || find out_field.dat -maxdepth 0 -delete
OMP_NUM_THREADS=1 "${serial_cmd[@]}" > "$out_dir/logs/serial.stdout" 2> "$out_dir/logs/serial.stderr"
wait_for_field
mv out_field.dat "$out_dir/fields/serial.dat"

printf 'strategy\tbinary\tstatus\n' > "$out_dir/summary.tsv"

validate_one() {
    local strategy=$1 binary=$2
    shift 2
    local -a extra=("$@")
    local -a cmd=(srun -A hpc4wc-course2026-ethz --uenv=prgenv-gnu/26.3:v1 --view=default \
        -N 4 -n 16 -c 1 --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=cores \
        "$binary" --nx "$nx" --ny "$ny" --nz "$nz" --num_iter "$num_iter" "${extra[@]}")

    printf '%q ' "${cmd[@]}" > "$out_dir/logs/${strategy}.command"
    printf '\n' >> "$out_dir/logs/${strategy}.command"
    [ ! -f out_field.dat ] || find out_field.dat -maxdepth 0 -delete
    if ! OMP_NUM_THREADS=1 "${cmd[@]}" > "$out_dir/logs/${strategy}.stdout" 2> "$out_dir/logs/${strategy}.stderr"; then
        printf '%s\t%s\tRUN_FAILED\n' "$strategy" "$binary" >> "$out_dir/summary.tsv"
        return 1
    fi
    wait_for_field || {
        printf '%s\t%s\tNO_FIELD\n' "$strategy" "$binary" >> "$out_dir/summary.tsv"
        return 1
    }
    mv out_field.dat "$out_dir/fields/${strategy}.dat"
    if "$python_cmd" "$repo_root/validation/compare_fields_f90.py" \
        --src "$out_dir/fields/serial.dat" --trg "$out_dir/fields/${strategy}.dat" \
        --rtol 1e-4 --atol 1e-4 \
        > "$out_dir/logs/${strategy}.compare" 2>&1; then
        printf '%s\t%s\tPASS\n' "$strategy" "$binary" >> "$out_dir/summary.tsv"
    else
        printf '%s\t%s\tMISMATCH\n' "$strategy" "$binary" >> "$out_dir/summary.tsv"
        return 1
    fi
}

status=0
for strategy in brokencycles sendrecv_evenodd sendrecv irecvsend isendrecv irecvisend; do
    validate_one "$strategy" ./stencil2d-comm_strats_timer.x --comm_strategy "$strategy" || status=1
done
validate_one mpi_leftright ./stencil2d-mpi.x || status=1
validate_one corners ./stencil2d-mpi-corners.x || status=1
validate_one pipelined ./stencil2d-mpi-corners-pipelined.x || status=1
validate_one waitany ./stencil2d-mpi-corners-waitany.x || status=1
validate_one comm_comp ./stencil2d-comm_comp.x || status=1

echo "Validation evidence written to $out_dir/"
column -t -s $'\t' "$out_dir/summary.tsv" 2>/dev/null || cat "$out_dir/summary.tsv"
exit "$status"
