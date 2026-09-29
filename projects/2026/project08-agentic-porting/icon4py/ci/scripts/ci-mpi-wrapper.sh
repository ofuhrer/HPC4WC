#!/usr/bin/env bash

# Log all output to separate logfiles, stored as artifacts in gitlab. Output to
# stdout only from rank 0.

set -euo pipefail

# Check a few different possibilities for the rank.
if [[ -n "${PMI_RANK:-}" ]]; then
    rank="${PMI_RANK}"
elif [[ -n "${OMPI_COMM_WORLD_RANK:-}" ]]; then
    rank="${OMPI_COMM_WORLD_RANK}"
elif [[ -n "${SLURM_PROCID:-}" ]]; then
    rank="${SLURM_PROCID}"
else
    echo "Could not determine MPI rank. Set PMI_RANK, OMPI_COMM_WORLD_RANK, or SLURM_PROCID."
    exit 1
fi

log_file="${CI_PROJECT_DIR:+${CI_PROJECT_DIR}/}pytest-log-rank-${rank}.txt"

# If ICON4PY_TEST_MPI_SUBCOMM_SIZE is set, print output from the first rank in
# each subcommunicator group (non-overlapping test sets). Otherwise only rank 0.
subcomm_size="${ICON4PY_TEST_MPI_SUBCOMM_SIZE:-0}"
if [[ "${subcomm_size}" -gt 0 ]]; then
    subcomm_rank=$(( rank % subcomm_size ))
else
    subcomm_rank="${rank}"
fi

if [[ "${subcomm_rank}" -eq 0 ]]; then
    echo "Starting pytest on rank ${rank}, logging to stdout and ${log_file}"
    "$@" |& tee "${log_file}"
else
    echo "Starting job on rank ${rank}, logging to ${log_file}"
    "$@" >& "${log_file}"
fi
