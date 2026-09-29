#!/usr/bin/env bash

# Start CUDA MPS server. CUDA_VISIBLE_DEVICES must already be set to select the correct GPU.
# This script is adapted from
# https://docs.cscs.ch/running/slurm/#multiple-ranks-per-gpu for used on shared
# partitions.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES is not set. It must be set to correctly start MPS on shared slurm partitions."
    exit 1
fi

mps_prefix="/tmp/$(id -un)/slurm-${SLURM_JOBID}.${SLURM_STEPID}/nvidia"

export CUDA_MPS_PIPE_DIRECTORY=${mps_prefix}-mps
export CUDA_MPS_LOG_DIRECTORY=${mps_prefix}-log
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}"
mkdir -p "${CUDA_MPS_LOG_DIRECTORY}"

if [[ "${SLURM_LOCALID}" -eq 0 ]]; then
    # Only start the MPS server on the first local rank to avoid multiple
    # servers on the same node.
    nvidia-cuda-mps-control -d
fi

pid_file="${CUDA_MPS_PIPE_DIRECTORY}/nvidia-cuda-mps-control.pid"
mps_pid_file_timeout=120
if ! timeout ${mps_pid_file_timeout} bash -c "until [[ -f \"${pid_file}\" ]]; do sleep 1; done"; then
    echo "The MPS wrapper script timed out waiting for MPS pid file ${pid_file} on rank ${SLURM_PROCID}. MPS daemon likely did not start correctly or the rank starting the MPS daemons took too long to start."
    exit 1
fi

echo "Using MPS server with GPU ${CUDA_VISIBLE_DEVICES}, MPS pipe directory ${CUDA_MPS_PIPE_DIRECTORY}, MPS log directory ${CUDA_MPS_LOG_DIRECTORY} on rank ${SLURM_PROCID}"

# Once the daemon has been started, unset CUDA_VISIBLE_DEVICES as the
# selection has already been made in the daemon.
unset CUDA_VISIBLE_DEVICES
