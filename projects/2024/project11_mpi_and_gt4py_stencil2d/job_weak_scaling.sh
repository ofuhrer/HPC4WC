#!/bin/bash -l
#SBATCH --export=ALL
#SBATCH --error=GT4py_mpi.err
#SBATCH --output=GT4py_mpi.out
#SBATCH --exclusive

#SBATCH --job-name="GT4py_mpi"
#SBATCH --account="class03"
#SBATCH --time=00:03:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

module load daint-gpu
module load jupyterlab/3.4.5-CrayGNU-21.09-batchspawner-cuda
module load Boost
module load cudatoolkit
module load FFmpeg
NVCC_PATH=$(which nvcc)
CUDA_PATH=$(echo $NVCC_PATH | sed -e "s/\/bin\/nvcc//g")
export CUDA_HOME=$CUDA_PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

source ~/HPC4WC_venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun python stencil2d-gt4py-mpi-base.py --nx 128 --ny 128 --nz 64 --num_iter 128 --backend cuda --plot_result True
# srun python stencil2d-gt4py-mpi-v2.py --nx 128 --ny 128 --nz 64 --num_iter 128 --backend cuda --plot_result True