#!/bin/bash -l

#SBATCH --constraint      gpu
#SBATCH --nodes           1
#SBATCH --time            01:00:00
#SBATCH --partition       normal
#SBATCH --ntasks-per-core 1
#SBATCH --hint            nomultithread
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task   12

# set -euo pipefail
IFS=$'\n\t'

project_root=$(git rev-parse --show-toplevel)/projects2020/group03
source ${project_root}/scripts/compilers.sh
source ${project_root}/scripts/versions.sh

IFS=' '
args="--nx 128 --ny 128 --nz 64 --num_iter 1024"

export OMP_NUM_THREADS=12
export CRAY_CUDA_MPS=1
export OMP_TARGET_OFFLOAD="MANDATORY"

cd ${project_root}/build

for compiler in ${compilers[@]}; do
	cd ${compiler}/src

	load_compiler ${compiler}

	for version in ${versions[@]}; do
		if [[ -e ./${version} && -x ./${version} ]]; then
			echo "Running ${compiler} ${version}"
			folder=${version%/*}
			binary=${version#*/}
			cd ${folder}
			if [[ ${compiler} == "gnu" && ${folder} == "openmp_split" ]]; then # doesn't work
				true
			elif [[ ${compiler} == "intel" && ${folder} == "openmp" ]]; then # doesn't work
				true
			#elif [[ ${folder} == "mpi" ]]; then
			#	srun --time 00:02:00 --ntasks-per-node 12 --cpus-per-task 1 ./${binary} ${args}
			else
				srun --time 00:02:00 ./${binary} ${args}
			fi
			cd ..
		fi
	done

	cd ../..
done
