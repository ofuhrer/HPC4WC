#!/bin/bash -l

#SBATCH --constraint gpu
#SBATCH --nodes      1
#SBATCH --time       00:20:00
#SBATCH --partition  debug

# set -euo pipefail
IFS=$'\n\t'

project_root=$(git rev-parse --show-toplevel)/projects2020/group03
source ${project_root}/scripts/compilers.sh
source ${project_root}/scripts/versions.sh

IFS=' '
args="--nx 128 --ny 128 --nz 64 --num_iter 1024"

OMP_TARGET_OFFLOAD="MANDATORY"

cd ${project_root}/build

for compiler in ${compilers[@]}; do
	cd ${compiler}/src

	load_compiler ${compiler}

	for version in ${versions[@]}; do
		if [[ -e ./${version} && -x ./${version} ]]; then
			echo "Running ${compiler} ${version}"
			cd ${version%/*}
			srun --time 00:02:00 ./${version#*/} ${args}
			cd ..
		fi
	done

	cd ../..
done
