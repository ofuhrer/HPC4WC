#!/bin/bash -l

#SBATCH --constraint gpu
#SBATCH --nodes      1
#SBATCH --time       00:10:00
#SBATCH --partition  debug

# set -euo pipefail
IFS=$'\n\t'

compilers=(
	"gnu"
	"intel"
	"cray"
	"pgi"
)

versions=(
	"orig/stencil2d-orig"
	"mpi/stencil2d-mpi"
	"openmp/stencil2d_openmp"
	"openmp_target/stencil2d_openmp_target"
	"openacc/stencil2d_openacc"
)

IFS=' '
args="--nx 128 --ny 128 --nz 64 --num_iter 1024"

PRGENV=$(module list 2>&1 | sed -E -n -e 's!.*(PrgEnv-\w+).*!\1!p')

# cd $(dirname ${BASH_SOURCE[0]})/../build
cd /users/$(whoami)/HPC4WC/projects2020/group03/build

for compiler in ${compilers[@]}; do
	cd ${compiler}/src

	module switch ${PRGENV} PrgEnv-${compiler}
	if [[ "${compiler}" == "gnu" || "${compiler}" == "cray" ]]; then
		module load craype-accel-nvidia60
	else
		module unload craype-accel-nvidia60 cray-libsci_acc
	fi

	for version in ${versions[@]}; do
		if [[ -e ./${version} && -x ./${version} ]]; then
			echo "Running ${compiler} ${version%/*}"
			srun ./${version} ${args}
		fi
	done

	module switch PrgEnv-${compiler} ${PRGENV}
	cd ../..
done
