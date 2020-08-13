#!/bin/bash -l

set -euo pipefail
IFS=$'\n\t'

compilers=(
	"gnu"
	"intel"
	"cray"
	"pgi"
)
modules_pre=(
	"modules"
	"craype"
	"cray-mpich"
	"slurm"
	"daint-gpu"
)
modules_post=(
	"xalt"
	"cudatoolkit"
)
modules_gnu=(
	"PrgEnv-gnu"
	"craype-accel-nvidia60"
)
modules_intel=(
	"PrgEnv-intel"
#	"perftools-lite-gpu"
)
modules_cray=(
	"PrgEnv-cray"
	"craype-accel-nvidia60"
)
modules_pgi=(
	"PrgEnv-pgi"
#	"perftools-lite-gpu"
)

load_compiler() {
	local modules_compiler
	local compiler
	compiler=$1

	module purge
	module load ${modules_pre[@]}
	modules_compiler=modules_${compiler}
	modules_compiler="${modules_compiler}[@]"
	module load ${!modules_compiler}
	module load ${modules_post[@]}
	# module list
}
