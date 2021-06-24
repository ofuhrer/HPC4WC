#!/bin/bash -l

set -euo pipefail
IFS=$'\n\t'

versions=(
#	"orig/stencil2d-orig" # uses floats instead of doubles
#	"mpi/stencil2d-mpi"
	"seq/stencil2d_seq"
	"seq/stencil2d_seq_base_cpp"
	"seq/stencil2d_seq_base_array_cpp"
	"seq/stencil2d_seq_arrayFusion_cpp"
	"openmp/stencil2d_openmp"
	"openmp_target/stencil2d_openmp_target"
	"openmp_split/stencil2d_openmp_split"
	"openacc/stencil2d_openacc"
	"openacc/stencil2d-base-array_acc"
	"openacc/stencil2d-arrayFusion-acc"
	"openacc_split/stencil2d_openacc_split"
)
