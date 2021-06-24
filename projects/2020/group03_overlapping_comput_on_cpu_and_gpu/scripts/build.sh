#!/bin/bash -l

# set -euo pipefail
source $(dirname ${BASH_SOURCE[0]})/compilers.sh

cd $(dirname ${BASH_SOURCE[0]})/..

mkdir -p build
cd build

BUILD_TYPE=${1:-Release}

for compiler in ${compilers[@]}; do
	echo "Building ${BUILD_TYPE} using ${compiler}"
	mkdir -p ${compiler}
	cd ${compiler}

	load_compiler ${compiler}

	cmake -D CMAKE_BUILD_TYPE=${BUILD_TYPE} ../..
	cmake --build .

	cd ..
done
