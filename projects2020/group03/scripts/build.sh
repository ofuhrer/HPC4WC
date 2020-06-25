#!/bin/bash

set -euo pipefail

cd $(dirname ${BASH_SOURCE[0]})/..

mkdir -p build
cd build

BUILD_TYPE=${1:-Release}
PRGENV=$(module list 2>&1 | sed -E -n -e 's!.*(PrgEnv-\w+)/.*!\1!p')

for compiler in cray gnu intel pgi; do
	echo "Building ${BUILD_TYPE} using ${compiler}"
	mkdir -p ${compiler}
	cd ${compiler}
	module switch ${PRGENV} PrgEnv-${compiler}
	cmake -D CMAKE_BUILD_TYPE=${BUILD_TYPE} ../..
	cmake --build .
	module switch PrgEnv-${compiler} ${PRGENV}
	cd ..
done
