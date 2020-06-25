#!/bin/sh
module load daint-gpu cudatoolkit

SSL_SPACK_ROOT=/apps/daint/SSL/software/spack-current
export PATH=${PATH}:${SSL_SPACK_ROOT}/bin
source ${SSL_SPACK_ROOT}/share/spack/setup-env.sh
spack load cmake@3.17.1
