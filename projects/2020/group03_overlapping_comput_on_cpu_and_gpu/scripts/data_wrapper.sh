#!/bin/bash

version=$(basename ${1})
nx=${2}
ny=${2}
nz=${3}
num_iter=${4}
z_slices_on_cpu=${5}

time=$(./${1} --nx ${nx} --ny ${ny} --nz ${nz} --num_iter ${num_iter} --z_slices_on_cpu ${z_slices_on_cpu} | sed -E -n -e 's/\[(\s*[0-9]+,){5}\s*(.*)\s*\].*/\2/p')
echo ${version} ${nx} ${ny} ${nz} ${num_iter} ${z_slices_on_cpu} ${time}
