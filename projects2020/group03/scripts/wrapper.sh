#!/bin/bash

compiler=$(basename $(realpath ../..))
version=$(basename ${1})
nz=64
num_iter=1024
for n in 128 256 512 1024; do
	nx=${n}
	ny=${n}
	time=$(./${1} --nx ${nx} --ny ${ny} --nz ${nz} --num_iter ${num_iter} | sed -E -n -e 's/\[(\s*[0-9]+,){5}\s*(.*)\].*/\2/p')
	echo ${compiler} ${version} ${nx} ${ny} ${nz} ${num_iter} ${time}
done
