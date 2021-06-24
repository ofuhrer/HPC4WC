#!/bin/bash

nz=64
num_iter=1024
export OMP_NUM_THREADS=12
for n in 128 256 512 1024
do
    echo "size $n"
    for count in {1..10}
    do
        nx=${n}
        ny=${n}
        srun ./arrayFusionACC.x --nx ${nx} --ny ${ny} --nz ${nz} --num_iter ${num_iter}
    done 
done 