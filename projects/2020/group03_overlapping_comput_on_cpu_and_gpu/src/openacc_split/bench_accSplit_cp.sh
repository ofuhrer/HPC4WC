#!/bin/bash

num_iter=1024
nz=64
nx=128 
ny=128 

export OMP_NUM_THREADS=12
for n in {1..16}
do
    num_split=${n}
    echo "offloaded to cpu : $num_split slices"
    for count in {1..10}
    do
        srun ./arrayFusionACCSplit.x --nx ${nx} --ny ${ny} --nz ${nz} --num_iter ${num_iter} --numSplit ${num_split}  
    done 
done 