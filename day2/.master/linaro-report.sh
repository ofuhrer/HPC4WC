#!/bin/bash

LD_LIBRARY_PATH=/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/cray-gtl-8.1.30-yjzx5mja37woqidkb7lj6lpbu3omqrsr/lib:$LD_LIBRARY_PATH /capstor/scratch/cscs/piccinal/linaro/25.0/bin/perf-report $@

