#!/usr/bin/env bash

module load cray-python/3.8.5.0
dir=$(pwd)
cd ~
if [ -d "hpc_env" ]; then
	echo "Environment does exist. Running cmake."
    source ~/hpc_env/bin/activate
    cd $dir/..
    mkdir -p build
    cd build
    cmake ..
else
	echo "Environment does not exist. Run 'bash create_env.sh' first."
fi
