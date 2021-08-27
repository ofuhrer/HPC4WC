#!/usr/bin/env bash

cd ~

module load cray-python/3.8.5.0

if [ -d "hpc_env" ]; then
	echo "Environment already exists."
else
	python -m venv hpc_env
fi

source hpc_env/bin/activate
pip install cmake==3.21.1

