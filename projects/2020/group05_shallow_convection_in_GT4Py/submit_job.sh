#!/bin/bash
sbatch -Cgpu -N 1 -t 60 -e slurm-numpy.err -o slurm-numpy.out -J 'gt4py-numpy-benchmark' --wrap="source ~/HPC4WC_project/env_daint;export GT4PY_BACKEND=numpy;ipython ~/HPC4WC_project/benchmark.py"
sbatch -Cgpu -N 1 -t 60 -e slurm-cuda.err -o slurm-cuda.out -J 'gt4py-cuda-benchmark' --wrap="source ~/HPC4WC_project/env_daint;export GT4PY_BACKEND=gtcuda;ipython ~/HPC4WC_project/benchmark.py"
sbatch -Cgpu -N 1 -t 60 -e slurm-x86.err -o slurm-x86.out -J 'gt4py-x86-benchmark' --wrap="source ~/HPC4WC_project/env_daint;export GT4PY_BACKEND=gtx86;ipython ~/HPC4WC_project/benchmark.py"
