# CUDA version

## Compiling
```bash
source ./modules.sh
./compile.sh
```

## Running
```bash
sbatch stencil2d_cpu.slurm
sbatch stencil2d_gpu.slurm
sbatch stencil2d_hybrid.slurm
```
