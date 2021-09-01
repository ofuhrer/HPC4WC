#### Srun commands

srun -C gpu -A class03 --time=0:01:00 --nodes=1 --ntasks-per-core=2 --ntasks-per-node=1 --cpus-per-task=24 --partition=normal --hint=multithread python ./src/main.py --out data/jax_cpu_full_2d.csv --libs jax --ns 8 16 32 64 128 256 512 1024 2048 4096 8192 16384



#### Plots we actually put into the report

###### Time/n, Lap 2D, CPU: jax, bohrium, numpy, GT4Py:

python src/plotting.py --input data/bohrium_cpu_2d.csv data/numpy_2d.csv data/gt4py_cpu_2d.csv data/jax_cpu_full_2d.csv --stencils Laplacian2D --output laplacian2d_cpu.pdf --logx --logy --title="Runtime of the 2D Laplacian stencil [CPU]"

###### Time/n, Bih 2D, CPU: jax, bohrium, numpy, GT4Py:

python src/plotting.py --input data/bohrium_cpu_2d.csv data/numpy_2d.csv data/gt4py_cpu_2d.csv data/jax_cpu_full_2d.csv --stencils Biharmonic2D --output biharmonic2d_cpu.pdf --logx --logy --title="Runtime of the 2D Biharmonic stencil [CPU]"

###### Time/n, Lap 3D, CPU: jax, bohrium, numpy, GT4Py:

python src/plotting.py --input data/bohrium_cpu_3d.csv data/numpy_3d.csv data/gt4py_cpu_3d.csv data/jax_cpu_full_3d.csv --stencils Laplacian3D --output laplacian3d_cpu.pdf --logx --logy --title="Runtime of the 3D Laplacian stencil [CPU]"

###### Bih 3D, CPU: jax, bohrium, numpy, GT4Py:

python src/plotting.py --input data/bohrium_cpu_3d.csv data/numpy_3d.csv data/gt4py_cpu_3d.csv data/jax_cpu_full_3d.csv --stencils Biharmonic3D --output biharmonic3d_cpu.pdf --logx --logy --title="Runtime of the 3D Biharmonic stencil [CPU]"

###### Time/n, Lap 2D, GPU: jax, bohrium, numpy (CPU), GT4Py:

python src/plotting.py --input data/bohrium_gpu_final.csv data/numpy_2d.csv data/gt4py_gpu_2d.csv data/jax_gpu_2d.csv --stencils Laplacian2D --output laplacian2d_gpu.pdf --logx --logy --title="Runtime of the 2D Biharmonic stencil [GPU]"

###### Barchart: cycles(time) / intensity, Lap 2D & Bih 2D: jax, bohrium, numpy, GT4Py:

python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_2d.csv data/gt4py_cpu_2d.csv data/jax_cpu_full_2d.csv --stencils Laplacian2D --logx --perfplot --output perf_laplacian2d_cpu.pdf --title="Performance of the 2D Laplacian stencil [CPU]"

python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_2d.csv data/gt4py_cpu_2d.csv data/jax_cpu_full_2d.csv --stencils Biharmonic2D --logx --perfplot --output perf_biharmonic2d_cpu.pdf --title="Performance of the 2D Biharmonic stencil [CPU]"

###### Barchart: Speedup,  Lap 2D GPU & CPU: jax, bohrium, GT4Py:

python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_2d.csv data/gt4py_cpu_2d.csv data/jax_cpu_full_2d.csv --stencils Laplacian2D --speedup="bar" --output speedupbar_laplacian2d_cpu.pdf --title="Speedup w.r.t. NumPy, Laplacian2D stencil [CPU]"

### Runtime plots

#### CPU

* **laplacian2d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_2d.csv data/gt4py_cpu_2d.csv data/jax_cpu_full_2d.csv --stencils Laplacian2D --output laplacian2d_cpu.pdf --logx --logy --title="Achieved runtime on a 2D grid [CPU]"
* **biharmonic2d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_2d.csv data/gt4py_cpu_2d.csv data/jax_cpu_full_2d.csv --stencils Biharmonic2D --output biharmonic2d_cpu.pdf --logx --logy --title="Achieved runtime on a 2D grid [CPU]"
* **laplacian3d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_3d.csv data/gt4py_cpu_3d.csv data/jax_cpu_full_3d.csv --stencils Laplacian3D --output laplacian3d_cpu.pdf --logx --logy --title="Achieved runtime on a 3D grid [CPU]"
* **biharmonic3d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_3d.csv data/gt4py_cpu_3d.csv data/jax_cpu_full_3d.csv --stencils Biharmonic3D --output biharmonic3d_cpu.pdf --logx --logy --title="Achieved runtime on a 3D grid [CPU]"

#### GPU

* **laplacian2d:** python src/plotting.py --input data/bohrium_gpu_final.csv data/numpy_2d.csv data/gt4py_gpu_2d.csv data/jax_gpu_2d.csv --stencils Laplacian2D --output laplacian2d_gpu.pdf --logx --logy --title="Achieved runtime on a 2D grid [GPU]"
* **biharmonic2d:** python src/plotting.py --input data/bohrium_gpu_final.csv data/numpy_2d.csv data/gt4py_gpu_2d.csv data/jax_gpu_2d.csv --stencils Biharmonic2D --output biharmonic2d_gpu.pdf --logx --logy --title="Achieved runtime on a 2D grid [GPU]"

### Performance plots

#### CPU

- **laplacian2d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_2d.csv data/gt4py_cpu_2d.csv data/jax_cpu_full_2d.csv --stencils Laplacian2D --logx --perfplot --output perf_laplacian2d_cpu.pdf --title="Approximate performance normalized per stencil [CPU]"
- **biharmonic2d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_2d.csv data/gt4py_cpu_2d.csv data/jax_cpu_full_2d.csv --stencils Biharmonic2D --logx --perfplot --output perf_biharmonic2d_cpu.pdf --title="Approximate performance normalized per stencil [CPU]"
- **laplacian3d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_3d.csv data/gt4py_cpu_3d.csv data/jax_cpu_full_3d.csv --stencils Laplacian3D --output perf_laplacian3d_cpu.pdf --logx --perfplot --title="Approximate performance normalized per stencil [CPU]"
- **biharmonic3d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_3d.csv data/gt4py_cpu_3d.csv data/jax_cpu_full_3d.csv --stencils Biharmonic3D --output perf_biharmonic3d_cpu.pdf --logx --perfplot --title="Approximate performance normalized per stencil [CPU]"

#### GPU

* **laplacian2d:** python src/plotting.py --input data/bohrium_gpu_final.csv data/numpy_2d.csv data/gt4py_gpu_2d.csv data/jax_gpu_2d.csv --stencils Laplacian2D --output perf_laplacian2d_gpu.pdf --logx --logy --perfplot --title="Approximate performance normalized per stencil [GPU]"
* **biharmonic2d:** python src/plotting.py --input data/bohrium_gpu_final.csv data/numpy_2d.csv data/gt4py_gpu_2d.csv data/jax_gpu_2d.csv --stencils Biharmonic2D --output perf_biharmonic2d_gpu.pdf --logx --perfplot --title="Approximate performance normalized per stencil [GPU]"



### Speedup (Barcharts)

#### GPU

* **biharmonic2d:** python src/plotting.py --input data/bohrium_gpu_final.csv data/numpy_2d.csv data/gt4py_gpu_2d.csv data/jax_gpu_2d.csv --stencils Biharmonic2D --output speedupbar_biharmonic2d_gpu.pdf --speedup="bar" --title="Speedup w.r.t. NumPy, Biharmonic2D stencil [GPU]"

#### CPU

* **laplacian2d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_2d.csv data/gt4py_cpu_2d.csv data/jax_cpu_full_2d.csv --stencils Laplacian2D --speedup="bar" --output speedupbar_laplacian2d_cpu.pdf --title="Speedup w.r.t. NumPy, Laplacian2D stencil [CPU]"
* **biharmonic3d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_3d.csv data/gt4py_cpu_3d.csv data/jax_cpu_full_3d.csv --stencils Biharmonic3D --output speedupbar_biharmonic3d_cpu.pdf --speedup="bar" --title="Speedup w.r.t. NumPy, Biharmonic3D stencil [CPU]"



### Speedup (Lineplot)

#### GPU

- **biharmonic2d:** python src/plotting.py --input data/bohrium_gpu_final.csv data/numpy_2d.csv data/gt4py_gpu_2d.csv data/jax_gpu_2d.csv --stencils Biharmonic2D --logy --logx --output speedupline_biharmonic2d_gpu.pdf --speedup="line" --title="Speedup w.r.t. NumPy [GPU]"

#### CPU

- **laplacian2d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_2d.csv data/gt4py_cpu_2d.csv data/jax_cpu_full_2d.csv --stencils Laplacian2D --speedup="line" --logy --logx --output speedupline_laplacian2d_cpu.pdf --title="Speedup w.r.t. NumPy [CPU]"
- **biharmonic3d:** python src/plotting.py --input data/bohrium_cpu_final_40_iter.csv data/numpy_3d.csv data/gt4py_cpu_3d.csv data/jax_cpu_full_3d.csv --stencils Biharmonic3D --logy --logx --output speedupline_biharmonic3d_cpu.pdf --speedup="line" --title="Speedup w.r.t. NumPy [CPU]"