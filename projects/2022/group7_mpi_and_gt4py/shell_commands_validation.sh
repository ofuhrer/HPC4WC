## Shell commands validation, all 512*512*64
# allow to execute with: chmod u+x name.sh


module load daint-gpu
module load Boost
 
# version a4
srun -n 16 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 1024 --plot_result True --backend=numpy # runs 80 seconds
srun -n 16 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 1024 --plot_result True --backend=gt:cpu_ifirst 
srun -n 16 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 1024 --plot_result True --backend=gt:cpu_kfirst
srun -n 16 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 1024 --plot_result True --backend=gt:gpu
srun -n 16 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 1024 --plot_result True --backend=cuda

# version a1 
srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 1024 --plot_result True --backend=numpy 
srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 1024 --plot_result True --backend=gt:cpu_ifirst
srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 1024 --plot_result True --backend=gt:cpu_kfirst
srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 1024 --plot_result True --backend=gt:gpu 
srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 1024 --plot_result True --backend=cuda

# OG version, 
srun -n 1 python stencil2d.py --nx 512 --ny 512 --nz 64 --num_iter 1024 --plot_result True # runs 10 min