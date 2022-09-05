## Shell commands version a1&a4 for strong scaling 2048*2048 grid
# allow to execute with: chmod u+x name.sh

module load daint-gpu
module load Boost

rm a1_a4_2048.txt # delete existing file

#backend = gt:cpu_ifirst
echo "gt:cpu_ifirst" >> a1_a4_2048.txt
echo "a4" >> a1_a4_2048.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
echo "a1" >> a1_a4_2048.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a1_a4_2048.txt

#backend = gt:cpu_kfirst
echo "gt:cpu_kfirst" >> a1_a4_2048.txt
echo "a4" >> a1_a4_2048.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
echo "a1" >> a1_a4_2048.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a1_a4_2048.txt

#backend = gt:gpu
echo "gt:gpu" >> a1_a4_2048.txt
echo "a4" >> a1_a4_2048.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
echo "a1" >> a1_a4_2048.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a1_a4_2048.txt

#backend = cuda
echo "cuda" >> a1_a4_2048.txt
echo "a4" >> a1_a4_2048.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
echo "a1" >> a1_a4_2048.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a1_a4_2048.txt

#backend = numpy
echo "numpy" >> a1_a4_2048.txt
echo "a4" >> a1_a4_2048.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
echo "a1" >> a1_a4_2048.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 2048 --ny 2048 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a1_a4_2048.txt