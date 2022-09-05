## Shell commands version-a1
# allow to execute with: chmod u+x name.sh

module load daint-gpu
module load Boost

rm a1.txt # delete existing file

#backend = numpy
echo "numpy" >> a1.txt
#strong scaling sized 64 * 64
echo "64" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt

#strong scaling sized 128*128
echo "128" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt

#strong scaling sized 256*256
echo "256" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt

#strong scaling sized 512*512
echo "512" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt

#strong scaling sized 1024*1024
echo "1024" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt

# weak scaling 
echo "weak" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=numpy >> a1.txt

#backend = gt:cpu_ifirst
echo "gt:cpu_ifirst" >> a1.txt
#strong scaling sized 64 * 64
echo "64" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt

#strong scaling sized 128*128
echo "128" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt

#strong scaling sized 256*256
echo "256" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt

#strong scaling sized 512*512
echo "512" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt

#strong scaling sized 1024*1024
echo "1024" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt

# weak scaling 
echo "weak" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_ifirst >> a1.txt

#backend = gt:cpu_kfirst
echo "gt:cpu_kfirst" >> a1.txt
#strong scaling sized 64 * 64
echo "64" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt

#strong scaling sized 128*128
echo "128" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt

#strong scaling sized 256*256
echo "256" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt

#strong scaling sized 512*512
echo "512" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt

#strong scaling sized 1024*1024
echo "1024" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt

# weak scaling 
echo "weak" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:cpu_kfirst >> a1.txt

#backend = gt:gpu
echo "gt:gpu" >> a1.txt
#strong scaling sized 64 * 64
echo "64" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt

#strong scaling sized 128*128
echo "128" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt

#strong scaling sized 256*256
echo "256" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt

#strong scaling sized 512*512
echo "512" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt

#strong scaling sized 1024*1024
echo "1024" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt

# weak scaling 
echo "weak" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=gt:gpu >> a1.txt

#backend = gt:cuda
echo "gt:cuda" >> a1.txt
#strong scaling sized 64 * 64
echo "64" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt

#strong scaling sized 128*128
echo "128" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt

#strong scaling sized 256*256
echo "256" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt

#strong scaling sized 512*512
echo "512" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt

#strong scaling sized 1024*1024
echo "1024" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 2 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 8 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 32 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 1024 --ny 1024 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt

# weak scaling
echo "weak" >> a1.txt
srun -n 1 python stencil2d-gt4py-a1.py --nx 64 --ny 64 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 4 python stencil2d-gt4py-a1.py --nx 128 --ny 128 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 16 python stencil2d-gt4py-a1.py --nx 256 --ny 256 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
srun -n 64 python stencil2d-gt4py-a1.py --nx 512 --ny 512 --nz 64 --num_iter 64 --plot_result False --backend=cuda >> a1.txt
