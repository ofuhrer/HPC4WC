## Shell commands version-a4
# allow to execute with: chmod u+x name.sh

module load daint-gpu
module load Boost

rm a4.txt # delete existing file

#backend = numpy
echo "numpy" >> a4.txt
#strong scaling sized 64 * 64
echo "64" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt


#strong scaling sized 128*128
echo "128" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt

#strong scaling sized 256*256
echo "256" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt

#strong scaling sized 512*512
echo "512" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt

#strong scaling sized 1024*1024
echo "1024" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt

# weak scaling 
echo "weak" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=numpy >> a4.txt

#backend = gt:cpu_ifirst
echo "gt:cpu_ifirst" >> a4.txt
#strong scaling sized 64 * 64
echo "64" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt

#strong scaling sized 128*128
echo "128" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt

#strong scaling sized 256*256
echo "256" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt

#strong scaling sized 512*512
echo "512" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt

#strong scaling sized 1024*1024
echo "1024" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt

# weak scaling 
echo "weak" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_ifirst >> a4.txt

#backend = gt:cpu_kfirst
echo "gt:cpu_kfirst" >> a4.txt
#strong scaling sized 64 * 64
echo "64" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt

#strong scaling sized 128*128
echo "128" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt

#strong scaling sized 256*256
echo "256" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt

#strong scaling sized 512*512
echo "512" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt

#strong scaling sized 1024*1024
echo "1024" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt

# weak scaling 
echo "weak" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:cpu_kfirst >> a4.txt

#backend = gt:gpu
echo "gt:gpu" >> a4.txt
#strong scaling sized 64 * 64
echo "64" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt

#strong scaling sized 128*128
echo "128" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt

#strong scaling sized 256*256
echo "256" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt

#strong scaling sized 512*512
echo "512" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt

#strong scaling sized 1024*1024
echo "1024" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt

# weak scaling 
echo "weak" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=gt:gpu >> a4.txt

#backend = gt:cuda
echo "gt:cuda" >> a4.txt
#strong scaling sized 64 * 64
echo "64" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt

#strong scaling sized 128*128
echo "128" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt

#strong scaling sized 256*256
echo "256" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt

#strong scaling sized 512*512
echo "512" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt

#strong scaling sized 1024*1024
echo "1024" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 2 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 8 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 32 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 1024 --ny 1024 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt

# weak scaling
echo "weak" >> a4.txt
srun -n 1 python stencil2d-gt4py-a4.py --nx 64 --ny 64 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 4 python stencil2d-gt4py-a4.py --nx 128 --ny 128 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 16 python stencil2d-gt4py-a4.py --nx 256 --ny 256 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
srun -n 64 python stencil2d-gt4py-a4.py --nx 512 --ny 512 --nz 64 --num_iter 256 --plot_result False --backend=cuda >> a4.txt
