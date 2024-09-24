REALSIZE=8

ACCCXX=nvc++
ACCFLAGS=-std=c++11 -O4 -tp=native -acc=gpu -gpu=cc60 -Minfo=accel

CUDACXX=nvcc
CUDAFLAGS=-std=c++11 -O3 -use_fast_math -extra-device-vectorization -arch=sm_60 -restrict -dlto -Xcompiler "-Ofast -march=native -Wall -Wextra -Wshadow -Wno-unknown-pragmas"

# Alternative CUDA compilation configuration utilising nvc++ instead of nvcc:
# CUDACXX=nvc++
# CUDAFLAGS=-std=c++11 -O4 -tp=native -gpu=cc60

RM=rm -f
