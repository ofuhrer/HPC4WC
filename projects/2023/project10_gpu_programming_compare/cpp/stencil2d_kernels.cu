
#include "stencil2d_common.h"
#include <cuda_runtime_api.h>

template <typename T>
struct IndexFld {
    int x, y, h;
    T* field;
    __device__ IndexFld(int x_, int y_, int h_, T* field_)
        : x(x_), y(y_), h(h_), field(field_)
    { }
    __device__ T& operator()(int i, int j, int k)
    {
        return field[(h + i) + (h + j) * (x + 2 * h) + k * (x + 2 * h) * (y + 2 * h)];
    }
};

template <typename T>
struct IndexBlk {
    int bx, by, h;
    T* field;
    __device__ IndexBlk(int bx_, int by_, int h_, T* field_)
        : bx(bx_), by(by_), h(h_), field(field_)
    { }
    __device__ T& operator()(int bi, int bj)
    {
        return field[(bi + h) + (bj + h) * (bx + 2 * h)];
    }
};

template <typename T>
__global__ void updateHaloX(JobData<T>* d_job)
{
    int x = d_job->x;
    int y = d_job->y;
    int h = d_job->h;
    int i = threadIdx.x + blockIdx.x * blockDim.x - h;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    IndexFld<T> fld(x, y, h, d_job->d_fld);

    fld(i, -1, k) = fld(i, y - 1, k);
    fld(i, -2, k) = fld(i, y - 2, k);
    fld(i, y + 0, k) = fld(i, 0, k);
    fld(i, y + 1, k) = fld(i, 1, k);
}

template <typename T>
__global__ void updateHaloY(JobData<T>* d_job)
{
    int x = d_job->x;
    int y = d_job->y;
    int h = d_job->h;
    int j = threadIdx.y + blockIdx.y * blockDim.y - h;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    IndexFld<T> fld(x, y, h, d_job->d_fld);

    fld(-1, j, k) = fld(x - 1, j, k);
    fld(-2, j, k) = fld(x - 2, j, k);
    fld(x + 0, j, k) = fld(0, j, k);
    fld(x + 1, j, k) = fld(1, j, k);
}

template <typename T>
__global__ void diffusion_noshared1(JobData<T>* d_job)
{
    int x = d_job->x;
    int y = d_job->y;
    int h = d_job->h;
    int i = threadIdx.x + blockIdx.x * blockDim.x - h;
    int j = threadIdx.y + blockIdx.y * blockDim.y - h;
    int k = blockIdx.z;
    bool inF1 = (i >= -1 && i < x + 1 && j >= -1 && j < y + 1);
    IndexFld<T> fld(x, y, h, d_job->d_fld);
    IndexFld<T> swp(x, y, h, d_job->d_swp);
    if (inF1) {
        swp(i, j, k) = -4 * fld(i, j, k)
            + fld(i - 1, j, k) + fld(i + 1, j, k)
            + fld(i, j - 1, k) + fld(i, j + 1, k);
    }
}

template <typename T>
__global__ void diffusion_noshared2(JobData<T>* d_job)
{
    int x = d_job->x;
    int y = d_job->y;
    int h = d_job->h;
    int i = threadIdx.x + blockIdx.x * blockDim.x - h;
    int j = threadIdx.y + blockIdx.y * blockDim.y - h;
    int k = blockIdx.z;
    bool inF0 = (i >= 0 && i < x + 0 && j >= 0 && j < y + 0);
    IndexFld<T> fld(x, y, h, d_job->d_fld);
    IndexFld<T> swp(x, y, h, d_job->d_swp);
    T alpha = d_job->alpha;
    if (inF0) {
        T laplap = -4 * swp(i, j, k)
            + swp(i - 1, j, k) + swp(i + 1, j, k)
            + swp(i, j - 1, k) + swp(i, j + 1, k);
        fld(i, j, k) = fld(i, j, k) - alpha * laplap;
    }
}

template <typename T>
__global__ void diffusion_shared(JobData<T>* d_job)
{
    extern __shared__ char shr[];

    int x = d_job->x;
    int y = d_job->y;
    int h = d_job->h;
    int bx = blockDim.x - 2 * h;
    int by = blockDim.y - 2 * h;
    int gi = blockIdx.x * bx;
    int gj = blockIdx.y * by;
    int bi = threadIdx.x - h;
    int bj = threadIdx.y - h;
    int i = gi + bi;
    int j = gj + bj;
    int k = blockIdx.z;

    bool inF0 = (i < x + 0 && j < y + 0);
    bool inF1 = (i < x + 1 && j < y + 1);
    bool inF2 = (i < x + 2 && j < y + 2);
    bool inB0 = (bi >= -0 && bi < bx + 0 && bj >= -0 && bj < by + 0);
    bool inB1 = (bi >= -1 && bi < bx + 1 && bj >= -1 && bj < by + 1);

    int blkBytes = (bx + 2 * h) * (by + 2 * h) * sizeof(FloatType);
    T* s_blk = reinterpret_cast<T*>(shr);
    T* s_tmp = reinterpret_cast<T*>(shr + blkBytes);

    IndexFld<T> fld(x, y, h, d_job->d_fld);
    IndexFld<T> swp(x, y, h, d_job->d_swp);
    IndexBlk<T> blk(bx, by, h, s_blk);
    IndexBlk<T> tmp(bx, by, h, s_tmp);

    if (inF2) {
        blk(bi, bj) = fld(i, j, k);
    }

    __syncthreads();

    if (inF1 && inB1) {
        tmp(bi, bj) = -4 * blk(bi, bj)
            + blk(bi - 1, bj) + blk(bi + 1, bj)
            + blk(bi, bj - 1) + blk(bi, bj + 1);
    }

    __syncthreads();

    T alpha = d_job->alpha;
    if (inF0 && inB0) {
        T laplap = -4 * tmp(bi, bj)
            + tmp(bi - 1, bj) + tmp(bi + 1, bj)
            + tmp(bi, bj - 1) + tmp(bi, bj + 1);
        blk(bi, bj) = blk(bi, bj) - alpha * laplap;
    }

    __syncthreads();

    if (inF0 && inB0) {
        swp(i, j, k) = blk(bi, bj);
    }
}

template <typename T>
__global__ void swapFields(JobData<T>* d_job)
{
    T* tmp = d_job->d_fld;
    d_job->d_fld = d_job->d_swp;
    d_job->d_swp = tmp;
}

int divCeil(int a, int b)
{
    return (a + b - 1) / b;
}

void stencil3d_noshared(cudaStream_t stream, JobData<FloatType> job, JobData<FloatType>* d_job, bool occ)
{
    int x = job.x;
    int y = job.y;
    int z = job.z;
    int h = job.h;
    int bx = job.b;
    int by = job.b;
    int gx = divCeil(x + 2 * h, bx);
    int gy = divCeil(y + 2 * h, by);

    dim3 blckD(bx, by, 1);
    dim3 gridD(gx, gy, z);
    dim3 blckHX(x + 2 * h, 1, 1);
    dim3 blckHY(1, y + 2 * h, 1);
    dim3 gridH(1, 1, z);

    if (occ) {
        stencil3d_occupancy(blckD, gridD);
        return;
    }

    void* args[] = { &d_job };

    for (int n = 0; n < job.iter; ++n) {
        CheckErrors(cudaLaunchKernel((void*)diffusion_noshared1<FloatType>, gridD, blckD, args, 0, stream));
        CheckErrors(cudaLaunchKernel((void*)diffusion_noshared2<FloatType>, gridD, blckD, args, 0, stream));
        CheckErrors(cudaLaunchKernel((void*)updateHaloX<FloatType>, gridH, blckHX, args, 0, stream));
        CheckErrors(cudaLaunchKernel((void*)updateHaloY<FloatType>, gridH, blckHY, args, 0, stream));
    }
}

void stencil3d_shared(cudaStream_t stream, JobData<FloatType> job, JobData<FloatType>* d_job, bool occ)
{
    int x = job.x;
    int y = job.y;
    int z = job.z;
    int h = job.h;
    int bx = job.b;
    int by = job.b;
    int gx = divCeil(x, bx);
    int gy = divCeil(y, by);

    dim3 blckD(bx + 2 * h, by + 2 * h, 1);
    dim3 gridD(gx, gy, z);
    dim3 blckHX(x + 2 * h, 1, 1);
    dim3 blckHY(1, y + 2 * h, 1);
    dim3 gridH(1, 1, z);

    if (occ) {
        stencil3d_occupancy(blckD, gridD);
        return;
    }

    int fldBytes = (bx + 2 * h) * (by + 2 * h) * sizeof(FloatType);
    int shrBytes = 2 * fldBytes;
    void* args[] = { &d_job };

    for (int n = 0; n < job.iter; ++n) {
        CheckErrors(cudaLaunchKernel((void*)diffusion_shared<FloatType>, gridD, blckD, args, shrBytes, stream));
        CheckErrors(cudaLaunchKernel((void*)swapFields<FloatType>, 1, 1, args, 0, stream));
        CheckErrors(cudaLaunchKernel((void*)updateHaloX<FloatType>, gridH, blckHX, args, 0, stream));
        CheckErrors(cudaLaunchKernel((void*)updateHaloY<FloatType>, gridH, blckHY, args, 0, stream));
    }
}

extern "C" void stencil3d_launch(cudaStream_t stream, JobData<FloatType> job, JobData<FloatType>* d_job, bool noshared, bool occ)
{
    if (noshared)
        stencil3d_noshared(stream, job, d_job, occ);
    else
        stencil3d_shared(stream, job, d_job, occ);
}
