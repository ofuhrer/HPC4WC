
#include "stencil2d_common.h"
#include "stencil2d_field3d.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

float stencil2d_cuda(int x, int y, int z, int iter, int b, bool noshared, bool occ, int runs, const char* filename)
{
    // Create field
    Field3D<FloatType> field(x, y, z, const_h);
    initialize(field);
    updateHalo(field);
    FloatType* field_ptr = &field.data[0];
    int field_bytes = field.data.size() * sizeof(FloatType);

    // Create job description
    JobData<FloatType> job;
    job.x = x;
    job.y = y;
    job.z = z;
    job.h = const_h;
    job.alpha = const_alpha;
    job.iter = iter;
    job.b = b;
    job.noshared = noshared;
    job.d_fld = nullptr;
    job.d_swp = nullptr;

    // Init device
    CheckErrors(cudaSetDevice(0));

    // Allocate device memory
    JobData<FloatType>* d_job;
    FloatType* d_fld;
    FloatType* d_swp;
    CheckErrors(cudaMalloc(&d_job, sizeof(JobData<FloatType>)));
    CheckErrors(cudaMalloc(&d_fld, field_bytes));
    CheckErrors(cudaMalloc(&d_swp, field_bytes));

    // Set device pointers in job description
    job.d_fld = d_fld;
    job.d_swp = d_swp;

    // Copy data to device
    CheckErrors(cudaMemcpy(d_job, &job, sizeof(JobData<FloatType>), cudaMemcpyHostToDevice));
    CheckErrors(cudaMemcpy(d_fld, field_ptr, field_bytes, cudaMemcpyHostToDevice));

    // Create user stream
    cudaStream_t stream;
    CheckErrors(cudaStreamCreate(&stream));

    // Create timing events
    cudaEvent_t start;
    cudaEvent_t stop;
    CheckErrors(cudaEventCreate(&start));
    CheckErrors(cudaEventCreate(&stop));

    // Run kernel multiple times and measure best time
    float time_ms = 1e99;
    for (int r = 0; r < runs; ++r) {

        // Run kernel
        CheckErrors(cudaProfilerStart());
        CheckErrors(cudaEventRecord(start));
        stencil3d_launch(stream, job, d_job, noshared, occ);
        CheckErrors(cudaEventRecord(stop));
        CheckErrors(cudaDeviceSynchronize());
        CheckErrors(cudaProfilerStop());

        // Get elapsed time
        float run_ms = 0;
        CheckErrors(cudaEventElapsedTime(&run_ms, start, stop));
        if (run_ms < time_ms)
            time_ms = run_ms;
    }

    // Copy data to host
    // Note: pointers in job description may have been changed by swapping mechanism
    CheckErrors(cudaMemcpy(&job, d_job, sizeof(JobData<FloatType>), cudaMemcpyDeviceToHost));
    CheckErrors(cudaMemcpy(field_ptr, job.d_fld, field_bytes, cudaMemcpyDeviceToHost));

    // Free device objects
    CheckErrors(cudaFree(d_fld));
    CheckErrors(cudaFree(d_swp));
    CheckErrors(cudaFree(d_job));
    CheckErrors(cudaStreamDestroy(stream));
    CheckErrors(cudaEventDestroy(start));
    CheckErrors(cudaEventDestroy(stop));

    // Write field to file
    field.writeToFile(filename);

    // Return best time
    return time_ms;
}
