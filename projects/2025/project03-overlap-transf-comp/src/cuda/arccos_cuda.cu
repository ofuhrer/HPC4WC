#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include "arccos_cuda.cuh"


__device__ __host__ inline float clampf(float x, float lower, float upper) {
    return fminf(fmaxf(x, lower), upper);
}


__global__ void compute_kernel_multiple(fType* d_data, int size, int num_arcos_calls) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int i = 0; i < num_arcos_calls; ++i) {
            // Map result to the range [-1, 1] after each arccos
            d_data[idx] = clampf( (2 * std::acos(d_data[idx]) / M_PI) - 1.0f, -1.0f, 1.0f );
        }
    }
}
 

int run_arccos(int num_arccos_calls, int size_per_stream, int num_streams, std::chrono::duration<double> &duration, fType* h_data[], fType* h_result[], fType* d_data[], cudaStream_t streams[]) {

    // Make sure there are enough blocks to cover the data
    int threads = THREADS_PER_BLOCK;
    int blocks = (size_per_stream + threads - 1) / threads;

    // Launch operations in streams
    auto start = std::chrono::high_resolution_clock::now();
    cudaError_t err = run_stream_operations(h_data, h_result, d_data, streams, num_arccos_calls, size_per_stream, num_streams, threads, blocks);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    duration = std::chrono::duration<double>(end - start);
    
    if (err != cudaSuccess) {
        std::cerr << "Cuda error after running stream operations: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    return 0;

}

int init_data(fType* h_data[], fType* h_result[], fType* h_reference[], fType* d_data[], size_t bytes, cudaStream_t streams[], int num_arccos_calls, int num_streams, int size_per_stream) {
    
    // Set up RNG
    std::mt19937 gen(SEED);
    std::uniform_real_distribution<fType> dis(-1.0, 1.0);
    
    // Allocate host and device memory, create streams
    for (int i = 0; i < num_streams; ++i) {

        // Allocate pinned host memory such that it can be used with cudaMemcpyAsync
        cudaError_t err = cudaHostAlloc(&h_data[i], bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaHostAlloc failed for h_data[" << i << "]: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        // Allocate pinned host memory such that it can be used with cudaMemcpyAsync
        err = cudaHostAlloc(&h_result[i], bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaHostAlloc failed for h_data[" << i << "]: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        // Allocate pinned host memory such that it can be used with cudaMemcpyAsync
        err = cudaHostAlloc(&h_reference[i], bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaHostAlloc failed for h_reference[" << i << "]: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        // Allocate device memory
        err = cudaMalloc(&d_data[i], bytes);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed for d_data[" << i << "]: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        // Create CUDA stream
        err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            std::cerr << "cudaStreamCreate failed for streams[" << i << "]: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }

        // Initialize host data and reference data
        init_h_local(h_data[i], h_reference[i], num_arccos_calls, size_per_stream, gen, dis);
    }

    // Return 0 on success
    return 0;
}


void init_h_local(fType* h_data, fType* h_reference, int num_arccos_calls, int chunksize, std::mt19937 &gen, std::uniform_real_distribution<fType> &dis) {
    
    // Initialize data with uniform random values in the range [-1, 1]
    for (int j = 0; j < chunksize; ++j) {

        // Generate random value
        h_data[j] = dis(gen);
        
        // Precompute the expected result
        h_reference[j] = clampf( (2 * std::acos(h_data[j]) / M_PI) - 1.0f, -1.0f, 1.0f );
        for (int k = 1; k < num_arccos_calls; ++k) {
            // Map result to the range [-1, 1] after each arccos
            h_reference[j] = clampf( (2 * std::acos(h_reference[j]) / M_PI) - 1.0f, -1.0f, 1.0f );
        }
    }
}


cudaError_t run_stream_operations(fType* h_data[], fType* h_result[], fType* d_data[], cudaStream_t streams[], int num_arccos_calls, int size_per_stream, int num_streams,
                                     int threads, int blocks) {
    // Loop through each stream and perform operations
    for (int i = 0; i < num_streams; ++i) {

        // Copy data from host to device
        cudaError_t err = cudaMemcpyAsync(d_data[i], h_data[i], size_per_stream * sizeof(fType), cudaMemcpyHostToDevice, streams[i]);
        if (err != cudaSuccess) {
            std::cerr << "Memcpy (H2D) failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            return err;
        }

        // Launch the kernel
        compute_kernel_multiple<<<blocks, threads, 0, streams[i]>>>(d_data[i], size_per_stream, num_arccos_calls);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            return err;
        }

        // Copy result from device to host
        err = cudaMemcpyAsync(h_result[i], d_data[i], size_per_stream * sizeof(fType), cudaMemcpyDeviceToHost, streams[i]);
        if (err != cudaSuccess) {
            std::cerr << "Memcpy (D2H) failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            return err;
        }
    }

    // Wait for all streams to finish
    cudaDeviceSynchronize();

    // Check for errors after synchronization
    return cudaGetLastError();
}


int validate_result(fType* h_reference[], fType* h_result[], int size_per_stream, int num_streams) {
    
    // Check if the results match the reference values
    for (int i = 0; i < num_streams; ++i) {
        for (int j = 0; j < size_per_stream; ++j) {
            // Also check for NaN values
            if (std::isnan(h_result[i][j]) || std::isnan(h_reference[i][j])) {
                std::cerr << "NaN detected at index " << j << " in stream " << i << ": "
                          << "Reference: " << h_reference[i][j] << ", Result: " << h_result[i][j] << std::endl;
                // Early exit on first NaN
                return 1;
            }
            // Check if the result matches the reference within a tolerance
            if (std::fabs(h_reference[i][j] - h_result[i][j]) > TOL) {
                std::cerr << "Mismatch at index " << j << " in stream " << i << ": "
                          << h_reference[i][j] << " != " << h_result[i][j] << " with a difference of " << std::fabs(h_reference[i][j] - h_result[i][j]) << std::endl;
                // Early exit on first mismatch
                return 1;
            }
        }
    }
    // All streams verified successfully
    return 0;
}


void cleanup(fType* h_data[], fType* h_result[], fType* h_refernce[], fType* d_data[], cudaStream_t streams[], int num_streams) {
    
    // Free host and device memory, destroy streams
    for (int i = 0; i < num_streams; ++i) {
        cudaFreeHost(h_data[i]);
        cudaFreeHost(h_result[i]);
        cudaFreeHost(h_refernce[i]);
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
    }
}

