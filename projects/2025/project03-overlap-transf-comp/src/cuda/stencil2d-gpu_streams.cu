#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

#ifdef CRAYPAT
#include "pat_api.h"
#endif
#include "stencil2d_helper/utils.h"
#include "stencil2d_helper/stencil_kernels.cuh"

// Apply diffusion using GPU kernels with multiple CUDA streams for z-level parallelism
// PRE: inField and outField are initialized Storage3D objects with dimensions x*y*z and halo width 'halo'
//      alpha is the diffusion coefficient
//      numIter is the number of diffusion iterations to perform
//      x, y, z are the interior dimensions (excluding halo)
//      halo is the width of the halo region (must be >= 2 for the stencil)
//      numStreams is the number of CUDA streams to use for parallel z-level processing
// POST: outField contains the result after numIter diffusion steps
//       Device memory is allocated and freed automatically
//       Each iteration applies: out = in - alpha * laplacian(laplacian(in))
//       Work is distributed across z-levels using multiple streams
void apply_diffusion_gpu_streams(Storage3D<double> &inField, Storage3D<double> &outField,
                               double alpha, unsigned numIter, int x, int y, int z,
                               int halo, int numStreams = 2) {
    
    // Allocate device memory for input and output fields
    inField.allocateDevice();
    outField.allocateDevice();
    
    // Create temporary field for GPU computation (stores intermediate Laplacian)
    Storage3D<double> tmp1Field(x, y, z, halo); 
    tmp1Field.allocateDevice();
    
    // Create CUDA streams for parallel execution
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Copy input data from host to device (initial setup)
    inField.copyToDevice();
    
    // Calculate work distribution: divide z-levels among streams
    int zPerStream = (z + numStreams - 1) / numStreams;
    
    // Set up CUDA execution configuration for diffusion kernels
    dim3 blockSize(16, 16);  // 2D thread block for xy-plane processing
    dim3 gridSize((x + halo * 2 + blockSize.x - 1) / blockSize.x,
                  (y + halo * 2 + blockSize.y - 1) / blockSize.y);
    
    // Halo update configuration (unused 3D configuration kept for reference)
    dim3 haloBlockSize(16, 16, 1);
    dim3 haloGridSize((inField.xSize() + haloBlockSize.x - 1) / haloBlockSize.x,
                     (inField.ySize() + haloBlockSize.y - 1) / haloBlockSize.y,
                     (z + haloBlockSize.z - 1) / haloBlockSize.z);

    // Calculate total halo points for 1D kernel launch
    int xInterior = x;
    int haloPointsPerZ = 2 * xInterior * halo + 2 * (y + 2 * halo) * halo;
    int totalHaloPoints = haloPointsPerZ * z;
    
    // 1D thread configuration for halo update kernel
    int haloThreadsPerBlock = 256;  // Optimized block size for halo updates
    int haloBlocks = (totalHaloPoints + haloThreadsPerBlock - 1) / haloThreadsPerBlock;
    
    // Main iteration loop
    for (unsigned iter = 0; iter < numIter; ++iter) {
        // Update halo regions with periodic boundary conditions on stream 0
        updateHaloKernel<<<haloBlocks, haloThreadsPerBlock, 0, streams[0]>>>(
            inField.deviceData(), inField.xSize(), inField.ySize(), inField.zMax(), halo
        );
        
        // Wait for halo update to complete before starting computation
        cudaStreamSynchronize(streams[0]);
        
        // Launch diffusion kernels on multiple streams (one z-level per kernel)
        for (int streamId = 0; streamId < numStreams; ++streamId) {
            int startK = streamId * zPerStream;
            int endK = std::min(startK + zPerStream, z);
            
            // Process assigned z-levels for this stream
            for (int k = startK; k < endK; ++k) {
                diffusionStepKernel<<<gridSize, blockSize, 0, streams[streamId]>>>(
                    inField.deviceData(), outField.deviceData(), tmp1Field.deviceData(),
                    inField.xSize(), inField.ySize(), inField.zMax(), k, halo, alpha
                );
            }
        }
        
        // Synchronize all streams before next iteration
        for (int i = 0; i < numStreams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        
        // If not the last iteration, copy output back to input for next iteration
        if (iter < numIter - 1) {
            cudaMemcpyAsync(inField.deviceData(), outField.deviceData(), 
                           inField.size() * sizeof(double), cudaMemcpyDeviceToDevice, 
                           streams[0]);
            cudaStreamSynchronize(streams[0]);
        }
    }
    
    // Cleanup: destroy all CUDA streams
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    
    // Copy final result from device back to host
    outField.copyFromDevice();
}

// Report timing results in a format suitable for analysis with stream information
// PRE: storage is a valid Storage3D object
//      nIter is the number of iterations performed
//      diff is the elapsed time in seconds
//      nStreams is the number of CUDA streams used
// POST: Outputs timing data to stdout with stream count
//       Format: ###<size>, <nx>, <ny>, <nz>, <num_iter>, <time>, <num_streams>
void reportTime(const Storage3D<double> &storage, int nIter, double diff, int nStreams = 1) {
    int size = 1; // Assuming single GPU
    std::cout << "###" << size << ", " << storage.xMax() - storage.xMin() << ", "
              << storage.yMax() - storage.yMin() << ", " << storage.zMax() << ", "
              << nIter << ", " << diff << ", " << nStreams << "\n" ;
}

// Main function: Parse command line arguments and run GPU streams diffusion simulation
// Expected arguments: program -nx <x> -ny <y> -nz <z> -iter <iterations> -streams <numStreams> [-test <true|false>]
// PRE: argc >= 11 and argv contains valid integer arguments for dimensions, iterations, and streams
//      Optional test flag controls output file generation and timing repetitions
// POST: Runs diffusion simulation with specified number of streams
//       Reports average timing over multiple repetitions (10 for performance, 1 for testing)
//       Optionally writes output field to "out_field_streams.dat" if test=true
int main(int argc, char const *argv[]) {
#ifdef CRAYPAT
    PAT_record(PAT_STATE_OFF);
#endif
    // Validate command line arguments
    if (argc < 11) {
        std::cerr << "Usage: " << argv[0] << " -nx <x> -ny <y> -nz <z> -iter <iterations> -streams <numStreams> -test <true|false>" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    int x = atoi(argv[2]);          // x dimension
    int y = atoi(argv[4]);          // y dimension
    int z = atoi(argv[6]);          // z dimension
    int iter = atoi(argv[8]);       // number of iterations
    int numStreams = atoi(argv[10]); // number of CUDA streams
    bool test = false;              // test mode flag
    if (argc == 13) {
        test = (std::string(argv[12]) == "true");
    }
    int nHalo = 3;                  // halo width (fixed at 3)
    
    // Validate input parameters
    assert(x > 0 && y > 0 && z > 0 && iter > 0);
    
    // Initialize input and output fields
    Storage3D<double> input(x, y, z, nHalo);
    // input.initialize();  // Commented out - initialization done per timing run
    Storage3D<double> output(x, y, z, nHalo);
    output.initialize();

    double alpha = 1. / 32.;  // Diffusion coefficient

#ifdef CRAYPAT
    PAT_record(PAT_STATE_ON);
#endif
    double totalTime = 0.0;
    const int TOTAL_REPS = test ? 1 : 10; // Number of repetitions for timing
    
    // Warm up the GPU to ensure consistent timing measurements
    input.initialize();
    apply_diffusion_gpu_streams(input, output, alpha, iter, x, y, z, nHalo, numStreams);
    
    // Run multiple repetitions for accurate timing (unless in test mode)
    for (int rep = 0; rep < TOTAL_REPS; ++rep) {
        input.initialize();
        auto start = std::chrono::steady_clock::now();
        apply_diffusion_gpu_streams(input, output, alpha, iter, x, y, z, nHalo, numStreams);
        auto end = std::chrono::steady_clock::now();
        totalTime += std::chrono::duration<double, std::milli>(end - start).count() / 1000.;
    }
    double avgTime = totalTime / TOTAL_REPS;
#ifdef CRAYPAT
    PAT_record(PAT_STATE_OFF);
#endif
    
    // If in test mode, write output field to file for verification
    if(test) {
        updateHalo(output);
        std::ofstream fout;
        fout.open("out_field_streams.dat", std::ios::binary | std::ofstream::trunc);
        output.writeFile(fout);
        fout.close();
    }

    // Report timing results including stream count
    reportTime(output, iter, avgTime, numStreams);

    return 0;
}