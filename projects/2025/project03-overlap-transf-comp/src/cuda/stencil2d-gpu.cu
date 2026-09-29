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

// Apply diffusion using GPU kernels with sequential z-level processing
// PRE: inField and outField are initialized Storage3D objects with dimensions x*y*z and halo width 'halo'
//      alpha is the diffusion coefficient (typically 1/32)
//      numIter is the number of diffusion iterations to perform
//      x, y, z are the interior dimensions (excluding halo)
//      halo is the width of the halo region (must be >= 2 for the stencil)
// POST: outField contains the result after numIter diffusion steps
//       Device memory is allocated and freed automatically
//       Each iteration applies: out = in - alpha * laplacian(laplacian(in))
//       Processes one z-level at a time sequentially
void apply_diffusion_gpu(Storage3D<double> &inField, Storage3D<double> &outField,
                        double alpha, unsigned numIter, int x, int y, int z,
                        int halo) {

    // Allocate device memory for input and output fields
    inField.allocateDevice();
    outField.allocateDevice();
    
    // Create temporary field for GPU computation - stores intermediate Laplacian
    Storage3D<double> tmp1Field(x, y, z, halo); 
    tmp1Field.allocateDevice();
    
    // Copy input data from host to device
    inField.copyToDevice();
    
    // Check for CUDA errors after setup
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA setup error: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    // Set up CUDA execution configuration for 2D diffusion kernels
    dim3 blockSize(16, 16);  // 2D thread block for xy-plane processing
    dim3 gridSize((x + halo * 2 + blockSize.x - 1) / blockSize.x,
                  (y + halo * 2 + blockSize.y - 1) / blockSize.y);
    
    // Debug output for kernel configuration
    std::cout << "Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
    std::cout << "Block size: " << blockSize.x << "x" << blockSize.y << std::endl;
    
    // Unused 3D configuration (kept for reference)
    dim3 haloBlockSize(16, 16, 1);
    dim3 haloGridSize((inField.xSize() + haloBlockSize.x - 1) / haloBlockSize.x,
                     (inField.ySize() + haloBlockSize.y - 1) / haloBlockSize.y,
                     (z + haloBlockSize.z - 1) / haloBlockSize.z);
    
    // Calculate total halo points for 1D halo update kernel
    int xInterior = x;
    int haloPointsPerZ = 2 * xInterior * halo + 2 * (y + 2 * halo) * halo;
    int totalHaloPoints = haloPointsPerZ * z;
    
    // 1D thread configuration for halo update kernel
    int haloThreadsPerBlock = 256;  // Optimized block size determined by testing
    int haloBlocks = (totalHaloPoints + haloThreadsPerBlock - 1) / haloThreadsPerBlock;
    
    // Main iteration loop
    for (unsigned iter = 0; iter < numIter; ++iter) {
        // Update halo regions with periodic boundary conditions using GPU kernel
        updateHaloKernel<<<haloBlocks, haloThreadsPerBlock>>>(
            inField.deviceData(), inField.xSize(), inField.ySize(), inField.zMax(), halo
        );
                
        cudaDeviceSynchronize(); // Ensure halo update completes before computation
        
        // Process each z-level sequentially using diffusion kernel
        for (int k = 0; k < z; ++k) {
            diffusionStepKernel<<<gridSize, blockSize>>>(
                inField.deviceData(), outField.deviceData(), tmp1Field.deviceData(),
                inField.xSize(), inField.ySize(), inField.zMax(), k, halo, alpha
            );
        }

        cudaDeviceSynchronize(); // Ensure all diffusion steps complete
        
        // If not the last iteration, copy output back to input for next iteration
        if (iter < numIter - 1) {
            cudaMemcpy(inField.deviceData(), outField.deviceData(), 
                      inField.size() * sizeof(double), cudaMemcpyDeviceToDevice);
        }
    }
    
    // Copy final result from device back to host
    outField.copyFromDevice();
}

// Report timing results in a format suitable for analysis
// PRE: storage is a valid Storage3D object
//      nIter is the number of iterations performed
//      diff is the elapsed time in seconds
// POST: Outputs timing data to stdout in CSV format
//       Format: <size>, <nx>, <ny>, <nz>, <num_iter>, <time>
void reportTime(const Storage3D<double> &storage, int nIter, double diff) {
    std::cout << "ranks nx ny nz num_iter time\n";
    int size = 1; // Assuming single GPU
    std::cout << size << ", " << storage.xMax() - storage.xMin() << ", "
              << storage.yMax() - storage.yMin() << ", " << storage.zMax() << ", "
              << nIter << ", " << diff << "\n";
}

// Main function: Parse command line arguments and run GPU diffusion simulation
// Expected arguments: program -nx <x> -ny <y> -nz <z> -iter <iterations>
// PRE: argc == 9 and argv contains valid integer arguments for dimensions and iterations
// POST: Writes input field to "in_field.dat", runs GPU diffusion simulation,
//       writes output field to "out_field.dat", and reports timing results
int main(int argc, char const *argv[]) {
#ifdef CRAYPAT
    PAT_record(PAT_STATE_OFF);
#endif
    // Validate command line arguments
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " -nx <x> -ny <y> -nz <z> -iter <iterations>" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    int x = atoi(argv[2]);     // x dimension
    int y = atoi(argv[4]);     // y dimension
    int z = atoi(argv[6]);     // z dimension
    int iter = atoi(argv[8]);  // number of iterations
    int nHalo = 3;             // halo width (fixed at 3)
    
    // Validate input parameters
    assert(x > 0 && y > 0 && z > 0 && iter > 0);
    
    // Initialize input and output fields
    Storage3D<double> input(x, y, z, nHalo);
    input.initialize();
    Storage3D<double> output(x, y, z, nHalo);
    output.initialize();

    double alpha = 1. / 32.;  // Diffusion coefficient

    // Write initial field to file for reference
    std::ofstream fout;
    fout.open("in_field.dat", std::ios::binary | std::ofstream::trunc);
    input.writeFile(fout);
    fout.close();

#ifdef CRAYPAT
    PAT_record(PAT_STATE_ON);
#endif
    // Time the GPU diffusion computation
    auto start = std::chrono::steady_clock::now();

    apply_diffusion_gpu(input, output, alpha, iter, x, y, z, nHalo);

    auto end = std::chrono::steady_clock::now();
#ifdef CRAYPAT
    PAT_record(PAT_STATE_OFF);
#endif

    // Update halo regions on the final output field using GPU kernel
    // (Alternative to using updateHalo(output) from utils.h)
    int xInterior = x;
    int haloPointsPerZ = 2 * xInterior * nHalo + 2 * (y + 2 * nHalo) * nHalo;
    int totalHaloPoints = haloPointsPerZ * z;
    
    // 1D thread configuration for final halo update
    int haloThreadsPerBlock = 256;  // Best choice determined by testing
    int haloBlocks = (totalHaloPoints + haloThreadsPerBlock - 1) / haloThreadsPerBlock;
    
    // Update halo on output field using GPU kernel
    updateHaloKernel<<<haloBlocks, haloThreadsPerBlock>>>(
        output.deviceData(), output.xSize(), output.ySize(), output.zMax(), nHalo
    );
    cudaDeviceSynchronize(); // Ensure halo update completes
    
    // Copy final result with updated halos back to host and write to file
    output.copyFromDevice();
    fout.open("out_field.dat", std::ios::binary | std::ofstream::trunc);
    output.writeFile(fout);
    fout.close();

    // Calculate and report timing results
    auto diff = end - start;
    double timeDiff = std::chrono::duration<double, std::milli>(diff).count() / 1000.;
    reportTime(output, iter, timeDiff);

    return 0;
}