#include <cuda_runtime.h>
#include <chrono>
#include <random>

// Number of threads used per block on GPU
#define THREADS_PER_BLOCK 256

// RNG seed for random number generation
#define SEED 42

// Define a tolerance for floating-point comparison
#define TOL 1e-5


// Define the floating-point type
using fType = float;


// Clamp function to restrict a value to a given range
// PRE: x is the value to clamp, lower is the lower bound, upper is the upper bound
// POST: Returns x clamped to the range [lower, upper]
__device__ __host__ inline float clampf(float x, float lower, float upper);


// CUDA kernel that computes the arccos of the element at the threads index
// PRE: d_data is allocated on the device and size is the number of elements in d_data
// POST: d_data contains the result of num_arccos_calls chained arccos operations
//       for the number of threads it is launched with
__global__ void compute_kernel_multiple(fType* d_data, int size, int num_arcos_calls);


// Initialize all host and device arrays and the streams
// PRE: h_data, h_result, h_reference, d_data and streams do not point to valid memory
// POST: h_data points to an aray of size "num_streams" of pinned memory of size "bytes" on the host
//       filled with random values in the range [-1, 1].
//       h_result points to an aray of size "num_streams" of uninitialized pinned memory of size 
//       "bytes" on the host.
//       h_refernce points to an aray of size "num_streams" of pinned memory of size "bytes" on the 
//       host filled with the expected result of 
//       "num_arccos_calls" chained arccos operations.
//       d_data points to an aray of size "num_streams" of uninitialized allocated device memory of size "bytes".
//       streams points to an aray of size "num_streams" of initialized CUDA streams.
//       Returns 0 on success, 1 on failure
int init_data(fType* h_data[], fType* h_result[], fType* h_reference[], fType* d_data[], size_t bytes, cudaStream_t streams[], int num_arccos_calls, int num_streams, int size_per_stream);


// Function to initialize host data and reference result
// PRE: h_data and h_result point to arrays of valid memory of size "chunksize".
//      gen is a valid random number generator and dis is a valid uniform distribution.
// POST: h_data is filled with random values in the range [-1, 1]
//       and h_result is filled with the expected result of "num_arccos_calls" consecutive arccos operations
//       on the values in h_data.
void init_h_local(fType* h_data, fType* h_reference, int num_arccos_calls, int chunksize, std::mt19937 &gen, std::uniform_real_distribution<fType> &dis);


// Run the arccos computation using multiple CUDA streams
// PRE: h_data, h_result and h_data point to arrays of size "num_streams" of valid memory of size "size_per_stream".
//      h_data and h_result must be pinned memory.
//      d_data points to an array of size "num_streams" of valid device memory of size "size_per_stream".
//      streams points to an array of size "num_streams" of valid CUDA streams.
//      duration is a valid duration object to store the time taken for the computation.
// POST: h_result contains the result of the arccos computation.
//       duration contains the time it took to perform the arrccos computations on the whole array, distributed to the streams.
//       Returns 0 if no error occured, otherwise returns 1.
int run_arccos(int num_arccos_calls, int size_per_stream, int num_streams, std::chrono::duration<double> &duration, fType* h_data[], fType* h_result[], fType* d_data[], cudaStream_t streams[]);


// Run stream operations
// PRE: h_data and h_result point to arrays of size "num_streams" of valid pinned memory of size "size_per_stream".
//      d_data points to an array of size "num_streams" of valid device memory of size "size_per_stream".
//      streams points to an array of size "num_streams" of valid CUDA streams.
// POST: h_result contains the result of the arccos computation.
//       Returns cudaSuccess on success, otherwise returns an error code.
cudaError_t run_stream_operations(fType* h_data[], fType* h_result[], fType* d_data[], cudaStream_t streams[], int num_arccos_calls, int size_per_stream, int num_streams,
                                     int threads, int blocks);


// Validate the result of the arccos computation
// PRE: h_reference and h_result point to arrays of size "num_streams" of valid memory of size "size_per_stream".
// POST: Returns 0 if all results are correct, otherwise returns 1.
int validate_result(fType* h_reference[], fType* h_result[], int size_per_stream, int num_streams);


// Delete all allocated memory and destroy all streams
// PRE: h_data, h_result and h_refernce point to arrays of size "num_streams" of valid pinned memory.
//      d_data points to an array of size "num_streams" of valid device memory.
//      streams points to an array of size "num_streams" of valid CUDA streams.
// POST: All allocated memory is freed and all streams are destroyed.
void cleanup(fType* h_data[], fType* h_result[], fType* h_refernce[], fType* d_data[], cudaStream_t streams[], int num_streams);