#include "utils.h"

// Number of GPU threads
#ifndef THREADS
#define THREADS 16
#endif

// Extra debug output
#ifndef DEBUG
#define DEBUG 0
#endif

// Perform multiple timing measurements
#ifndef ACCURATE_TIMING
#define ACCURATE_TIMING 1
#endif

// Warmup timing measurements to discard
#ifndef WARMUP_REPEATS
#define WARMUP_REPEATS 2
#endif

// Timing measurements to save
#ifndef TIMING_REPEATS
#define TIMING_REPEATS 3
#endif

#define size(x) ( sizeof(x) / sizeof(x[0]) )

__constant__ grid_info gpu_grid;


// Identify CUDA errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
	fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
    }
}



// Simple CUDA timer
typedef struct cuda_timer_struct {
    cudaEvent_t start, stop;
} cuda_timer;

void init_cuda_timer(cuda_timer *timer) {
    gpuErrchk(cudaEventCreate(&(timer->start)));
    gpuErrchk(cudaEventCreate(&(timer->stop)));
}

void start_cuda_timer(cuda_timer *timer) {
    gpuErrchk(cudaEventRecord(timer->start));
}

float stop_cuda_timer(cuda_timer *timer) {
    gpuErrchk(cudaEventRecord(timer->stop));

    float elapsed;
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaEventElapsedTime(&elapsed, timer->start, timer->stop));
    return elapsed;
}

void destroy_cuda_timer(cuda_timer *timer) {
    gpuErrchk(cudaEventDestroy(timer->start));
    gpuErrchk(cudaEventDestroy(timer->stop));
}



// Horizontal halo exchange
__global__ void halo_horizontal(float *field) {
    unsigned int w = gpu_grid.nx;
    unsigned int h = gpu_grid.ny;
    unsigned int nh = gpu_grid.num_halo;

    unsigned int x = threadIdx.x;
    unsigned int y = blockIdx.y + nh;
    unsigned int z = blockIdx.z;

    field[A3D(x, y, z, w + 2 * nh, h + 2 * nh)] = field[A3D(x + w, y, z, w + 2 * nh, h + 2 * nh)];
    field[A3D(x + nh + w, y, z, w + 2 * nh, h + 2 * nh)] = field[A3D(x + nh, y, z, w + 2 * nh, h + 2 * nh)];
}



// Vertical halo exchange
__global__ void halo_vertical(float *field) {
    unsigned int w = gpu_grid.nx;
    unsigned int h = gpu_grid.ny;
    unsigned int nh = gpu_grid.num_halo;

    unsigned int x = blockIdx.x;
    unsigned int y = threadIdx.y;
    unsigned int z = blockIdx.z;

    field[A3D(x, y, z, w + 2 * nh, h + 2 * nh)] = field[A3D(x, y + h, z, w + 2 * nh, h + 2 * nh)];
    field[A3D(x, y + nh + h, z, w + 2 * nh, h + 2 * nh)] = field[A3D(x, y + nh, z, w + 2 * nh, h + 2 * nh)];
}



// First Laplace computation
__global__ void laplace(float *gpu_field, float *tmp_field) {
    unsigned int w = gpu_grid.nx;
    unsigned int h = gpu_grid.ny;
    unsigned int nh = gpu_grid.num_halo;

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x + nh - 1;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y + nh - 1;
    unsigned int z = blockIdx.z;

    // Check bounds
    if (x >= nh - 1 && x < nh + w + 1 &&
	y >= nh - 1 && y < nh + h + 1) {
	tmp_field[A3D(x, y, z, w + 2 * nh, h + 2 * nh)] = -4.0 * gpu_field[A3D(x, y, z, w + 2 * nh, h + 2 * nh)]
	    + gpu_field[A3D(x-1, y, z, w + 2 * nh, h + 2 * nh)]
	    + gpu_field[A3D(x+1, y, z, w + 2 * nh, h + 2 * nh)]
	    + gpu_field[A3D(x, y-1, z, w + 2 * nh, h + 2 * nh)]
	    + gpu_field[A3D(x, y+1, z, w + 2 * nh, h + 2 * nh)];
    }
}



// Second Laplace computation and Euler step
__global__ void finalize(float *gpu_field, float *tmp_field, float alpha) {
    unsigned int w = gpu_grid.nx;
    unsigned int h = gpu_grid.ny;
    unsigned int nh = gpu_grid.num_halo;

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x + nh;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y + nh;
    unsigned int z = blockIdx.z;

    if (x >= nh && x < nh + w &&
	y >= nh && y < nh + h) {
	float laplap = -4.0 * tmp_field[A3D(x, y, z, w + 2 * nh, h + 2 * nh)]
	    + tmp_field[A3D(x-1, y, z, w + 2 * nh, h + 2 * nh)]
	    + tmp_field[A3D(x+1, y, z, w + 2 * nh, h + 2 * nh)]
	    + tmp_field[A3D(x, y-1, z, w + 2 * nh, h + 2 * nh)]
	    + tmp_field[A3D(x, y+1, z, w + 2 * nh, h + 2 * nh)];

	gpu_field[A3D(x, y, z, w + 2 * nh, h + 2 * nh)] -= alpha * laplap;
    }
}



// Stencil computation main loop
void stencil(float *gpu_field, float *tmp_field,
	     float alpha, float num_iter, grid_info grid) {
    unsigned int w = grid.nx;
    unsigned int h = grid.ny;
    unsigned int d = grid.nz;
    unsigned int nh = grid.num_halo;

    dim3 block_grid_1 = dim3((w + 2 + THREADS - 1) / THREADS, (h + 2 + THREADS - 1) / THREADS, d);
    dim3 thread_grid_1 = dim3(THREADS, THREADS, 1);

    dim3 block_grid_2 = dim3((w + THREADS - 1) / THREADS, (h + THREADS - 1) / THREADS, d);
    dim3 thread_grid_2 = dim3(THREADS, THREADS, 1);

    dim3 block_grid_h = dim3(1, h, d);
    dim3 thread_grid_h = dim3(nh, 1, 1);

    dim3 block_grid_v = dim3(w + 2 * nh, 1, d);
    dim3 thread_grid_v = dim3(1, nh, 1);

    halo_horizontal<<<block_grid_h, thread_grid_h>>>(gpu_field);
    halo_vertical<<<block_grid_v, thread_grid_v>>>(gpu_field);
    for (unsigned int iter = 0; iter < num_iter; ++iter) {
	laplace<<<block_grid_1, thread_grid_1>>>(gpu_field, tmp_field);
	finalize<<<block_grid_2, thread_grid_2>>>(gpu_field, tmp_field, alpha);
	halo_horizontal<<<block_grid_h, thread_grid_h>>>(gpu_field);
	halo_vertical<<<block_grid_v, thread_grid_v>>>(gpu_field);
    }
}



// Fill array with initial condition for stencil computation
__global__ void field_init (float *in_field) {
    unsigned int w = gpu_grid.nx;
    unsigned int h = gpu_grid.ny;
    unsigned int d = gpu_grid.nz;
    unsigned int nh = gpu_grid.num_halo;

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int z = blockIdx.z + d / 4;

    if (x >= nh + w / 4 && x < nh + 3 * w / 4 &&
	y >= nh + h / 4 && y < nh + 3 * h / 4) {
	in_field[A3D(x, y, z, w + 2 * nh, h + 2 * nh)] = 1.0;
    } else if (x < w + 2 * nh && y < h + 2 * nh) {
	in_field[A3D(x, y, z, w + 2 * nh, h + 2 * nh)] = 0.0;
    }
}




// Allocate and initialize arrays for stencil computation
void setup (float **in_field, float **out_field, float **tmp_field, float **gpu_field,
		grid_info *gpu_grid, grid_info *grid) {
    unsigned int w = grid->nx;
    unsigned int h = grid->ny;
    unsigned int d = grid->nz;
    unsigned int nh = grid->num_halo;

    // CPU allocations
    *in_field = (float *) malloc((w + 2 * nh) * (h + 2 * nh) * d * sizeof(float));
    *out_field = (float *) malloc((w + 2 * nh) * (h + 2 * nh) * d * sizeof(float));

    // GPU allocations
    gpuErrchk(cudaMalloc((void **) gpu_field, (w + 2 * nh) * (h + 2 * nh) * d * sizeof(float)));
    gpuErrchk(cudaMalloc((void **) tmp_field, (w + 2 * nh) * (h + 2 * nh) * d * sizeof(float)));

    // Initialize constant memory grid info
    gpuErrchk(cudaMemcpyToSymbol(*gpu_grid, grid, sizeof(grid_info)));

    // Create initial condition
    dim3 block_grid((w + 2 * nh + THREADS - 1) / THREADS, (h + 2 + THREADS - 1) / THREADS, 3 * d / 4 - d / 4);
    dim3 thread_grid(THREADS, THREADS, 1);
    field_init<<<block_grid, thread_grid>>>(*gpu_field);

    // Copy initial condition to CPU for output
    gpuErrchk(cudaMemcpy(*in_field, *gpu_field,
			 (w + 2 * nh) * (h + 2 * nh) * d * sizeof(float),
			 cudaMemcpyDeviceToHost));
}



// Destroy arrays for stencil computation
void cleanup (float **in_field, float **out_field, float **tmp_field, float **gpu_field) {
    free(*in_field);
    free(*out_field);

    *in_field = NULL;
    *out_field = NULL;

    gpuErrchk(cudaFree(*gpu_field));
    gpuErrchk(cudaFree(*tmp_field));

    *gpu_field = NULL;
    *tmp_field = NULL;
}




int main (int argc, char *argv[]) {
    float *in_field;
    float *out_field;

    float alpha = 1.0 / 32.0;
    grid_info grid;
    grid.num_halo = 2;
    unsigned int num_iter;
    int scan;

    float *gpu_field, *tmp_field;

    unsigned int num_setups = 1;
    unsigned int nx_setups[] = {12, 28, 44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252, 268, 284, 300};
    unsigned int ny_setups[] = {12, 28, 44, 60, 76, 92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252, 268, 284, 300};

    // Initialize values
    read_cmd_line_arguments(argc, argv, &grid, &num_iter, &scan);

    // Output numpy compatible array
    printf("# ranks nx ny nz num_iter time\n");
    printf("data = np.array( [ \\\n");
    fflush(stdout);

    if (scan) {
	num_setups = size(nx_setups) * size(ny_setups);
    }

    for (unsigned int cur_setup = 0; cur_setup < num_setups; ++cur_setup) {
	if (scan) {
	    grid.nx = nx_setups[cur_setup % size(ny_setups)];
	    grid.ny = ny_setups[cur_setup / size(ny_setups)];
	}

	// Initialize arrays
	setup(&in_field, &out_field, &tmp_field, &gpu_field,
	      &gpu_grid, &grid);

	if (!scan) {
	    write_field_to_file(in_field, &grid, "in_field.dat");
	}

#if DEBUG == 1
	printf("Launching stencil.\n");
	fflush(stdout);
#endif

	// Warmup timing measurements
#if ACCURATE_TIMING == 1 && WARMUP_REPEATS > 0
	for (unsigned int i = 0; i < WARMUP_REPEATS; ++i) {
	    stencil(gpu_field, tmp_field,
		    alpha, num_iter, grid);
	}
#endif

	float time = 0;
	cuda_timer timer;
	init_cuda_timer(&timer);

	// Timing measurements
#if ACCURATE_TIMING == 1
	float msec_data[TIMING_REPEATS];

	for (unsigned int i = 0; i < TIMING_REPEATS; ++i) {
#endif
	    start_cuda_timer(&timer);

	    stencil(gpu_field, tmp_field,
		    alpha, num_iter, grid);

	    time = stop_cuda_timer(&timer);

#if ACCURATE_TIMING == 1
	    msec_data[i] = time;
	}

	write_timing_to_file(msec_data, TIMING_REPEATS, "timing.dat");
#endif
	destroy_cuda_timer(&timer);

	// Find minimum time
#if ACCURATE_TIMING == 1
	for (unsigned int i = 0; i < TIMING_REPEATS; ++i) {
	    if(msec_data[i] < time) {
		time = msec_data[i];
	    }
	}
#endif

#if DEBUG == 1
	printf("Kernel executed in %f msec.\n", time);
	fflush(stdout);
#endif

	// Repeat stencil computation from scratch for accurate output
	dim3 block_grid((grid.nx + 2 * grid.num_halo + THREADS - 1) / THREADS,
			(grid.ny + 2 + THREADS - 1) / THREADS,
			3 * grid.nz / 4 - grid.nz / 4);
	dim3 thread_grid(THREADS, THREADS, 1);
	field_init<<<block_grid, thread_grid>>>(gpu_field);
	stencil(gpu_field, tmp_field,
		alpha, num_iter, grid);

	// Copy stencil output to host
	gpuErrchk(cudaMemcpy(out_field, gpu_field,
			     (grid.nx + 2 * grid.num_halo) *
			     (grid.ny + 2 * grid.num_halo) *
			     grid.nz * sizeof(float), cudaMemcpyDeviceToHost));

	// Write result to disk
	if (!scan) {
	    write_field_to_file(out_field, &grid, "out_field.dat");
	}

	cleanup(&in_field, &out_field, &tmp_field, &gpu_field);

	// Output info
	printf("[%d, %d, %d, %d, %E], \\\n", grid.nx, grid.ny, grid.nz, num_iter, time);
	fflush(stdout);
    }

    // Finish numpy compatible array
    printf("] )\n");

    return 0;
}
