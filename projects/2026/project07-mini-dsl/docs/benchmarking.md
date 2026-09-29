The goal of the benchmarking infrastructure is to assess the correctness of yasmin code, measure the overhead of the code generation pipeline, and compare the efficiency of the resulting code to hand optimized programs and other DSLs.

To that end, we use two examples. First, the classic 2D (TODO: 3D?) Laplacian stencil. Second, TODO (maybe Burger's equation, like GT4Py?). For both examples we set grid sizes, number of iterations, combinations of cores and threads to test and any other variable. For correctness, we compare the yasmin result to a simple python implementation. We time code generation, compilation (if it applies), and runtime. We compare each backend to different reference implementations:
 - NumPy backend -> NumPy reference implementation, gt4py reference implementation with numpy backend
 - C++ backend -> C++ reference implementation
 - OpenMP backend -> C++ OpenMP reference implementation

## Example 1 - Laplacian

The first benchmark is the standard 2D 5-point Laplacian stencil:

$$
\Delta u_{i,j} = u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_{i,j}
$$

The yasmin implementation uses a single field read and one write, on the interior points only.

### Fixed parameters

- Dimension: 2D, with indices `(x, y)`
- Stencil type: 5-point Laplacian
- Data type: `float64`, `double`
- Boundary conditions: homogeneous Dirichlet, 0 on all outer boundaries
- Initial condition: Sinusoidal field.
- Grid sizes: `128, 256, 512, 1024, 2048, 4096, 8192`
- Total runs per configuration: 5 timed runs, 2 warm-up runs
- Time stepping: single stencil application for the kernel benchmark, plus a short repeated iteration benchmark with `nsteps = 10`
- Compiler/runner settings:
  - `OMP_NUM_THREADS` set to the tested thread count
  - CPU thread counts: 1 through 72


## Example 2 -  two-dimensional viscid Burgers' equations

In this example, the 2D viscid Burgers' equations are solved using a fifth-order upwind advection scheme and a fourth-order centered scheme for diffusion. See https://github.com/GridTools/gt4py/blob/main/examples/cartesian/demo_burgers.ipynb for more details
