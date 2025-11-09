# Project overview
The goal of this project is to compare the performance of various unstructured grid implementations by accessing the horizontal planes of a 3D Cartesian grid through unstructured vectors.  
We implemented three unstructured grid versions - j-stride, Z-order curve, and random grid - each in two ways:
* using a lookup table, and
* using an explicit formula (where applicable).

In addition, we included a Hilbert curve implementation contributed by last year’s group.

# Code structure
## Fortran Files
For performance reasons, we implemented each variant in its own Fortran file:
1. `stencil2d-structured.F90`: Reference version with loop inlining (from Day 1 of the block course). This serves as the performance reference.
2. `stencil2d-jstride_lookup.F90`: Sequential j-stride grid ordering with neighbor access via a lookup table.
3. `stencil2d-jstride_formula.F90`: Sequential j-stride grid ordering with neighbor access via an explicit formula.
4. `stencil2d-zcurve_lookup.F90`: Z-order curve grid ordering with neighbor access via a lookup table.
5. `stencil2d-zcurve_formula.F90`: Z-order curve grid ordering with neighbor access via an explicit formula.
6. `stencil2d-random_lookup.F90`: Random grid ordering with neighbor access via a lookup table.
7. `stencil2d-hilbert.F90`: Hilbert curve grid ordering with neighbor access via a lookup table. (Adopted from last year's implementation.)

## Python Files
We provide one main Python notebook to run, validate, and analyze all versions:
* `launcher.ipynb`
  
  Contains sections for:
    * Result validation against the baseline
    * Performance analysis (single and repeated executions)

## Other Files
* `perf_wrap.sh`: Shell script used for performance analysis via `perf`. Originally from the block course, it was extended to measure flop counts.
* `Makefile`: Originally from the block course, we modified the file to allow inlining.
* `perf/` directory: Stores raw performance outputs (e.g., memory access, FLOP counts, etc.).
* `plots/` directory: Output plots saved as `.png` images. 
