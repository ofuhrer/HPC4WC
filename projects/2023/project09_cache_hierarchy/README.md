# HPC4WC Project 09 Cache Hierarchy

This is the source code of the cache hierarchy project for the course *701-1270-00L High Performance Computing for Weather and Climate* at ETH Zurich.

## Build

### Requirements

- Unix-like operating system
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)

### Compilation

```bash
mkdir build
cd build
cmake ..
cmake --build . -j
```

## Test

### Test stencils

```bash
cd build
./test/stencil_test
```

### Test cache simulator

```bash
cd build
./test/cache_simulator_test
```

## Run

### Measure runtime

```bash
cd build
./src/cache_hierarchy
```

### Measure cache hit rates

```bash
cd build
./src/cache_hierarchy_hr
```

## Plot

Once an executable has generated output, run the appropriate plotting script.

For example for the runtime, run the following:
```bash
cd scripts
python plot_runtime.py
```
