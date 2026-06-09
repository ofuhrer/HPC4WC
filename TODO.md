# TODO

- [ ] `day4/01-GPU-programming-cupy.ipynb`: Bonus Exercise 15 currently cannot be solved because we do not have access to an OpenACC Fortran compiler on Alps.
- [ ] `day1/01-roofline-model.ipynb`: reconcile the stencil roofline values with the counter-derived values and the Day 1 stencil-program notebook.
- [ ] `day1/01-roofline-model.ipynb`: resolve the DGEMM arithmetic-intensity solution, including the denominator choice and the remaining internal TODO comment.
- [ ] `day1`: make the cache-size explanations consistent across the Day 1 notebooks and with the target Alps hardware.
- [ ] `day2/03-OpenMP-concepts_bonus.ipynb`: align the B8 prompt with the `fully_parallel.cpp` implementation, which uses a parallel reduction rather than a single-thread sum.
- [ ] `day2/02-OpenMP-exercises.ipynb`: complete the J-loop scaling solution explanation after "parallelization being at a different level".
- [ ] `day2/omp_examples/a07-private.cpp`: decide whether the uninitialized `private(myvar)` read is intentional teaching material; if so, explain the undefined private value explicitly in the notebook.
