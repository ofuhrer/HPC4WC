This readme provides an overview over the different scripts used in this project. Further comments and explanations can be found in the respective files. For unanswered questions feel free to contact plonerp@student.ethz.ch

- "utils.h" contains the Storage3D class, which is used to store three dimensional fields including their halos. It is the same as the class we used in the c++ versions of the day 2 exercises.
- "partitioner.cpp" contains a c++ version of the partitioner class. It is used for scattering and gathering a global field on a root node to and from a given number of nodes using MPI. This class was built from the ground up over the scope of this project.
- "test_partitioner.cpp" is a debugging script for the partitioner class, provided for completeness.
- "stencil_1node.cpp" contains a linear version of the diffusion stencil. It is taken from an exercise on day2.
- "stencil_nnode.cpp" contains the MPI parallelized version of the diffusion stencil and was written as part of this project.
- "runMeasurementSeries.py" is used to perform the measurement series. The current scope of variables requires 9 nodes to run the whole series.
- "verifyResults.py" uses the "compare_fields.py" script from day3 of the exercises to compare the created fields with a reference field created with the non-parallel stencil_1node code.
- "test_notebook.ipynb" allows for further analysis of the differences between parallelized and reference fields.
- "output_folder" contains the measured halo update performances and and the created output fields.
- "data_analysis.ipynb" contains the data analysis script used for reading out and plotting the measured performance data.

In order to reproduce our halo update performance measurements, one just needs to run the runMeasurementSeries.py script. The outputs will then be stored in the output_folder and overwrite current outputs in those folders.