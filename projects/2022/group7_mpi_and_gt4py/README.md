# hpc4wc_project

*Task:*

MPI version of GT4Py code: Merge your work from MPI parallelization into the gt4py version and make sure that all code is ported using gt4py (including halo-updates). This code will be able to fully leverage the Piz Daint supercomputer, wohoo! Analyze and optimize its performance and do a weak and strong scaling investigation on Piz Daint for different backends.

*Content*:

"stencil2d.py" and "partitioner.py" from the blockcourse were used as basis to start the project from. "stencil2d-gt4py-a1.py" is the first version, where the halo updates were made by indexing and copying arrays. "stencil2d-gt4py-a4.py" is the version leveraging gt4py. Both versions where validated with the basecode "stencil2d.py" in the notebook "Validation.ipynb"; the script and the results are found in the folder "Validation". The codes where run with the shell scrips and the corresponding outputs written to a ".txt" file. The notebook "Evaluation_and_Plotting.ipynb" was used to evaluate the output files and do the plotting for the weak and strong scaling analysis. The plots can be found in the folder "Plots". Further details can be found in the report. 
