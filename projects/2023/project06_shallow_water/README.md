# Project 6: Shallow water equations on a torus
By Killian P. Brennan, Dana Grund, Joren Janzing, and Franco Lee
Project for the course: High Performance Computing for Weather and Climate.
21.08.2023

During this project we implemented the shallow water equations on a toroidal planet and converted the code to GT4Py.
This folder contains the used scripts in our project. 

**sw_gt4py**:  
The main folder of interest. The code can be run with `driver.py`. 
In this file, you can select if you want to run the code in numpy on a sphere (`swes_numpy.py`) or on a torus (`swes_numpy_toroidal.py`).
You can also select that you want to run the code in gt4py on a sphere  (`swes_gt4py.py`) or on a torus (`swes_gt4py_toroidal.py`).

The gt4py functions can be found in `gt4py_functions.py`.
For performance analysis, there are two files: `preformance_analysis.py` and `backend_preformance_analysis.py`.
The data and plots for the performance analysis can be found in the subfolders **performance_data** and **performance_plots**, respectively.

**sw_numpy** and **toroidal_physics**
The files in these folders were used to convert spherical code was converted to a torus.

**visualization**
Here we worked on visualizing the flow on the torus.