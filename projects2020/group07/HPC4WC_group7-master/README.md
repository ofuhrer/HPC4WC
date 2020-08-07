# HPC4WC_group7
Comparison of high-level programming techniques (Numba, GT4Py, Cupy, Numpy) for stencil computation (Coursework HPC4WC 2020)
Numba (http://numba.pydata.org/ 
https://nyu-cds.github.io/python-numba/
https://www.youtube.com/watch?v=x58W9A2lnQc 

## Setup environment:
- open the console
- which python --> prints the current environment
- source HPC4WC_venv/bin/activate --> loads the correct environment
- check again with which python, if it worked

## Run stencil_main_{}.py
- move to its folder and execute directly in console
- or use one of the Jupyternotebooks

#### Validation
- run stencil_main_validation:
python3 stencil_main_validation.py --nx 32 --ny 32 --nz 32 --stencil_name lapoflap1d --backend numba_vector_function --create_field True

- Set the option create_field = True (Default) to create a new random field that is saved as a .npy file.
- Set then the option create_field to False to validate the out fields of different stencils with the original NumPy field.

#### Performance
- run stencil_main_performance:
python3 stencil_main_performance.py --nx 32 --ny 32 --nz 32 --stencil_name lapoflap3d --backend numba_loop --num_iter 20

#### Bandwidth
- run main_bandwidth:
python3 main_bandwidth.py --nx 32 --ny 32 --nz 32 --backend gt4py --num_iter 20


#### Help
For help about available options type: 
python3 stencil_main_{}.py --help

### Project Report
The project report is saved as pdf in this repository under the name "Project_report_group7_HPC4WC.pdf"


  
