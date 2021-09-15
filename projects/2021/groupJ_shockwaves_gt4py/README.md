# HPC4WC: Simulating Shock Waves Using a Domain-Specific Language (DSL) in Python

Implement the 5th-order weighted essentially non-oscillatory (WENO) scheme (WENO5) for spatial approximation and the 3rd-order 3-stage Strong Stability Preserving Runge Kutta (SSP-RK(3,3)) as time integration using GT4Py to solve Burger's equation.

### Root directory
* ```./src/BurgersWENOdriver.py``` Driver for main program
* ```./src/BurgersWENOcalc.py``` Contains functions required for the main program
* ```./src/BurgersWENOperf.py``` Plot performance from log file
* ```./src/BurgersWENO.py``` Main program
* ```./run``` Run performance tests with different settings
* ```./output/log``` Output to validate the performance
* ```./output/WENO2D_*``` Figure outputs for different GT4Py backends
* ```./requirements.txt``` Python requirements
* ```./report.pdf```
* ```./README.md``` 

### Setup environment
```source HPC4WC_venv/bin/activate ```

### Plot the figure using command line
```python3 BurgersWENOdriver.py --nx=50 --ny=30 --nz=1 --plot=True --backend=gtcuda --time=True --tmp=True```

### Running options
* ```--m```  m(2m-1)th order of ENO(WENO) approximation 
* ```--nx``` Number of gridpoints in x-direction
* ```--ny``` Number of gridpoints in y-direction
* ```--nz``` Number of gridpoints in z-direction 
* ```--gt4py``` Use of GT4Py?
* ```--tmp``` Use of temporary variables?
* ```--backend``` NumPy or GT4Py backends?
* ```--plot``` Make a plot of the result?
* ```--time``` Write the elapsed time?

### Write the performance
```bash run``` 


© Ruoyi Cui, Shuchang Liu, Shihao Zeng