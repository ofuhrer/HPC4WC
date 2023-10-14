import numpy as np
# srun -n 1 python stencil2d-V03.py --nx 120 --ny 120 --nz 64 --num_iter 1024 --plot_result True
# n_threads = 1
OMP_1 = np.mean(np.array([
  0.1879582E+01,
  0.1872281E+01,
  0.1880370E+01,
  0.1872594E+01,
  0.1876432E+01,
  0.1874941E+01,
  0.1885240E+01,
  0.1871981E+01,
  0.1885557E+01,
  0.1885234E+01]))

# srun -n 1 python stencil2d-V03.py --nx 120 --ny 120 --nz 64 --num_iter 1024 --plot_result True
# n_threads = 4
OMP_4 = np.mean(np.array([
  0.8818104E+00,
  0.8832014E+00,
  0.8873789E+00,
  0.8819685E+00,
  0.8863139E+00,
  0.8828952E+00,
  0.8833463E+00,
  0.8866482E+00,
  0.8811376E+00,
  0.8896797E+00]))

# srun -n 1 python stencil2d-V03.py --nx 120 --ny 120 --nz 64 --num_iter 1024 --plot_result True
# n_threads = 9
OMP_9 = np.mean( np.array([
  0.5215709E+00,
  0.5239401E+00,
  0.5183389E+00,
  0.5199304E+00,
  0.5218239E+00,
  0.5258501E+00,
  0.5265915E+00,
  0.5232286E+00,
  0.5247090E+00,
  0.5242949E+00]))

# srun -n 1 python stencil2d-V03.py --nx 120 --ny 120 --nz 64 --num_iter 1024 --plot_result True
# n_threads = 16
OMP_16 = np.mean(np.array([
  0.4603441E+00,
  0.4572358E+00,
  0.4565263E+00,
  0.4567869E+00,
  0.4584394E+00,
  0.4563448E+00,
  0.4574578E+00,
  0.4560769E+00,
  0.4569824E+00,
  0.4579358E+00]))

# srun -n 1 python stencil2d-V03.py --nx 120 --ny 120 --nz 64 --num_iter 1024 --plot_result True
# n_threads = 25
OMP_25 = np.mean(np.array([
  0.5186660E+00,
  0.5137587E+00,
  0.5263271E+00,
  0.5264308E+00,
  0.5451496E+00,
  0.5552118E+00,
  0.5348547E+00,
  0.5162623E+00,
  0.5584013E+00,
  0.5274127E+00]))
    
# srun -n 1 python stencil2d-base.py --nx 120 --ny 120 --nz 64 --num_iter 1024 --plot_result True

baseline = np.mean(np.array([
  0.7386208E+00,
  0.7542617E+00,
  0.7420247E+00,
  0.7356305E+00,
  0.7340975E+00,
  0.7391365E+00, 
  0.7355399E+00,
  0.7350745E+00,
  0.7331917E+00,
  0.7464905E+00]))
