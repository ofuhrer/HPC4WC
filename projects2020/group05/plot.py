import matplotlib.pyplot as plt
import numpy as np

lengths = np.array([32, 128, 512, 2048, 8192, 32768, 131072])
np_avg = np.array([1.284, 1.415, 2.028, 4.080, 12.102, 52.790, 252.910])
np_std = np.array([9.604e-02, 7.587e-02, 9.302e-02, 1.172e-01, 2.248e-01, 2.485e-01, 1.105e+00])
fort_avg = np.array([9.752e-04, 2.578e-03, 1.093e-02, 4.600e-02, 2.102e-01, 8.900e-01, 3.877e+00])
fort_std = np.array([6.433e-04, 6.065e-05, 1.321e-03, 4.048e-03, 1.797e-02, 5.209e-02, 1.457e-01])
x86_avg = np.array([0.090, 0.092, 0.103, 0.147, 0.330, 1.216, 5.303])
x86_std = np.array([7.458e-04, 4.345e-04, 1.491e-03, 5.882e-03, 1.973e-02, 7.084e-02, 6.407e-02])
cuda_avg = np.array([0.097, 0.147])
cuda_std = np.array([1.726e-03, 4.610e-02])
tscale = 2.262/np.sqrt(10) # student-t distirbution double side 95% confidence interval

capsize=1.0
plt.figure()
plt.errorbar(lengths, fort_avg, yerr=2*tscale*fort_std, capsize=capsize, label="Original Fortran")
plt.errorbar(lengths, np_avg, yerr=2*tscale*np_std, capsize=capsize, label="Numpy backend")
plt.errorbar(lengths, x86_avg, yerr=2*tscale*x86_std, capsize=capsize, label="x86 backend")
plt.errorbar(lengths[:2], cuda_avg, yerr=2*tscale*cuda_std, capsize=capsize, label="Cuda backend")
plt.xscale("log", basex=2)
plt.yscale("log")
plt.xlabel("Number of columns (ix)")
plt.ylabel("Elapsed time (seconds)")
plt.legend()

plt.savefig("benchmark.pdf") #dpi=300
