import numpy as np
import matplotlib.pyplot as plt

nodes = np.array([1, 4, 16, 64])

n = np.array([64, 128, 256, 512])

t_cuda = np.array([
    0.19574499130249023,
    0.20641589164733887,
    0.20900678634643555,
    0.21465563774108887
])

t_cpui = np.array([
    0.12395024299621582,
    0.1373739242553711,
    0.149505615234375,
    0.16939902305603027
])

t_cpuk = np.array([
    0.10215520858764648,
    0.1146395206451416,
    0.12143230438232422,
    0.1306746006011963
])

t_numpy = np.array([
    2.7832746505737305,
    3.4239232540130615,
    3.4602155685424805,
    3.807326316833496,
])

t_cuda_ideal = np.full(4, t_cuda[0])
t_cpui_ideal = np.full(4, t_cpui[0])
t_cpuk_ideal = np.full(4, t_cpuk[0])
t_numpy_ideal = np.full(4, t_numpy[0])

fig1, ax1 = plt.subplots()
ax1.plot(nodes, t_cuda, 'ko', markersize=5, markerfacecolor='none', label = 'cuda')
ax1.plot(nodes, t_cuda_ideal, 'r--', label = 'ideal')
ax1.set_ylim([t_cuda_ideal[0] * 0.5, t_cuda_ideal[0] * 1.5])
ax1.set_xlabel("node count")
ax1.set_ylabel("run time [s]")
ax1.legend(loc="lower right")
fig1.savefig("gt4py_weak_scaling_cuda.png")
plt.close()

fig2, ax2 = plt.subplots()
ax2.plot(nodes, t_cpui, 'ko', markersize=5, markerfacecolor='none', label = 'gt:cpu_ifirst')
ax2.plot(nodes, t_cpui_ideal, 'r--', label = 'ideal')
ax2.set_ylim([t_cpui_ideal[0] * 0.5, t_cpui_ideal[0] * 1.5])
ax2.set_xlabel("node count")
ax2.set_ylabel("run time [s]")
ax2.legend(loc="lower right")
fig2.savefig("gt4py_weak_scaling_cpui.png")

fig3, ax3 = plt.subplots()
ax3.plot(nodes, t_cpuk, 'ko', markersize=5, markerfacecolor='none', label = 'gt:cpu_kfirst')
ax3.plot(nodes, t_cpuk_ideal, 'r--', label = 'ideal')
ax3.set_ylim([t_cpuk_ideal[0] * 0.5, t_cpuk_ideal[0] * 1.5])
ax3.set_xlabel("node count")
ax3.set_ylabel("run time [s]")
ax3.legend(loc="lower right")
fig3.savefig("gt4py_weak_scaling_cpuk.png")

fig4, ax4 = plt.subplots()
ax4.plot(nodes, t_numpy, 'ko', markersize=5, markerfacecolor='none', label = 'numpy')
ax4.plot(nodes, t_numpy_ideal, 'r--', label = 'ideal')
ax4.set_ylim([t_numpy_ideal[0] * 0.5, t_numpy_ideal[0] * 1.5])
ax4.set_xlabel("node count")
ax4.set_ylabel("run time [s]")
ax4.legend(loc="lower right")
fig4.savefig("gt4py_weak_scaling_numpy.png")