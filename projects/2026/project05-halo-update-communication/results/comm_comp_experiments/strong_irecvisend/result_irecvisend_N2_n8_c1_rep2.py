# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[    8, 1024, 1024,   64,    1024,  0.1261545E+02,  0.3242606E+01,  0.9534951E+01,  0.1245552E+01,  0.8978239E+01], \
] )

strategy = "irecvisend"
nodes = 2
ranks = 8
cores_per_rank = 1
mode = "strong"
rep = 2
