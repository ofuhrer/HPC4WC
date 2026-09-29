# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[    2, 1024, 1024,   64,    1024,  0.4447528E+02,  0.1252773E+01,  0.3392076E+02,  0.1081585E+01,  0.3375216E+02], \
] )

strategy = "irecvisend"
nodes = 1
ranks = 2
cores_per_rank = 1
mode = "strong"
rep = 3
