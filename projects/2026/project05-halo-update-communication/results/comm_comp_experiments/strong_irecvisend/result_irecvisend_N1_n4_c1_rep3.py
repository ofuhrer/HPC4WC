# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[    4, 1024, 1024,   64,    1024,  0.2553572E+02,  0.1547992E+01,  0.1928481E+02,  0.1224202E+01,  0.1903908E+02], \
] )

strategy = "irecvisend"
nodes = 1
ranks = 4
cores_per_rank = 1
mode = "strong"
rep = 3
