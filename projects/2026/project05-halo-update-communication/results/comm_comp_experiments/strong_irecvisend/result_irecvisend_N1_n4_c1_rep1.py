# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[    4, 1024, 1024,   64,    1024,  0.2572262E+02,  0.1428630E+01,  0.1941308E+02,  0.1193279E+01,  0.1920249E+02], \
] )

strategy = "irecvisend"
nodes = 1
ranks = 4
cores_per_rank = 1
mode = "strong"
rep = 1
