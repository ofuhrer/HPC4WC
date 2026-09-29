# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[    1, 1024, 1024,   64,    1024,  0.9305283E+02,  0.1366152E+01,  0.7176609E+02,  0.1366152E+01,  0.7176609E+02], \
] )

strategy = "irecvisend"
nodes = 1
ranks = 1
cores_per_rank = 1
mode = "strong"
rep = 2
