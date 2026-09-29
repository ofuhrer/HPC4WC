# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[   16, 1024, 1024,   64,    1024,  0.8549132E+01,  0.4646684E+01,  0.6406246E+01,  0.1279082E+01,  0.5773381E+01], \
] )

strategy = "irecvisend"
nodes = 4
ranks = 16
cores_per_rank = 1
mode = "strong"
rep = 2
