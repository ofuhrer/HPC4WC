# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[   32, 1024, 1024,   64,    1024,  0.4586125E+01,  0.2647304E+01,  0.3417045E+01,  0.8736100E+00,  0.2969827E+01], \
] )

strategy = "irecvisend"
nodes = 8
ranks = 32
cores_per_rank = 1
mode = "strong"
rep = 3
