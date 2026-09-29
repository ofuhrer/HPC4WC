# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[   32, 1024, 1024,   64,    1024,  0.4634087E+01,  0.2675601E+01,  0.3465127E+01,  0.9226068E+00,  0.2966956E+01], \
] )

strategy = "irecvisend"
nodes = 8
ranks = 32
cores_per_rank = 1
mode = "strong"
rep = 2
