# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[   64, 1024, 1024,   64,    1024,  0.3481686E+01,  0.2369545E+01,  0.2591665E+01,  0.8170991E+00,  0.2122587E+01], \
] )

strategy = "irecvisend"
nodes = 16
ranks = 64
cores_per_rank = 1
mode = "strong"
rep = 3
