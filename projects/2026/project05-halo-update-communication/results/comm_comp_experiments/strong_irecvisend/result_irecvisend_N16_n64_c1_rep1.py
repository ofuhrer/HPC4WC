# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[   64, 1024, 1024,   64,    1024,  0.3476901E+01,  0.2361135E+01,  0.2574545E+01,  0.8050458E+00,  0.2126830E+01], \
] )

strategy = "irecvisend"
nodes = 16
ranks = 64
cores_per_rank = 1
mode = "strong"
rep = 1
