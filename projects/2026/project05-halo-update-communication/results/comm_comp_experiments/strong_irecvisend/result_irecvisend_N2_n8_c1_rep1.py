# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[    8, 1024, 1024,   64,    1024,  0.1274418E+02,  0.3238026E+01,  0.9659933E+01,  0.1196595E+01,  0.9139102E+01], \
] )

strategy = "irecvisend"
nodes = 2
ranks = 8
cores_per_rank = 1
mode = "strong"
rep = 1
