# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[    4, 1024, 1024,   64,    1024,  0.2528963E+02,  0.1183615E+01,  0.1911602E+02,  0.1058299E+01,  0.1893256E+02], \
] )

strategy = "irecvisend"
nodes = 1
ranks = 4
cores_per_rank = 1
mode = "strong"
rep = 2
