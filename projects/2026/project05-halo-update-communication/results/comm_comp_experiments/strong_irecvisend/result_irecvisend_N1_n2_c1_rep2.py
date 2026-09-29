# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[    2, 1024, 1024,   64,    1024,  0.4440208E+02,  0.1300486E+01,  0.3381351E+02,  0.1107466E+01,  0.3365372E+02], \
] )

strategy = "irecvisend"
nodes = 1
ranks = 2
cores_per_rank = 1
mode = "strong"
rep = 2
