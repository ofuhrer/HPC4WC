# ranks nx ny nz num_iter time max_comm max_compute avg_comm avg_compute
# comm_strategy = irecvisend
data = np.array( [ \
[   64, 1024, 1024,   64,    1024,  0.3502792E+01,  0.2383227E+01,  0.2584815E+01,  0.8324042E+00,  0.2128905E+01], \
] )

strategy = "irecvisend"
nodes = 16
ranks = 64
cores_per_rank = 1
mode = "strong"
rep = 2
