How to run the scripts:

In the CSCS Jupyter environment, open a terminal and run the script with the following command:

srun -n 11 python stencil2d-mpi-overlapping.py --nx 128 --ny 128 --nz 64 --num_iter 1024 --plot_result True

Adjust the path, n, and the other parameters as needed.


Scripts:


01_stencil2d-mpi-IsendRecv.py

works until -n 22. 
-n 23, -n 24 don't work anymore


02_stencil2d_mpi_SendIrecv.py

only works with -n  1, 2, 3, 4, 8 


03_stencil2d-mpi-Sendrecive.py

works


04_stencil2d_evenodd_version1.py

doesn't work, can create a deadlock


05_stencil2d_evenodd_version2.py

doesn't work, can create a deadlock


06_stencil2d-mpi-overlapping.py

With the for loop, it works but it is not faster


MPI.Request.Waitall(reqs) does not work

  # wait and unpack
    #for req in reqs_tb:
    #    req.barrier()
   MPI.Request.Waitall(reqs)

07_stencil2d-mpi-sendIrecve.py

works


  
  


