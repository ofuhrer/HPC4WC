# ******************************************************
#     Program: compare_fields.py
#      Author: Stefano Ubbiali
#       Email: subbiali@phys.ethz.ch
#        Date: 04.06.2020
# Description: Comparing two NumPy arrays
# ******************************************************
import click
import numpy as np
import matplotlib.pyplot as plt
### READ BINARY DATA HERE - to do ###
#np.fromfile(stuff)

@click.command()
@click.option("--src", type=str, required=True, help="Path to the first field.")
@click.option("--trg", type=str, required=True, help="Path to the second field.")
@click.option(
    "--rtol", type=float, required=False, default=1e-5, help="Relative tolerance."
)
@click.option(
    "--atol", type=float, required=False, default=1e-8, help="Absolute tolerance."
)
def main(src, trg, rtol=1e-4, atol=1e-4): 
    src_f = np.fromfile(src,float) #np.load(src)
    print(np.shape(src_f))
    #src_f = np.array(src_f)
    trg_f = np.fromfile(trg,float) #np.load(trg)
    print(np.shape(trg_f))
    #trg_f = np.array(trg_f)
    truth = np.zeros(len(src_f))
    
    for i in range (0,len(src_f)):
        if np.allclose(src_f[i], trg_f[i], rtol=rtol, atol=atol, equal_nan=True): 
            #src_f[i] == trg_f[i]:
            truth[i] = 1
        else:
            print('not equal at this point:', i, src_f[i], trg_f[i])
            truth[i] = 0
    
    if np.allclose(src_f, trg_f, rtol=rtol, atol=atol, equal_nan=True):
        print(f"HOORAY! '{src}' and '{trg}' are equal!")
    else:
        print(f"{src} and {trg} are not equal.")
        
    print(truth)
    np.savetxt('equivalence_mpi20.txt', truth)


if __name__ == "__main__":
    main()
    
    
#src, trg, rtol=1e-5, atol=1e-8)