import time
import csv
import os.path
import argparse
from stencil import Dummy_Stencil
# from numpy_stencils import Laplacian2D, Laplacian3D, Biharmonic2D, Biharmonic3D
import numpy_stencils
# import gt4py_stencils
# import jax_stencils


def dicts_to_csv(to_csv, file_name, header):
    keys = to_csv[0].keys()
    if header:
        with open(file_name, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(to_csv)
    else:
        with open(file_name, 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writerows(to_csv)


def time_iterations(stencil, number_of_iterations):
    # returns duration of a single stencil.run() call in ms
    stencil.activate()
    stencil.run()  # warmup

    tic = time.perf_counter()
    for i in range(number_of_iterations):
        stencil.run()
    try:
        stencil.sync()
    except:
        pass
    toc = time.perf_counter()
    stencil.deactivate()
    return (toc-tic)/number_of_iterations


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--libs",
        type=str,
        nargs="+",
        default="numpy",
        choices=['numpy', 'gt4py', 'jax']
    )
    parser.add_argument("--backend",
        type=str,
        nargs="+",
        default="cpu",
        choices=['cpu', 'gpu']
    )
    parser.add_argument("--ns",
        type=int,
        nargs="+",
        required=True,
        help="List of Domain Sizes (e.g. 100 200 300)"
    )
    parser.add_argument("--cores",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="number of cores"
    )

    parser.add_argument("--n_iter", type=int, default=4, help="Number of iterations")
    parser.add_argument("--out", type=str, default="timings.csv", help="Output csv file")
    
    args = parser.parse_args()
    number_of_iterations = args.n_iter
    ns = args.ns
    
    timings_dicts = []
    for n in ns:
        current_row = {"n": n}
        if args.cores != None:
            print("write cores")
            current_row["cores"] = args.cores[0]
        stencils = []
        
        if "numpy" in args.libs:
            import numpy_stencils
            stencils.extend([
                numpy_stencils.Laplacian2D(n),
#                numpy_stencils.Laplacian3D(n),
                numpy_stencils.Biharmonic2D(n),
#                numpy_stencils.Biharmonic3D(n)
            ])
        if "gt4py" in args.libs:
            import gt4py_stencils
            if "cpu" in args.backend:
                gt4py_backend = "gtx86"
            elif "gpu" in args.backend:
                gt4py_backend = "gtcuda"
            else:
                raise Exception("No backend defined")
            stencils.extend([
                gt4py_stencils.Laplacian2D(n, gt4py_backend),
#                gt4py_stencils.Laplacian3D(n, gt4py_backend),
                gt4py_stencils.Biharmonic2D(n, gt4py_backend),
#                gt4py_stencils.Biharmonic3D(n, gt4py_backend)
            ])
        if "jax" in args.libs:
            import jax_stencils
            stencils.extend([
                jax_stencils.Laplacian2D(n),
#                jax_stencils.Laplacian3D(n),
                jax_stencils.Biharmonic2D(n),
#                jax_stencils.Biharmonic3D(n)
            ]) 

        for s in stencils:
            try:
                average_duration = time_iterations(s, number_of_iterations)
                print(f"Stencil: {s}\nN: {n}\nNumber of Iterations: {number_of_iterations}\nTime: {average_duration}s\n")

                current_row[str(s)]=average_duration
            except Exception as e:
                print(f"Error in {s} with msg:\n {e}")
                current_row[str(s)]=None

        timings_dicts.append(current_row)

    if args.cores != None:
        if not os.path.isfile(args.out):
            print("file does not exist")
            dicts_to_csv(timings_dicts, args.out, True)

        else:
            print("file does exist")
            dicts_to_csv(timings_dicts, args.out, False)
    else:
        dicts_to_csv(timings_dicts, args.out, True)

if __name__ == "__main__":
    main()
