# ******************************************************
#     Program: Solution Checker
#      Author: Tobias Rahn
#       Email: tobias.rahn@inf.ethz.ch
#        Date: 28.06.2024
# Description: Compares two solutions of 4th-order diffusion for similarity
# ******************************************************
import argparse
import re
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from stencil2d_baseline import calculations as solution_calc


"""Path of the folder where the baseline solutions are stored"""
BASELINE_SOLUTION_PATH = "../data/baseline"


def parse_file_args(file_path):
    """Extract the arguments used for the 4th order diffusion from the file path.

    :param file_path: The path of the solution file
    :return: (xn, yn, zn, num_iter)
    """

    file_name = file_path.split("/")[-1]
    raw_file_name = re.sub("[^0-9|_|-]", "", file_name)  # Remove field descriptions
    return tuple([int(v) for v in raw_file_name.split("-")[-1].split("_")])


def main():
    parser = argparse.ArgumentParser(
        prog="2d Stencil solution checker",
        description="Compare the solution of two '2d Stencil' implementation to check if they provide the same results",
        epilog="",
    )

    parser.add_argument(
        "-s", "--solution", metavar="S", type=str, help="Provide the path of a solution file", required=True
    )
    parser.add_argument(
        "-r", "--rtol", metavar="T", type=float, default=1e-5, help="Relative tolerance for the comparison."
    )
    parser.add_argument(
        "-a", "--atol", metavar="T", type=float, default=1e-8, help="Absolute tolerance for the comparison."
    )
    parser.add_argument("-p", "--plot_result", type=bool, default=False, help="Make a plot of the result?")
    parser.add_argument("-b", "--plot_baseline", type=bool, default=False, help="Make a plot of the baseline?")
    args = parser.parse_args()

    solution = np.load(args.solution)

    # Parse file path
    try:
        nx, ny, nz, num_iter, num_halo, precision = parse_file_args(args.solution)
    except ValueError:
        raise ValueError(
            "The file path does not contain the necessary information! Consult the README.md in the scripts folder for more information."
        )

    # Load Baseline Solution
    baseline = None
    for f in os.listdir(BASELINE_SOLUTION_PATH):
        if f"nx{nx}_ny{ny}_nz{nz}_iter{num_iter}_halo{num_halo}_p{precision}" in f:
            baseline = np.load(os.path.join(BASELINE_SOLUTION_PATH, f))

    if baseline is None:
        # Solution has not been generated yet
        baseline = solution_calc(
            nx, ny, nz, num_iter, num_halo, precision, result_dir=BASELINE_SOLUTION_PATH, return_result=True
        )

    # Compare the two solutions
    comparison_result = np.allclose(solution, baseline, rtol=args.rtol, atol=args.atol, equal_nan=True)

    if comparison_result:
        print("The check was successful!")
    else:
        print("The check was NOT successful!")

    # Plot the results if desired
    if args.plot_result:
        plt.ioff()
        plt.imshow(solution[solution.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.show()

    # Plot the baseline if desired
    if args.plot_baseline:
        plt.ioff()
        plt.imshow(baseline[baseline.shape[0] // 2, :, :], origin="lower")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    os.chdir(sys.path[0])  # Change the directory
    main()
