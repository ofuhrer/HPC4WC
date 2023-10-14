import csv

"""
Description

This file generates different grid sizes and stores them into a CSV file.

Those CSV files are used to have a uniform grid sizes for benchmarking
across all languages.

IMPORTANT:
DO NOT WRITE WHOLE DATA SETS TO DISK AS THEY WILL EAT UP A LOT OF SPACE!
"""

# Define 2-exponents
min_exp = 5
max_exp = 9
alpha = 1.0/32.0
num_iter = 100

with open('./universal_input/input_dimensions.csv', 'w') as csvfile:
    # Define header
    fieldnames = ['x_dim', 'y_dim', 'z_dim', 'alpha', 'num_iter']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Create grid and store it as 'input_dimensions.csv'
    file_input = csv.writer(csvfile, delimiter=',')
    for xyz_dim in range(min_exp, max_exp):
        file_input.writerow([2**xyz_dim, 2**xyz_dim, 2**xyz_dim, alpha, num_iter])
