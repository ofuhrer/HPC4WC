# This code uses the compare_fields.py function from day3 of the exercises to look see which fields are completely close to the reference fields, given some relative and absolute tolerances. It shows that fields with 3 nodes and more than 4 nodes are not everywhere the same as the reference fields. The differences are looked at in more detail in test_notebook.ipynb, together with an outline of their most likely cause.

import subprocess
import itertools

# Define the ranges for nNodes and nHalo
nNodes_range = [1, 2, 3, 4, 5, 6, 7, 8, 9]  
nHalo_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]   

# Loop over all combinations of nNodes and nHalo and measure their halo_update() performance
for nNodes, nHalo in itertools.product(nNodes_range, nHalo_range):
    # Construct the command
    command = [
        'python', 'compare_fields.py', '--src', 
        'output_folder/out_fields/out_field_nNodes_' + str(nNodes) + '_nHalo_' + str(nHalo) + '.dat',
        '--trg',
        'output_folder/out_fields/out_field_reference_' + 'nHalo_' + str(nHalo) + '.dat',
        '--rtol', str(1e-8),
        '--atol', str(1e-5)
    ]
    
    # Print the current command
    print(f"Running command: {' '.join(command)}")
    
    # Run the current command
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print the output and error (if any)
    print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Warnings and errors:\n{result.stderr}")

print("Verification series terminated.")


