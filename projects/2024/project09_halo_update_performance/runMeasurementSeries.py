# This code runs the series of measurmenets over a given set of parameters

import subprocess
import itertools

# Define the ranges for nNodes and nHalo
nNodes_range = [1, 2, 3, 4, 5, 6, 7, 8, 9]  
nHalo_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]   

# Loop over all combinations of nNodes and nHalo and measure their halo_update() performance
for nNodes, nHalo in itertools.product(nNodes_range, nHalo_range):
    # Construct the command
    command = [
        'srun', '-n', str(nNodes),
        'stencil_nnode',
        '--nx', '256',
        '--ny', '256',
        '--nz', '64',
        '--nhalo', str(nHalo),
        '--niter', '1024'
    ]
    
    # Print the current command
    print(f"Running command: {' '.join(command)}")
    
    # Run the current command
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print the output and error (if any)
    print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Warnings and errors:\n{result.stderr}")

# Find reference fields to later verify correct diffusion
for nHalo in nHalo_range:
    # Construct the command
    command = [
        'srun', '-n', str(1),
        'stencil_1node',
        '--nx', '256',
        '--ny', '256',
        '--nz', '64',
        '--nhalo', str(nHalo),
        '--niter', '1024'
    ]

    # Print the current command
    print(f"Running command: {' '.join(command)}")
    
    # Run the current command
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print the output and error (if any)
    print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Warnings and errors:\n{result.stderr}")

print("Measurement series terminated.")


