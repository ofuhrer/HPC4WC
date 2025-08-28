import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os



# function for data extraction
def extract_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            if "###" in line:
                line = line.replace("###", "")
                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        size = int(parts[0])
                        num_streams = int(parts[1])
                        time = float(parts[2])
                        data.append((size, num_streams, time))
                    except Exception as e:
                        print(f"Skipping line (parsing error): {line}")
                    colnames = ["Size", "NUM_STREAMS", "Time"]
                elif len(parts) == 4:
                    try:
                        calls = int(parts[0])
                        size = int(parts[1])
                        num_streams = int(parts[2])
                        time = float(parts[3])
                        data.append((calls, size, num_streams, time))
                    except Exception as e:
                        print(f"Skipping line (parsing error): {line}")
                    colnames = ["Calls", "Size", "NUM_STREAMS", "Time"]
                elif len(parts) == 7:
                    try:
                        nx = int(parts[1])
                        ny = int(parts[2])
                        nz = int(parts[3])
                        num_iter = int(parts[4])
                        time = float(parts[5])
                        num_streams = int(parts[6])
                        data.append((nx, ny, nz, num_iter, num_streams, time))
                    except Exception as e:
                        print(f"Skipping line (parsing error): {line}")
                    colnames = ["Nx", "Ny", "Nz", "NUM_ITER", "NUM_STREAMS", "Time"]
    return pd.DataFrame(data, columns=colnames)
    



# Take filename as argument
if len(sys.argv) < 2:
    print("Usage: python analyze_output.py <output_file.out>")
    sys.exit(1)

filename = sys.argv[1]
print(f"Reading from: {filename}")
data = extract_data(filename)

# Set pandas display options to show scientific notation
pd.set_option('display.float_format', '{:.6e}'.format)

print(data)

# Save to CSV for future analysis
csv_path = os.path.splitext(filename)[0] + ".csv"
print(f"Saving CSV to: {csv_path}")

# First save the CSV (this will preserve full precision)
# Format the Time column in scientific notation for CSV output
data_formatted = data.copy()
data_formatted['Time'] = data_formatted['Time'].apply(lambda x: f'{x:.6e}')
data_formatted.to_csv(csv_path, index=False)

# Then read it back and print a few lines to verify
check_df = pd.read_csv(csv_path)
print("Reloaded CSV preview:")
print(check_df.head())