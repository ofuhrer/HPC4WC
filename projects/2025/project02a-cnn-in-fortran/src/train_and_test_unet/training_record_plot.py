
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import glob

matplotlib.rcParams.update({'font.size': 20})

def load_and_plot_npy_files(file_paths):
    """
    Load multiple .npy files and plot them as lines on a single figure.
    
    Args:
        *file_paths: Variable number of paths to .npy files
    """
    if not file_paths:
        print("Error: No input files provided.")
        print("Usage: python plot_npy.py file1.npy file2.npy ...")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Track if any files were successfully loaded
    loaded_files = 0
    
    for file_path in file_paths:
        try:
            # Load the .npy file
            data = np.load(file_path)
            
            # Check if it's a 1D array
            if data.ndim != 1:
                print(f"Warning: {file_path} is not a 1D array (shape: {data.shape}). Flattening...")
                data = data.flatten()

            # Remove first training loss
            data[0] = data[1]
            
            # Create x-axis values (indices)
            x = np.arange(len(data))
            
            # Get filename for label
            label = Path(file_path).stem
            
            # Plot the line
            plt.plot(x, data, label=label, linewidth=1.5, marker='o')
            
            loaded_files += 1
            print(f"Successfully loaded and plotted: {file_path} (length: {len(data)})")
            
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if loaded_files == 0:
        print("No files were successfully loaded.")
        return
    
    # Customize the plot
    plt.xlabel('Epoch')
    plt.ylabel('L1 loss')
    plt.title(f'Plot of {loaded_files} NPY Arrays')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Show the plot
    plt.savefig(f"./results/training.png")

def main():
    
    # Get file paths from command line arguments (excluding script name)
    file_paths = glob.glob('./checkpoint/unet_d*_train.npy')
    
    # Validate that all files have .npy extension
    for file_path in file_paths:
        if not file_path.endswith('.npy'):
            print(f"Warning: {file_path} doesn't have .npy extension")
            
    # print(file_paths)
    # Plot the files
    load_and_plot_npy_files(file_paths)

if __name__ == "__main__":
    main()