import yaml
import pandas as pd
import subprocess
import sys
from Fields_Manager import Fields_Manager
import numpy as np
import os
import shutil


# load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

n_repeats = config.get("n_repeats", 10)
num_halo = config.get("num_halo", 2)
if config.get("save_plots", False):
    os.makedirs(config.get("plot_folder", "plots"), exist_ok=True)

fields_manager = Fields_Manager()

# get currently running python installation
executable = sys.executable

# init results
results = []

# Build nx and ny combinations (with regards to square config option)
if config.get("only_calc_squares", True):
    nx_ny_combinations = [[nx, nx] for nx in config.get("nx", [16])]
else:
    nx_ny_combinations = []
    for nx in config.get("nx", [16]):
        for ny in config.get("ny", [16]):
            nx_ny_combinations.append([nx, ny])


for nx, ny in nx_ny_combinations:
    for nz in config["nz"]:
        # building the in field
        fields_manager.build_field(config.get("test_field_style", "broad_center_spike"), nx, ny, nz, num_halo)

        for framework in config.get("used_frameworks", []):
            framework_cfg = config.get("frameworks", {}).get(framework, {})

            for device in framework_cfg.get("devices", []):
                if device in config.get("ignored_devices", []):
                    continue
        
                for num_iter in config["num_iter"]:
                    command = [
                        executable,
                        framework_cfg.get("script_path", ""),
                        "--in_field_path", fields_manager.in_field_path,
                        "--out_field_path", fields_manager.out_field_path,
                        "--num_iter", str(num_iter),
                        "--num_halo", str(num_halo),
                        "--device", str(device)
                    ]

                    # Warming up
                    #print(f"Warming up {framework} on {device} for {nx, ny, nz} domain...")
                    #print(command)
                    #subprocess.run(command, capture_output=True, text=True, check=True)

                    
                    
                    print(f"Running {framework} on {device} with...\n {' '.join(command[2:])}")
                    
                    runtimes = []

                    for rep in range(n_repeats):
                        output = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        runtime = float(output.stdout.strip().split('\n')[-1])
                        runtimes.append(runtime)
                        
                        print(f"  repetition {rep+1}/{n_repeats} took: {runtime:.6f}s")
                        
                    for rep, runtime in enumerate(runtimes):
                    
                        results.append({
                            "framework": framework,
                            "nx": nx,
                            "ny": ny,
                            "nz": nz,
                            "num_iter": num_iter,
                            "device": device,
                            "repetition": rep,
                            "time": runtime,
                            "time_per_work": runtime/(nx*ny*nz),
                            "field_type": config.get("test_field_style", "broad_center_spike")
                        })

                        array_filename = f"{config.get("test_field_style", "broad_center_spike")}_{framework}_{device}_{nx}_{ny}_{nz}_{num_iter}.npy"
                        array_filepath = f"{config.get("array_folder", "array")}/{array_filename}"
                        shutil.copy(fields_manager.out_field_path, array_filepath)
                        
                    if config.get("save_plots", False):
                        plot_filename = f"{config.get("test_field_style", "broad_center_spike")}_{framework}_{device}_{nx}_{ny}_{nz}_{num_iter}.png"
                        plot_filepath = f"{config.get("plot_folder", "plots")}/{plot_filename}"
                        #array_filename = f"{config.get("test_field_style", "broad_center_spike")}_{framework}_{device}_{nx}_{ny}_{nz}_{num_iter}.npy"
                        #array_filepath = f"{config.get("array_folder", "array")}/{array_filename}"


                        fields_manager.plot_field_comparison(plot_filepath)
                        output_array = np.load(fields_manager.out_field_path)
                        #np.save(array_filepath, output_array)
                        
data = pd.DataFrame(results)
#data.to_csv("output/results.csv", index=False)
print(data)

