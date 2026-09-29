import yaml
import pandas as pd
import subprocess
import sys


# load config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# get currently running python installation
executable = sys.executable

# init results
results = []

for framework, script_path in config["script-paths"].items():
   # which params this framework's script actually accepts (default: all)
    framework_cfg = config.get("script-options", {}).get(framework, {"options": ["nx", "ny", "nz", "num_iter", "device"]})
    allowed_options = framework_cfg.get("options", ["nx", "ny", "nz", "num_iter", "device"])
    default_device = framework_cfg.get("default_device", None)

    n_repeats = config.get("n_repeats", 10)
    devices = (
            config["device"] 
            if "device" in allowed_options
            else [default_device]
                )

    for device in devices:
        
        for nx in config["sizes"]:
            for nx in config["sizes"]:
                for nz in config["nz"]:
                            
                    for num_iter in config["num_iter"]:
                        command = [
                            executable,
                            script_path,
                            "--nx", str(nx),
                            "--ny", str(ny),
                            "--nz", str(nz),
                            "--num_iter", str(num_iter)
                        ]
    
                        if "device" in allowed_options:
                            command += ["--device", str(device)]
    
                        # determine the actual device used for this run
                        used_device = device if "device" in allowed_options else default_device
    
                        # warmup = True
                        # Warming up if not already done
                        # if warmup:
                        print(f"Warming up {framework} on {used_device} for {nx, ny, nz} domain...")
                        subprocess.run(command, capture_output=True, text=True, check=True)
                        # warmup = False
    
                        
                        print(f"Running {framework} on {used_device} with...\n {' '.join(command[2:])}")
                        
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
                                "device": used_device,
                                "repetition": rep,
                                "time": runtime,
                                "time_per_work": runtime/(nx*ny*nz)
                            })
                        
data = pd.DataFrame(results)
data.to_csv("output/results.csv", index=False)
print(data)

