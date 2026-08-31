"""
Sweep driver for the Burgers project.

Usage:
    python run_sweep.py [mode]
    modes:
      - 0: Run in benchmark mode with no output
      - 1: Output initial and final fields
      - 2: Output fields at preset frequency

Outputs:
    timing_results.csv    -- one row per (precision, domain size) run
    ./output/n*_f*/*.bin  -- field snapshots, as produced by burgers.c
"""
import os
import sys
import re
import subprocess
import time
import csv

# ---- OMP settings --------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "72"
os.environ["OMP_PROC_BIND"]   = "close"
os.environ["OMP_PLACES"]      = "cores"

# ---- Sweep parameters -----------------------------------------------------
PRECISIONS    = [16, 32, 64]
DOMAIN_SIZES  = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]  # nx = ny
NZ            = 10          # per the project plan, doesn't affect the physics
SOURCE        = "src/burgers.c"
STD           = "c17" #after change to the CSCS environment gcc doesn't support c23 anymore  
TIME_BUDGET_S = None      

# ---- Compile once per precision -------------------------------------------
def compile_precision(fp):
    exe = f"burgers_{fp}"
    cmd = ["gcc", f"-DFLOATXX={fp}", "-fopenmp", f"-std={STD}",
           "-Wfatal-errors", "-O3", "-mcpu=neoverse-v2", SOURCE, "-o", exe, "-lm"]
    print("compiling:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return exe

# ---- Run one (precision, domain size) case --------------------------------
TIME_RE = re.compile(r"Time elapsed\s*=\s*([\d.eE+-]+)")

def run_case(exe, nx, nz, mode):
    t0 = time.time()
    proc = subprocess.run([f"./{exe}", str(nx), str(nz), str(mode)],
                           capture_output=True, text=True)
    wall = time.time() - t0
    m = TIME_RE.search(proc.stdout)
    internal = float(m.group(1)) if m else None
    if proc.returncode != 0:
        print(f"  !! run failed (nx={nx}): {proc.stderr[:300]}")
    return wall, internal


CSV_PATH = "timing_results.csv"
FIELDNAMES = ["precision", "nx", "ny", "nz", "wall_time_s", "internal_time_s"]

def already_done():
    """Rows from a previous (possibly interrupted) run, so reruns can resume
    instead of redoing everything from scratch."""
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["precision"]), int(row["nx"])))
    return done

def append_row(row):
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def main():
    if len(sys.argv) < 2:
        mode = 0
    else:
        mode = int(sys.argv[1])

    if mode == 0:
        print("Running in benchmark mode with no output")
    elif mode == 1:
        print("Outputting initial and final fields")
    elif mode == 2:
        print("Outputting fields at preset frequency")
    else:
        print("Invalid argument, defaulting to outputting initial and final fields")
        mode = 1
    
    exes = {fp: compile_precision(fp) for fp in PRECISIONS}
    done = already_done()
    if mode == 0 and done:
        print(f"resuming: {len(done)} (precision, nx) cases already in {CSV_PATH}, skipping those")

    for fp in PRECISIONS:
        exe = exes[fp]
        for n in DOMAIN_SIZES:
            if mode == 0:
                if (fp, n) in done:
                    continue
                print(f"running fp{fp:>3}  nx=ny={n:>5}  nz={NZ} ...", end=" ", flush=True)
                wall_sum = 0.0
                internal_sum = 0.0
                num_runs = 0
                while wall_sum < 10.0:
                    wall, internal = run_case(exe, n, NZ, mode)
                    wall_sum += wall
                    internal_sum += internal
                    num_runs += 1
                wall_avg = wall_sum / num_runs
                internal_avg = internal_sum / num_runs
                print(f"wall={wall_avg:.3g}s  internal={internal_avg:.3g}s")
                append_row(dict(precision=fp, nx=n, ny=n, nz=NZ,
                                 wall_time_s=wall_avg, internal_time_s=internal_avg))
            else:
                print(f"running fp{fp:>3}  nx=ny={n:>5}  nz={NZ} ...")
                wall, internal = run_case(exe, n, NZ, mode)
            

    print(f"\nDone. Results in {CSV_PATH}")


if __name__ == "__main__":
    main()