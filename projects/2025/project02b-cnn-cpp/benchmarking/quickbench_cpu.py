#!/usr/bin/env python3
# benchmarking/quickbench_cpu.py
import argparse, csv, os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from statistics import median

# Resolve repo root so the script works from any CWD
ROOT = Path(__file__).resolve().parent.parent

# ------------------- configuration -------------------
MODELS = [
    # name, binary path, is_parallel
    ("naive",               "build-cpu/test_mnist",                 False),
    ("stencil",             "build-cpu/test_mnist_stencil",         False),
    ("opt",                 "build-cpu/test_mnist_opt",              True),
    ("opt_alt",             "build-cpu/test_mnist_opt_alt",          True),
    ("stencil_opt",         "build-cpu/test_mnist_stencil_opt",      True),
    ("stencil_opt_alt",     "build-cpu/test_mnist_stencil_alt",      True),
]

# Human-friendly display names (output-only labels)
DISPLAY_NAME = {
    "naive":             "Naive - serial",
    "stencil":           "Stencil - serial",
    "opt":               "Naive — parallelized + optimized",
    "opt_alt":           "Naive — simple parallelization",
    "stencil_opt":       "Stencil — parallelized + optimized",
    "stencil_opt_alt":   "Stencil — simple parallelization",
}

DEFAULT_THREADS = [1, 2, 4, 8, 16]       # clamped to available cores/Slurm cpus-per-task
WARMUPS = 1
REPEATS = 2
DATA_REQUIRED = ["data/mnist/images.npy", "data/mnist/labels.npy"]

# test binaries print:  RESULT compute_time_sec=<float> samples=<int>
TIME_RE = re.compile(r"compute_time_sec\s*=\s*([0-9]*\.?[0-9]+)")
# -----------------------------------------------------

def sh(msg): print(msg, flush=True)

def clamp_threads(tlist):
    avail = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or (os.cpu_count() or 1)
    return [t for t in tlist if t <= max(1, avail)]

def libtorch_fallback(env):
    # Optional: mirror your bash fallback for LibTorch CPU runtime
    torch_dir = env.get("Torch_DIR")
    if torch_dir:
        # Torch_DIR usually points to .../share/cmake/Torch
        lib = Path(torch_dir).parents[2] / "lib"
        if lib.is_dir():
            env["LD_LIBRARY_PATH"] = str(lib) + (":" + env["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in env else "")
    else:
        home_lib = Path.home() / "libtorch-cpu" / "lib"
        if home_lib.is_dir():
            env["LD_LIBRARY_PATH"] = str(home_lib) + (":" + env["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in env else "")
    return env

def set_env_for_run(threads, samples):
    env = os.environ.copy()
    # OpenMP and math libs
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "false"
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    # test control
    env["MNIST_SAMPLES"] = str(samples)
    env["MNIST_QUIET"] = "1"
    # runtime fallback
    env = libtorch_fallback(env)
    return env

def ensure_prereqs():
    for name, path, _ in MODELS:
        p = (ROOT / path)
        if not (p.is_file() and os.access(p, os.X_OK)):
            sys.exit(f"Missing binary: {path} for model '{name}' (looked at: {p})")
    for rel in DATA_REQUIRED:
        p = (ROOT / rel)
        if not p.is_file():
            sys.exit(f"Missing dataset file: {rel} (looked at: {p})")

def try_parse_compute_time(stdout_text):
    m = TIME_RE.search(stdout_text)
    return float(m.group(1)) if m else None

def run_once(bin_path, threads, samples):
    env = set_env_for_run(threads, samples)
    t0 = time.monotonic()
    try:
        proc = subprocess.run([str(bin_path)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, check=True, cwd=str(ROOT))  # run from repo root
        wall = time.monotonic() - t0
        comp = try_parse_compute_time(proc.stdout)
        return True, wall, comp
    except subprocess.CalledProcessError:
        wall = time.monotonic() - t0
        return False, wall, None

def warmup(bin_path, threads, samples):
    for _ in range(WARMUPS):
        run_once(bin_path, threads, samples)

def main():
    parser = argparse.ArgumentParser(description="Quick CPU benchmark (no third-party deps required)")
    parser.add_argument("--threads", nargs="*", type=int, default=DEFAULT_THREADS,
                        help="Thread sweep for parallel models")
    parser.add_argument("--samples", type=int, default=int(os.environ.get("MNIST_SAMPLES", "10000")),
                        help="MNIST samples per run")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help="Repeats per (model,threads)")
    parser.add_argument("--outdir", default=None,
                        help="Optional explicit output directory")
    args = parser.parse_args()

    ensure_prereqs()

    threads = clamp_threads(args.threads)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    base = Path(__file__).resolve().parent / "results" / f"{ts}_quick_cpu"
    if args.outdir:
        base = Path(args.outdir)
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / "results.csv"

    sh(f"Quick CPU benchmark")
    sh(f"- samples per run: {args.samples}")
    sh(f"- repeats per point: {args.repeats}")
    sh(f"- thread sweep (parallel models): {threads}\n")

    rows = []  # model, threads, run_id, total_time_s, compute_time_s

    for name, bin_path, is_par in MODELS:
        tlist = [1] if not is_par else threads
        pretty = DISPLAY_NAME.get(name, name)
        sh(f"== {pretty} ==")
        bin_exec = (ROOT / bin_path)
        for t in tlist:
            warmup(bin_exec, t, args.samples)
            totals, computes = [], []
            for r in range(args.repeats):
                ok, wall, comp = run_once(bin_exec, t, args.samples)
                if ok:
                    rows.append(dict(model=pretty, threads=t, run_id=r,
                                     total_time_s=f"{wall:.6f}",  # keep consistent numeric formatting
                                     compute_time_s=(f"{comp:.6f}" if comp is not None else "")))
                    totals.append(wall)
                    if comp is not None:
                        computes.append(comp)
                    sh(f"T={t:2d} run {r+1}/{args.repeats}  wall={wall:.3f}s  compute={('%.3f'%comp) if comp is not None else 'n/a'}")
                else:
                    rows.append(dict(model=pretty, threads=t, run_id=r,
                                     total_time_s="", compute_time_s=""))
                    sh(f"T={t:2d} run {r+1}/{args.repeats}  FAILED")

            if totals:
                med_wall = median(totals)
                med_comp = median(computes) if computes else None
                sh(f"-> median  wall={med_wall:.3f}s  compute={(f'{med_comp:.3f}s' if med_comp is not None else 'n/a')}")
        sh("")

    # write CSV
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "threads", "run_id", "total_time_s", "compute_time_s"])
        w.writeheader()
        w.writerows(rows)

    sh(f"Saved results to: {csv_path.parent}")
    sh("Done.")

if __name__ == "__main__":
    main()
