#!/usr/bin/env python3
"""
stencil_bench.py — run NumPy / Numba / Torch / JAX stencil drivers across sizes,
repeat each configuration N times, and save per-program boxplots.

these scripts need ot exist in CWD:
  - numpy: stencil2d_new.py
  - numba: stencil2d_numba_new.py
  - torch: stencil2d_torch_new.py
  - jax:   stencil2d_jax_new.py
"""
from __future__ import annotations
import argparse, subprocess, sys, time, re, json, os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNTIME_REGEX = re.compile(r"Elapsed time for work\s*=\s*([0-9]*\.?[0-9]+)\s*s")

PROGRAMS = {
    "numpy": ["python", "stencil2d_new.py"],
    "numba": ["python", "stencil2d_numba_new.py"],
    "torch": ["python", "stencil2d_torch_new.py", "--device", "cpu"],
    "jax":   ["python", "stencil2d_jax_new.py", "--device", "cpu"],
}

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark stencil drivers and plot boxplots.")
    p.add_argument("--programs", nargs="+", default=["numpy", "numba", "torch", "jax"],
                   choices=list(PROGRAMS.keys()))
    p.add_argument("--sizes", nargs="+", type=int, default=[32, 48, 64, 96, 128, 192])
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--iters", type=int, default=128)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--halo", type=int, default=2)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--srun", action="store_true", help="Wrap calls in 'srun -n 1'.")
    # Advanced (optional): you can still override defaults if you want.
    p.add_argument("--threads", type=int, default=0, help="Force thread count for children.")
    p.add_argument("--numba-threading-layer", choices=["omp","workqueue","tbb","default"], default=None,
                   help="Force NUMBA_THREADING_LAYER for Numba children.")
    p.add_argument("--env", action="append", default=[], help="Extra KEY=VAL for children.")
    p.add_argument("--extra", type=str, default="", help="Extra CLI appended to every call.")
    return p.parse_args()

def detect_threads() -> int:
    # sensible default on HPC nodes
    for k in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        v = os.environ.get(k)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1

def run_once(cmd: List[str], env: Dict[str,str]) -> Tuple[float, str, str]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=env if env else None)
    out, err = proc.stdout, proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDERR:\n{err}\nSTDOUT:\n{out}")
    m = RUNTIME_REGEX.search(out)
    if not m:
        raise RuntimeError(f"Could not parse runtime from output.\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return (float(m.group(1)), out, err)

def working_set_mb(nx:int, ny:int, nz:int, bytes_per_scalar:int=4, fields:int=3)->float:
    return (nx*ny*nz*fields*bytes_per_scalar)/1e6

def runtime_per_gridpoint_us(time_s: float, nx:int, ny:int, nz:int)->float:
    return (time_s/(nx*ny*nz))*1e6

def make_child_env(base_env: Dict[str,str], prog_name: str, threads:int,
                   numba_layer: str|None, extra_env_kv: List[str]) -> Dict[str,str]:
    env = dict(base_env)
    if threads and threads > 0:
        env["OMP_NUM_THREADS"] = str(threads)
        env["NUMBA_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = "1"
        # Also good for Torch/JAX on CPU; avoids oversubscription.
    if prog_name == "numba":
        # Default to a *safe* layer unless user overrode it.
        if numba_layer is None:
            env["NUMBA_THREADING_LAYER"] = "workqueue"
        elif numba_layer != "default":
            env["NUMBA_THREADING_LAYER"] = numba_layer
        else:
            env.pop("NUMBA_THREADING_LAYER", None)
    for item in extra_env_kv:
        if "=" in item:
            k, v = item.split("=", 1)
            env[k.strip()] = v.strip()
    return env

def bench_program(name:str, sizes:List[int], nz:int, iters:int, reps:int, halo:int,
                  use_srun:bool, extra_cli:str, outdir:Path, threads:int,
                  numba_layer:str|None, extra_env_kv:List[str]) -> Dict:
    base_cmd = PROGRAMS[name][:]
    results = {"program":name,"sizes":sizes,"nz":nz,"iters":iters,"reps":reps,"halo":halo,"records":[]}
    for n in sizes:
        ws_mb = working_set_mb(n, n, nz)
        for r in range(reps):
            args = base_cmd + ["--nx",str(n),"--ny",str(n),"--nz",str(nz),
                               "--num_iter",str(iters),"--num_halo",str(halo)]
            if extra_cli: args += extra_cli.split()
            cmd = (["srun","-n","1"] + args) if use_srun else args

            # Primary env
            env_primary = make_child_env(os.environ, name, threads, numba_layer, extra_env_kv)

            # Fallback env ONLY for Numba: ultra safe (workqueue + 1 thread)
            env_fallback = None
            if name == "numba":
                env_fallback = dict(env_primary)
                env_fallback["NUMBA_THREADING_LAYER"] = "workqueue"
                env_fallback["NUMBA_NUM_THREADS"] = "1"
                env_fallback["OMP_NUM_THREADS"] = "1"
                env_fallback["MKL_NUM_THREADS"] = "1"

            # Try primary, then fallback if needed
            attempts = [(env_primary, "primary")]
            if env_fallback is not None:
                attempts.append((env_fallback, "fallback"))

            success = False
            last_err = None
            for env, tag in attempts:
                try:
                    elapsed, out, err = run_once(cmd, env)
                    success = True
                    break
                except Exception as e:
                    last_err = e
                    continue

            if not success:
                print(f"[WARN] {name} size={n} rep={r+1} failed after fallback: {last_err}", file=sys.stderr)
                elapsed = float("nan")

            rpg_us = runtime_per_gridpoint_us(elapsed, n, n, nz)
            results["records"].append({
                "nx":n,"ny":n,"nz":nz,"iter":iters,"rep":r+1,
                "elapsed_s":elapsed,"rpg_us":rpg_us,"workset_mb":ws_mb
            })

    # Save raw JSON
    json_path = outdir / f"{name}_raw.json"
    with json_path.open("w") as f: json.dump(results, f, indent=2)

    # Build data -> filter NaNs/negatives; drop sizes with no valid datapoints
    by_size: Dict[int, List[float]] = {}
    for rec in results["records"]:
        val = rec["rpg_us"]
        if np.isfinite(val) and val > 0:
            by_size.setdefault(rec["nx"], []).append(val)

    sizes_sorted = [n for n in sorted(set(rec["nx"] for rec in results["records"])) if n in by_size]
    if not sizes_sorted:
        fig, ax = plt.subplots(figsize=(10, 3)); ax.axis("off")
        ax.text(0.5, 0.5, f"No valid data for {name} (all runs failed).",
                ha="center", va="center", fontsize=12)
        fig.savefig(outdir / f"{name}_boxplot.png", dpi=160, bbox_inches="tight"); plt.close(fig)
        return results

    data = [by_size[n] for n in sizes_sorted]
    positions = np.array([working_set_mb(n, n, nz) for n in sizes_sorted], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        data, positions=positions, widths=positions*0.08,
        manage_ticks=False, showfliers=True, patch_artist=False,
    )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Working set size [MB]"); ax.set_ylabel("Runtime / gridpoint [µs]")
    ax.grid(True, which="both")
    ax.set_xticks(positions, [f"{p:.2g}" for p in positions])

    skipped = sorted(set(rec["nx"] for rec in results["records"]) - set(sizes_sorted))
    if skipped:
        ax.text(0.01, 0.01, "Skipped sizes (no valid runs): " + ", ".join(map(str, skipped)),
                transform=ax.transAxes, fontsize=9, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(outdir / f"{name}_boxplot.png", dpi=160); plt.close(fig)
    return results

def main():
    args = parse_args()
    # Robust defaults so just run 'python stencil_bench.py' possible
    threads = args.threads if args.threads > 0 else detect_threads()
    numba_layer = args.numba_threading_layer  # may be None; we choose 'workqueue' in make_child_env

    outdir = Path(args.outdir) if args.outdir else Path(f"bench_{int(time.time())}")
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {"args": vars(args) | {"threads_effective": threads,
                                     "numba_layer_effective": numba_layer or "workqueue(default)"},
               "programs": {}}

    for prog in args.programs:
        print(f"==> Running {prog} ...", flush=True)
        res = bench_program(
            prog, sizes=args.sizes, nz=args.nz, iters=args.iters, reps=args.reps,
            halo=args.halo, use_srun=args.srun, extra_cli=args.extra, outdir=outdir,
            threads=threads, numba_layer=numba_layer, extra_env_kv=args.env
        )
        summary["programs"][prog] = {"records": res["records"]}

    with (outdir / "summary.json").open("w") as f: json.dump(summary, f, indent=2)
    print(f"Done. Results saved to: {outdir.resolve()}")
    print("Generated per-program files: *_raw.json, *_boxplot.png")

if __name__ == "__main__":
    main()
