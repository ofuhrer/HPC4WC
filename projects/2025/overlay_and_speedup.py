import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_root(argpath: str|None) -> Path:
    if argpath:
        p = Path(argpath)
        if p.is_dir(): return p
        print(f"[ERROR] Not a directory: {p}", file=sys.stderr); sys.exit(1)
    # try local bench_* dirs, newest first
    cands = sorted(Path(".").glob("bench_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in cands:
        if any((c / f"{name}_raw.json").exists() for name in ["numpy","numba","torch","jax"]):
            return c
    print("[ERROR] Could not find a bench directory. Pass it explicitly: python overlay_and_speedup.py ./bench_YYYY", file=sys.stderr)
    sys.exit(1)

def load_data(p: Path):
    data = json.loads(p.read_text())
    recs = data["records"]
    by_n = {}
    for r in recs:
        v = r.get("rpg_us", np.nan)
        if np.isfinite(v) and v > 0:
            by_n.setdefault(r["nx"], []).append(v)
    sizes = sorted(by_n)
    if not sizes: return None
    med = np.array([np.median(by_n[n]) for n in sizes], float)
    ws  = np.array([ (n*n*recs[0]["nz"]*3*4)/1e6 for n in sizes ], float)  # MB
    return np.array(sizes), ws, med

def main(argpath=None):
    root = find_root(argpath)
    print("Using bench dir:", root)
    series = {}
    for name in ["numpy","numba","torch","jax"]:
        f = root / f"{name}_raw.json"
        if f.exists():
            loaded = load_data(f)
            if loaded: series[name]=loaded
        else:
            print(f"[WARN] Missing {f}, skipping {name}")
    if not series:
        print("[ERROR] No *_raw.json files with data in", root, file=sys.stderr); sys.exit(1)

    # Overlay medians
    fig, ax = plt.subplots(figsize=(8,5))
    for name,(sizes,ws,med) in series.items():
        ax.plot(ws, med, marker='o', label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Working set size [MB]"); ax.set_ylabel("Median runtime / gridpoint [µs]")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout(); fig.savefig(root / "overlay_medians.png", dpi=160)
    print("Wrote:", root / "overlay_medians.png")

    # Speedup vs NumPy @ largest common size
    if "numpy" not in series or len(series) < 2:
        print("[WARN] Need NumPy + another framework for speedup bars; skipping."); return
    common = set(series["numpy"][0])
    for nm in series.keys():
        common = common.intersection(set(series[nm][0]))
    if not common:
        print("[WARN] No common sizes across frameworks; skipping speedup bars."); return

    N = max(common)
    def med_at(name, N):
        sizes,_,med = series[name]; idx = list(sizes).index(N); return med[idx]
    base = med_at("numpy", N)
    others = [nm for nm in series.keys() if nm != "numpy"]
    speedups = [base/med_at(nm, N) for nm in others]

    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(others, speedups)
    ax.set_ylabel(f"Speedup vs NumPy @ nx=ny={N}")
    for i,v in enumerate(speedups):
        ax.text(i, v*1.02, f"{v:.2f}×", ha="center", va="bottom")
    ax.set_ylim(0, max(speedups)*1.15)
    fig.tight_layout(); fig.savefig(root / "speedup_vs_numpy.png", dpi=160)
    print("Wrote:", root / "speedup_vs_numpy.png")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else None)
