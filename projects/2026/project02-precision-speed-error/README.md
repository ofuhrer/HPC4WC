# HPC4WC Project - Precision, speed, error growth [2026]

### By: Haoanqin Gao, Jonas Hermann, Rochelle Leung

**Included in this repository:**
- Project report (project_report.pdf)
- C Source code (src/burgers.c)
- Python script for benchmarking and simulation output (run_sweep.py)
- Figures from project report (figures/)
- A few animations compiled from simulation output (animations/)
- Recorded benchmark result (reference_timings.csv)

**To run the benchmark:**
```bash
python run_sweep.py 0
```

**To output the computed u,v fields at 0.01 second interval:**
```bash
python run_sweep.py 2
```

**To output only the initial and final u,v fields:**
```bash
python run_sweep.py 1
```

Feel free to edit run_sweep.py to adjust the precisions and domain sizes that should be run!