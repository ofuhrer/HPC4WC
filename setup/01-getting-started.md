# Getting Started

These instructions set up the Python environment used by the HPC4WC course on CSCS Santis.

## Launch JupyterHub

1. Open <https://jupyter-santis.cscs.ch>.
2. Sign in with your CSCS account.
3. Choose the Santis JupyterLab environment with the `prgenv-gnu/26.3:v1` uenv.
4. Start the server and wait until JupyterLab opens.

JupyterHub sessions run inside a Slurm job on Santis compute nodes. This is different from `ssh santis`, which logs in to a login node. The login node is useful for repository work and Slurm submission, but GPU, CUDA, and `srun` behavior must be checked on compute nodes.

## Clone the Course Repository

Open a JupyterLab terminal and run:

```bash
cd "$HOME"
git clone https://github.com/ofuhrer/HPC4WC.git
```

If you want direct access to your scratch space from the file browser, create a symlink:

```bash
ln -sfn "$SCRATCH" "$HOME/scratch"
```

## Install the Course Environment

Run the setup script from the JupyterLab terminal:

```bash
cd "$HOME/HPC4WC"
./setup/HPC4WC_setup.sh
```

The script creates a Python virtual environment under `$SCRATCH`, symlinks it as `$HOME/HPC4WC_venv`, registers the `HPC4WC_kernel` Jupyter kernel, and validates the CPU-side Python stack. It also installs a small `srun` wrapper inside the virtual environment so nested `srun -n ...` commands and `%ipcluster start -n ...` work from the single-task JupyterHub allocation. It does not modify `.bashrc`.

If you need to recreate the environment, or if an older course setup already exists, run:

```bash
HPC4WC_FORCE=1 ./setup/HPC4WC_setup.sh
```

For terminals, activate the environment with:

```bash
source "$HOME/activate_hpc4wc.sh"
```

## Restart JupyterHub

After the setup finishes, stop and restart your JupyterHub server. Then open notebooks with the `HPC4WC_kernel` kernel.

## Test the Setup

Run the notebook:

```text
setup/02-test-setup.ipynb
```

It checks NumPy, Matplotlib, MPI through `ipyparallel`, CuPy, GT4Py, and the package versions expected for the course.

For a compute-node validation from a Santis login node, submit:

```bash
cd "$HOME/HPC4WC"
sbatch setup/validate_on_compute.sh
```

Validation logs are written below `$SCRATCH/HPC4WC_validation_logs`.
