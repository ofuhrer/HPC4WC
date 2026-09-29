# Porting the ICON 2-moment Microphysics to GT4Py

## Setup

You can use [uv](https://docs.astral.sh/uv/) to manage a single virtual
environment that lives in the repo root (`.venv/`) and is accessible from
anywhere in the tree. It bundles `numpy` and the vendored `gt4py` (the `next`
DSL, installed editable from the local `gt4py/` folder).

With uv installed, from the repo root run:

```bash
uv sync
```

This creates `.venv/` and installs everything from the committed `uv.lock`,
reproducing the exact same environment on any machine. uv also downloads a
compatible Python (≥3.10) automatically if you don't have one. The variants are
installed editable as part of that sync, so `satad_only`, `minimal_mcrph`,
`complete_mcrph` and `mcrph_common` are importable from any working directory.

Then run commands through uv (no manual activation needed):

```bash
uv run pytest                                 # every variant's tests
uv run python -m satad_only.column_driver     # the GT4Py satad driver
uv run python -m minimal_mcrph.column_driver  # the GT4Py minimal microphysics driver
```

or activate the environment directly with `source .venv/bin/activate`.

Running the fortran column model is indicated in more detail in the respective subfolder's readmes. In short, all that needs to be done is to navigate to the respective fortran source directory (see structure outline below) and issue `make run`. 

## structure

Each microphysics variant lives in its own top-level folder, which is set up as a Python
package for that variant: the GT4Py port sits at the top level of the folder, beside
the `fortran/` reference it was ported from, the `example/` column they share, and
the `tests/` that hold the two to each other. Each variant's `__init__.py` exposes
`FORTRAN_DIR` and `EXAMPLE_DIR`, anchored to the file rather than to the working
directory — use those instead of relative paths so everything runs from anywhere.

`mcrph_common/` holds the parts every variant needs in the same form: the column-CSV
reader/writer, the Fortran build-and-run plumbing, and the comparison helper.

The variants, in increasing order of complexity:

- `satad_only` contains fortran and GT4Py versions of the saturation adjustment routine in isolation - this is a starting point for efforts and can be verified against the existing port of the single-moment microphysics.
- `minimal_mcrph` contains a stripped-back version of the microphysics that does away with the hard-to-port precipitation and riming microphysics - i.e., a scheme that deals only with activation and depositional/condensational growth/evaporation of hydrometeors. This saves us having to deal with gamma function lookups and the vertical solver for precip sedimentation.
- `complete_mcrph` contains a standalone version of the original Fortran Seifert and Beheng (2006) two-moment microphysics scheme as implemented in ICON.


## Rules
- If an agent makes a plan for porting in a folder, that folder should contain an `agents_plan` directory which we copy plans made by our agents into
- create a pull request every time you have completed a step - e.g. when there is a plan for how to port, when a skeleton of a test harness is there, etc.
- general notes e.g. about how to work with GT4Py should go into the base directory so agents in all folders can see them
- If GT4Py creates files at runtime that do not need to be committed/would pollute our git (such as .gt_cache/), add them to our .gitignore. Look at the gt4py and icon4py .gitignores in the online repos for reference (they are deleted from the versions cloned into here to avoid confusion with our own .gitignore).