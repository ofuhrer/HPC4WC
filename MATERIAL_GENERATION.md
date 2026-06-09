# Course Material Generation

The course days use `day<n>/.master` as the source of truth. From that source,
`tools/generate_from_master.py` generates two views:

- `day<n>/`: the student-facing material.
- `day<n>/solutions/`: the visible solution bundle, generated locally when
  needed and not committed during course preparation.

## Source Files

For text source files such as `.py`, `.cpp`, `.F90`, and `.sh`, shared code is
written normally. Sections that differ between student and solution output use
`hpc4wc` markers:

```python
# hpc4wc:student-begin
# hpc4wc:student | value = None  # TODO
# hpc4wc:student-end
# hpc4wc:solution-begin
value = compute_value()
# hpc4wc:solution-end
```

Student lines must use `hpc4wc:student |`. The generator strips the comment
prefix and emits the payload after the marker. Solution blocks are copied as
normal code. Markers cannot be nested.

The student marker works with common comment styles (`#`, `!`, `//`) so the same
mechanism can be used for Python, Fortran, C++, and shell scripts.

Unmarked regular files in `.master` are solution-only. Shared files that should
also appear in the solution bundle are usually symlinked from `.master` to the
tracked student file in the day root.

## Notebooks

Master notebooks may use `hpc4wc` cell metadata:

```json
{
  "hpc4wc": {
    "student_source": ["x = None  # TODO\n"],
    "student_cell_type": "code",
    "student": "remove"
  }
}
```

Supported keys:

- `student_source`: replaces the cell source in the generated student notebook.
- `student_cell_type`: changes the generated student cell type, for example from
  `raw` or `markdown` to `code`.
- `student: "remove"`: removes the cell from the generated student notebook.

At notebook top level, `hpc4wc.student = "remove"` makes the whole notebook
solution-only. Generated student notebooks have outputs and execution counts
stripped. Generated student and solution notebooks have `hpc4wc` metadata
removed.

## Editing Metadata In JupyterLab

To edit cell metadata in JupyterLab:

1. Open the notebook in JupyterLab.
2. Select the cell whose metadata you want to edit.
3. Open the right sidebar Property Inspector / Notebook Tools panel.
4. Expand the cell metadata editor and edit the JSON under the `hpc4wc` key.
5. Save the notebook.

If the sidebar is hidden, enable it from the View menu or use the command
palette and search for the property inspector or notebook tools command. For
top-level notebook metadata, use the notebook metadata editor in the same tools
area or edit the `.ipynb` JSON directly.

## Generator Commands

Run commands from the repository root:

```bash
python tools/generate_from_master.py --student --force day5
python tools/generate_from_master.py --solution --force day5
python tools/generate_from_master.py --student --check day5
python tools/generate_from_master.py --solution --check day5
```

From inside a day directory, use `.` as the day argument:

```bash
python ../tools/generate_from_master.py --student --force .
```

Defaults:

- Without `--student` or `--solution`, both views are generated.
- `--student` generates the student tree and removes a matching `solutions/`
  directory.
- `--solution` generates only `day<n>/solutions/`.
- `--check` verifies that generated files are current without writing them.
- `--force` allows overwriting generated targets and removing stale solution
  files.

`tools/check_material.py` runs the generation check as part of the material
checks. From the repository root, `python tools/check_material.py` checks all
present `day<n>` directories. From inside a day directory, running the script
without arguments checks that day.

## Development Workflow

Edit `.master` first. Then regenerate and validate:

```bash
python tools/generate_from_master.py --student --force day5
python tools/generate_from_master.py --student --check day5
python tools/check_material.py day5
```

When preparing visible solutions, generate `day<n>/solutions/` locally, inspect
or publish it, then remove it before committing:

```bash
python tools/generate_from_master.py --solution --force day5
rm -rf day5/solutions
```

When reviewing changes, check all three views conceptually:

- `.master`: the maintainable source of truth.
- `day<n>/`: generated student material, including TODO scaffolds.
- `day<n>/solutions/`: generated visible solution bundle, created locally for
  validation or publication.

For notebook exercises where students copy code into scripts, keep the notebook
cell source and the corresponding script source formatted consistently. If the
student notebook uses `student_source` metadata, update that metadata too;
otherwise regeneration will undo edits made only to the generated notebook.
