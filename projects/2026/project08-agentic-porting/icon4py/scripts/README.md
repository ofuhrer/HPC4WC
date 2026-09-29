# scripts/ — Dev-Scripts Toolbox

Conventions for standardized project development scripts using Python, Bash, and (possibly) other recipes. All scripts should be executable and testable standalone, but a single entry point (`./scripts/run`) is provided as an extra feature.

## Using the unified CLI to run scripts

```bash
# Show all available commands
./scripts/run --help

# Run any command
./scripts/run my-command --option arg
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — Python scripts (including the single entry point) use a special `uv` shebang (more or less: `uv` `uv run --isolated --only-group scripts`) so all dependencies declared in the scripts dependency group (e.g. `typer`) are installed automatically into an ephemeral isolated environment.
- [bats-core](https://github.com/bats-core/bats-core) — only needed to run Bash tests.

## Layout

```
scripts/
├── run                      # Single entry point (uv shebang)
│
├── python/                  # Python scripts (importable package)
│   ├── __init__.py
│   ├── helpers/             # Shared helpers
│   │   ├── __init__.py
│   │   └── common.py        # Common definitions (e.g. paths)
│   │
│   └── <...>.py             # Script: contains a `cli = typer.Typer()` global symbol
│
├── sh/                      # Bash scripts
│   ├── _lib.sh              # Shared functions (sourced, not executed)
│   ├── _<...>.sh            # Utility: helper functions, not a script
│   ├── <...>.sh             # Script: setup-env sub-command
│
├── tests/
│   ├── python/              # Python scripts tests
│   │   ├── conftest.py      # Shared pytest fixtures
│   │   └── test_<...>.py    # Command test
│   └── sh/
│       └── test_lib.bats    # Bash library tests (bats-core)
│
└── data/                    # Files needed by the scripts
```

## Conventions

| Convention    | Meaning                                                                         |
| ------------- | ------------------------------------------------------------------------------- |
| `_` prefix    | Shared or private infrastructure, **not** a sub-command (`_lib.sh`, ...)        |
| `python/*.py` | Each (non-prefixed) python module defining a `cli` typer app is auto-discovered |
| `sh/*.sh`     | Each non-prefixed script is auto-discovered and wrapped as a Typer sub-command  |

## Adding a New Sub-Command

### Python

1. Create a new `python/my_tool.py` module
   - use the same *she-bang* as the `run` script
   ```bash
   #!/usr/bin/env -S uv run -q --frozen --isolated --python 3.12 --only-group scripts python3
   ```
   - it should define a global `cli` Typer app
   ```python
   cli = typer.Typer(...)
   ```
   - Call the `cli` app when running as a standalone script
   ```python
    if __name__ == "__main__":
        sys.exit(cli())
   ```
2. If needed, import shared helper modules from `helpers`.
3. If possible, add tests in `tests/python/test_my_tool.py`.

Note: avoid very expensive imports in the global scope to minimize startup time. You can either import locally inside functions or use global lazy imports with [`lazy-loader`](https://github.com/scientific-python/lazy-loader?tab=readme-ov-file#external-libraries) (e.g.`lazy.load('library')`)

### Bash

1. Create `sh/my_tool.sh` (it will be auto-discovered).
   - use `[help]` marker in comments to customize the CLI help message
   ```bash
    #!/usr/bin/env bash
    # [help] Remove common build/cache artifacts from the repo tree.
   ```
2. Source `_lib.sh` at the top for shared helpers if needed.
3. Add tests in `tests/sh/test_my_tool.bats`.

## Testing scripts

```bash
# Python tests
./scripts/test python

# Bash tests
./scripts/test sh
```
