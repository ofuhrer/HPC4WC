"""Reduces a benchmark job log to a table of median runtimes.

The log alternates header lines of the form
`### v=VARIANT mode=MODE order=N size=NX iters=M rep=R` with the data rows
printed by stencil2d. Runtimes are normalized to milliseconds per 1024
iterations so that constant-work runs at different sizes are comparable.

Usage:
    parse_bench.py JOB_OUTPUT.out

Authors:
    Stefanie Boersig <stefanie.boersig@env.ethz.ch>
    Boaz Ko <boazko@student.ethz.ch>
    Ben Bullinger <ben.bullinger@inf.ethz.ch>
"""

import collections
import re
import statistics
import sys

_HEADER = re.compile(
    r"### v=(\S+) mode=(\S+) order=(\d+) size=(\d+) iters=(\d+) rep=(\d+)")
_VARIANTS = ("filter", "filter-kblock", "filter-inline")
_MODES = ("off", "lim", "bfree")
_TIME_COLUMN = 5
_REFERENCE_ITERATIONS = 1024


def parse(path):
    """Groups the runtimes in a job log by run configuration.

    Args:
        path: Path to the job output file.

    Returns:
        A dict mapping (size, order, variant, mode) to a list of runtimes
        in milliseconds per 1024 iterations.
    """
    runs = collections.defaultdict(list)
    header = None
    with open(path, encoding="utf-8") as log:
        for line in log:
            match = _HEADER.match(line)
            if match:
                variant, mode, order, size, iters, _ = match.groups()
                header = (int(size), int(order), variant, mode, int(iters))
            elif line.startswith("[") and header:
                seconds = float(
                    line.split(",")[_TIME_COLUMN].split("]")[0].strip())
                size, order, variant, mode, iters = header
                runs[(size, order, variant, mode)].append(
                    seconds * _REFERENCE_ITERATIONS / iters * 1000)
    return runs


def report(runs):
    """Prints one row per (size, order, variant), one column per mode."""
    print(f"{'size':>5} {'n':>2} {'variant':<14} | "
          + " | ".join(f"{mode:>10}" for mode in _MODES))
    for size in sorted({key[0] for key in runs}):
        for order in sorted({key[1] for key in runs}):
            for variant in _VARIANTS:
                cells = []
                for mode in _MODES:
                    times = runs.get((size, order, variant, mode))
                    cells.append(f"{statistics.median(times):10.1f}"
                                 if times else " " * 10)
                print(f"{size:>5} {order:>2} {variant:<14} | "
                      + " | ".join(cells))
            print()


def main():
    """Parses the log named on the command line and prints the table."""
    report(parse(sys.argv[1]))


if __name__ == "__main__":
    main()
