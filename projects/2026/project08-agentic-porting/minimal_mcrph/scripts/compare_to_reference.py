"""Print per-field differences between the GT4Py port and the real Fortran.

Same comparison the test suite makes, but printed as a table instead of asserted, for
when you want to see how far apart things are rather than just pass/fail.

The Fortran is built and run at invocation, so the numbers are current -- this no longer
reads the committed ``example/output_fields.csv``, which was written at 9 significant
digits and so put a ~1e-9 floor under every comparison made against it.

Usage (from anywhere in the repo, with the uv env synced):
    uv run python minimal_mcrph/scripts/compare_to_reference.py
    uv run python minimal_mcrph/scripts/compare_to_reference.py --backend gtfn_cpu
    uv run python minimal_mcrph/scripts/compare_to_reference.py --scenario deep_cold
    uv run python minimal_mcrph/scripts/compare_to_reference.py --stages
"""

from __future__ import annotations

import argparse

import numpy as np

from minimal_mcrph.column_driver import BACKENDS as _BACKENDS
from minimal_mcrph.driver import ColumnState, Driver
from minimal_mcrph.tests.conftest import CHECKED_FIELDS, RTOL, as_columns
from minimal_mcrph.tests.fortran_runner import STAGES, ensure_driver_built, run_fortran
from minimal_mcrph.tests.scenarios import SCENARIOS

_STATE_FIELDS = tuple(
    f.name for f in ColumnState.__dataclass_fields__.values() if f.name != "hhl"
)


def _table(title, got, ref, fields=CHECKED_FIELDS):
    print(f"\n{title}")
    print(f"{'field':<8}{'max_abs_diff':>16}{'max_rel_err':>16}{'status':>10}")
    worst = 0.0
    all_ok = True
    for name in fields:
        mine, theirs = np.asarray(got[name]), np.asarray(ref[name])
        abs_diff = np.abs(mine - theirs)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(theirs != 0.0, abs_diff / np.abs(theirs), 0.0)
        ok = bool(np.all(rel <= RTOL))
        worst = max(worst, float(np.nanmax(rel)))
        all_ok &= ok
        print(
            f"{name:<8}{abs_diff.max():>16.3e}{np.nanmax(rel):>16.3e}"
            f"{'ok' if ok else 'FAIL':>10}"
        )
    print(f"worst relative deviation: {worst:.3e}  (rtol={RTOL:g})")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(_BACKENDS), default="embedded")
    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS) + ["all"], default="bundled"
    )
    parser.add_argument(
        "--stages", action="store_true",
        help="also break the comparison down by process boundary",
    )  # fmt: skip
    args = parser.parse_args()

    ensure_driver_built()
    backend = _BACKENDS[args.backend]
    names = sorted(SCENARIOS) if args.scenario == "all" else [args.scenario]

    all_ok = True
    for name in names:
        fields, hhl, dt = SCENARIOS[name]
        reference, fortran_stages = run_fortran(fields, hhl, dt=dt, want_stages=True)

        state = ColumnState(hhl=hhl, **{n: fields[n].copy() for n in _STATE_FIELDS})
        port_stages = {}
        result = Driver(backend=backend).run_timestep(state, dt=dt, stages=port_stages)

        all_ok &= _table(f"=== {name} / {args.backend} / final ===", as_columns(result), reference)

        if args.stages:
            for stage in STAGES:
                all_ok &= _table(
                    f"--- {name} / {stage} ---", port_stages[stage], fortran_stages[stage]
                )

    print()
    print("ALL FIELDS MATCH" if all_ok else "MISMATCH -- see FAIL rows above")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
