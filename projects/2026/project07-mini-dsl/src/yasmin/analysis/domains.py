# src/yasmin/analysis/domains.py

from dataclasses import dataclass

from yasmin.analysis.halos import Halo


@dataclass(frozen=True, slots=True)
class Interval:
    lower: int
    upper_trim: int


@dataclass(frozen=True, slots=True)
class ExecutionDomain:
    intervals: tuple[Interval, ...]


def infer_execution_domain(halo: Halo) -> ExecutionDomain:
    return ExecutionDomain(
        intervals=tuple(Interval(lower=left, upper_trim=right) for left, right in halo)
    )
