from yasmin.analysis.domains import (
    ExecutionDomain,
    Interval,
    infer_execution_domain,
)


def test_infer_execution_domain() -> None:
    halo = ((2, 1), (0, 3))

    domain = infer_execution_domain(halo)

    assert domain == ExecutionDomain(
        intervals=(
            Interval(lower=2, upper_trim=1),
            Interval(lower=0, upper_trim=3),
        )
    )
