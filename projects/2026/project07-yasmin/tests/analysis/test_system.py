import os

import pytest

import yasmin.analysis.system as system


def test_available_cores_honors_omp_num_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: set(range(72)),
        raising=False,
    )

    assert system.available_cores() == 4
