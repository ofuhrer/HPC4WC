import numpy as np
import pytest
from baseline.stencil2d_cupy import main as reference_main
from custom_graph_capture import stencil2d_custom_unrolled_graph
from custom_kernel import (
    stencil2d_custom_v1,
    stencil2d_custom_v2,
    stencil2d_custom_v3,
    stencil2d_custom_v4,
    stencil2d_custom_v5,
    stencil2d_custom_v6,
    stencil2d_custom_v7,
    stencil2d_custom_v8,
)
from graph_capture import (
    stencil2d_memcpy_graph,
    stencil2d_two_graphs,
    stencil2d_unrolled_graph,
)

IMPLEMENTATIONS = [
    stencil2d_custom_unrolled_graph,
    stencil2d_memcpy_graph,
    stencil2d_two_graphs,
    stencil2d_unrolled_graph,
    stencil2d_custom_v1,
    stencil2d_custom_v2,
    stencil2d_custom_v3,
    stencil2d_custom_v4,
    stencil2d_custom_v5,
    stencil2d_custom_v6,
    stencil2d_custom_v7,
    stencil2d_custom_v8,
]

NUM_HALO = 2


@pytest.mark.parametrize(
    "implementation",
    [
        pytest.param(stencil2d_custom_unrolled_graph, id="graph-custom-unrolled"),
        pytest.param(stencil2d_memcpy_graph, id="graph-memcpy"),
        pytest.param(stencil2d_two_graphs, id="graph-double"),
        pytest.param(stencil2d_unrolled_graph, id="graph-unrolled"),
        pytest.param(stencil2d_custom_v1, id="custom-v1"),
        pytest.param(stencil2d_custom_v2, id="custom-v2"),
        pytest.param(stencil2d_custom_v3, id="custom-v3"),
        pytest.param(stencil2d_custom_v4, id="custom-v4"),
        pytest.param(stencil2d_custom_v5, id="custom-v5"),
        pytest.param(stencil2d_custom_v6, id="custom-v6"),
        pytest.param(stencil2d_custom_v7, id="custom-v7"),
        pytest.param(stencil2d_custom_v8, id="custom-v8"),
    ],
)
@pytest.mark.parametrize(
    "nz", [pytest.param(64, id="nz=64"), pytest.param(128, id="nz=128")]
)
@pytest.mark.parametrize(
    "domain_size",
    [
        pytest.param(8, id="8x8"),
        pytest.param(16, id="16x16"),
        pytest.param(24, id="24x24"),
        pytest.param(32, id="32x32"),
        pytest.param(64, id="64x64"),
    ],
)
@pytest.mark.parametrize(
    "num_iter",
    [
        pytest.param(128 + 1, id="num_iter=129"),
        pytest.param(256 + 1, id="num_iter=257"),
    ],
)
def test_correctness(implementation, domain_size, nz, num_iter):
    reference = reference_main(
        nx=domain_size,
        ny=domain_size,
        nz=nz,
        num_iter=num_iter,
        num_halo=NUM_HALO,
        plot_result=False,
        save_result=False,
        benchmark=False,
        return_result=True,  # We want to compare the results.
    )

    result = implementation.main(
        nx=domain_size,
        ny=domain_size,
        nz=nz,
        num_iter=num_iter,
        num_halo=NUM_HALO,
        plot_result=False,
        save_result=False,
        benchmark=False,
        return_result=True,
    )

    assert np.allclose(reference, result, atol=1e-07)
