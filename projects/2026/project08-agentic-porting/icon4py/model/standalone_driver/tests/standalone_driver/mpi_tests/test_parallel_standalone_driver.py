# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import pathlib

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common import model_backends, time
from icon4py.model.common.decomposition import definitions as decomp_defs, mpi_decomposition
from icon4py.model.standalone_driver import config as driver_config, driver_utils, standalone_driver
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions as test_defs,
    grid_utils,
    parallel_helpers,
    test_utils,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    download_ser_data,
    experiment_description,
    process_props,
)


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)

_log = logging.getLogger(__file__)


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "experiment_description, end_of_simulation",
    [
        pytest.param(
            test_defs.Experiments.JW,
            time.NumTimeSteps(1),
            marks=[pytest.mark.level("integration")],
            id="integration",
        ),
        pytest.param(
            test_defs.Experiments.JW,
            time.RelativeTime(days=7),
            marks=[pytest.mark.level("validation")],
            id="validation",
        ),
    ],
)
def test_standalone_driver_compare_single_multi_rank(  # noqa: PLR0917 [too-many-positional-arguments]
    download_ser_data: None,
    experiment_description: test_defs.ExperimentDescription,
    end_of_simulation: time.EndOfSimulation,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    _run_standalone_driver_compare_single_multi_rank(
        experiment_description=experiment_description,
        end_of_simulation=end_of_simulation,
        tmp_path=tmp_path,
        process_props=process_props,
        backend=backend,
    )


def _run_standalone_driver_compare_single_multi_rank(
    experiment_description: test_defs.ExperimentDescription,
    end_of_simulation: time.EndOfSimulation,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    if experiment_description.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    if model_backends.is_cpu_backend(backend) and test_utils.is_gtfn_backend(backend):
        atol = 1e-12
        rtol = 1e-14
    else:
        atol = 1e-10
        rtol = 0.0
    atol, rtol = test_utils.get_mpi_comparison_tolerance(backend, atol=atol, rtol=rtol)

    _log.info(
        f"running on {process_props.comm} with {process_props.comm_size} ranks and atol = {atol}, rtol = {rtol}"
    )

    allocator = model_backends.get_allocator(backend)

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_experiment_config_from_fortran(config_file_path)

    single_rank_process_props = decomp_defs.SingleNodeProcessProperties()
    single_rank_config = config.with_overrides(
        driver={
            "output_path": tmp_path / f"ci_driver_output_serial_rank_{process_props.rank}",
            "end_of_simulation": end_of_simulation,
        }
    )
    single_rank_grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=single_rank_config.vertical_grid,
        allocator=allocator,
        process_props=single_rank_process_props,
    )
    single_rank_ds, _ = standalone_driver.run_driver(
        config=single_rank_config,
        grid_manager=single_rank_grid_manager,
        process_props=single_rank_process_props,
        backend=backend,
    )

    multi_rank_config = config.with_overrides(
        driver={
            "output_path": tmp_path / f"ci_driver_output_mpi_rank_{process_props.rank}",
            "end_of_simulation": end_of_simulation,
        }
    )
    multi_rank_grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=multi_rank_config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    multi_rank_ds, multi_rank_driver = standalone_driver.run_driver(
        config=multi_rank_config,
        grid_manager=multi_rank_grid_manager,
        process_props=process_props,
        backend=backend,
    )

    fields = ["vn", "w", "exner", "theta_v", "rho"]
    single_rank_reference_fields: dict[str, object] = {
        field_name: getattr(single_rank_ds.prognostics.current, field_name).asnumpy()
        for field_name in fields
    }

    for field_name in fields:
        print(f"\nverifying field {field_name}")
        global_reference_field = process_props.comm.bcast(
            single_rank_reference_fields.get(field_name),
            root=0,
        )
        local_field = getattr(multi_rank_ds.prognostics.current, field_name)
        dim = local_field.domain.dims[0]
        parallel_helpers.check_local_global_field(
            decomposition_info=multi_rank_driver.decomposition_info,
            process_props=process_props,
            dim=dim,
            global_reference_field=global_reference_field,
            local_field=local_field.asnumpy(),
            check_halos=True,
            atol=atol,
            rtol=rtol,
        )
