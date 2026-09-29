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

from icon4py.model.common import initial_condition, model_backends, model_options
from icon4py.model.common.decomposition import definitions as decomp_defs, mpi_decomposition
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
)
from icon4py.model.standalone_driver import (
    config as driver_config,
    driver_states,
    driver_utils,
    standalone_driver,
)
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
    experiment,
    experiment_description,
    process_props,
)


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)

_log = logging.getLogger(__file__)


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize(
    "experiment_description",
    [
        test_defs.Experiments.JW,
        test_defs.Experiments.EXCLAIM_APE_AES,
    ],
)
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_initial_conditions_compare_single_multi_rank(  # noqa: PLR0917 [too-many-positional-arguments]
    experiment: test_defs.Experiment,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend_like: model_backends.BackendLike,
    backend: gtx_typing.Backend,
    download_ser_data: None,
) -> None:
    if experiment.description.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    atol = 0.0 if model_backends.is_cpu_backend(backend_like) else 2e-11
    # NOTE: actually vn, w, exner, theta_v, rho have delta = 0.0 also on
    # distributed GPU, u,v do not because they are computed with RBF

    if model_backends.is_cpu_backend(backend_like) and test_utils.is_gtfn_backend(
        model_options.customize_backend(program=None, backend=backend_like)
    ):
        # NOTE: we use gtfn_cpu to check that single and multirank give bitwise
        # identical results, as gtfn_cpu is deterministic. In this case we can
        # set atol = 0.0 and check for bitwise identical results. In theory
        # dace_cpu should also be deterministic, but it results in 1.8e-14
        # delta on vn and 4.3e-19 on w on the initial condition. We haven't
        # investigated this yet.
        # atol = 0.0 has been relaxed with rtol = 1e-16 because on torus grid
        # global sum/avg reductions result in ~2e-16 roundoff errors, so atol =
        # 0.0 is too strict.
        rtol = 1e-15
        atol = 0.0
    else:
        rtol = 0.0
        atol = 2e-11

    _log.info(
        f"running on {process_props.comm} with {process_props.comm_size} ranks and atol = {atol}, rtol = {rtol}"
    )

    allocator = model_backends.get_allocator(backend)

    grid_file_path = grid_utils._download_grid_file(experiment.description.grid)

    single_rank_process_props = decomp_defs.SingleNodeProcessProperties()
    single_rank_config = experiment.config.with_overrides(
        driver={"output_path": tmp_path / f"ci_driver_output_serial_rank_{process_props.rank}"}
    )
    single_rank_grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=single_rank_config.vertical_grid,
        allocator=allocator,
        process_props=single_rank_process_props,
    )
    single_rank_icon4py_driver: standalone_driver.Icon4pyDriver = (
        standalone_driver.initialize_driver(
            config=single_rank_config,
            grid_manager=single_rank_grid_manager,
            process_props=single_rank_process_props,
            backend=backend,
        )
    )

    single_rank_prognostic = prognostics.initialize_prognostic_state(
        grid=single_rank_icon4py_driver.grid,
        allocator=allocator,
        tracer_config=single_rank_icon4py_driver.config.tracer_config,
    )
    initial_condition.create(
        config=single_rank_icon4py_driver.config.initial_condition,
        vertical_config=single_rank_icon4py_driver.config.vertical_grid,
        grid=single_rank_icon4py_driver.grid,
        static_fields=single_rank_icon4py_driver.static_field_factories,
        prognostic_state_now=single_rank_prognostic,
        backend=single_rank_icon4py_driver.backend,
        exchange=single_rank_icon4py_driver.exchange,
        global_reductions=single_rank_icon4py_driver.global_reductions,
    )
    single_rank_diagnostic = diagnostics.initialize_diagnostic_state(
        grid=single_rank_icon4py_driver.grid, allocator=allocator
    )
    single_rank_ds: driver_states.DriverStates = driver_states.assemble_driver_states(
        grid=single_rank_icon4py_driver.grid,
        allocator=allocator,
        backend=single_rank_icon4py_driver.backend,
        exchange=single_rank_icon4py_driver.exchange,
        static_fields=single_rank_icon4py_driver.static_field_factories,
        prognostic_state_now=single_rank_prognostic,
        diagnostic_state=single_rank_diagnostic,
        experiment_config=single_rank_icon4py_driver.config,
    )

    multi_rank_config = experiment.config.with_overrides(
        driver={"output_path": tmp_path / f"ci_driver_output_mpi_rank_{process_props.rank}"}
    )
    multi_rank_grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=multi_rank_config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    multi_rank_icon4py_driver: standalone_driver.Icon4pyDriver = (
        standalone_driver.initialize_driver(
            config=multi_rank_config,
            grid_manager=multi_rank_grid_manager,
            process_props=process_props,
            backend=backend,
        )
    )

    multi_rank_prognostic = prognostics.initialize_prognostic_state(
        grid=multi_rank_icon4py_driver.grid,
        allocator=allocator,
        tracer_config=multi_rank_icon4py_driver.config.tracer_config,
    )
    initial_condition.create(
        config=multi_rank_icon4py_driver.config.initial_condition,
        vertical_config=multi_rank_icon4py_driver.config.vertical_grid,
        grid=multi_rank_icon4py_driver.grid,
        static_fields=multi_rank_icon4py_driver.static_field_factories,
        prognostic_state_now=multi_rank_prognostic,
        backend=multi_rank_icon4py_driver.backend,
        exchange=multi_rank_icon4py_driver.exchange,
        global_reductions=multi_rank_icon4py_driver.global_reductions,
    )
    multi_rank_diagnostic = diagnostics.initialize_diagnostic_state(
        grid=multi_rank_icon4py_driver.grid, allocator=allocator
    )
    multi_rank_ds: driver_states.DriverStates = driver_states.assemble_driver_states(
        grid=multi_rank_icon4py_driver.grid,
        allocator=allocator,
        backend=multi_rank_icon4py_driver.backend,
        exchange=multi_rank_icon4py_driver.exchange,
        static_fields=multi_rank_icon4py_driver.static_field_factories,
        prognostic_state_now=multi_rank_prognostic,
        diagnostic_state=multi_rank_diagnostic,
        experiment_config=multi_rank_icon4py_driver.config,
    )

    fields_to_check: list[tuple[str, object, object]] = [
        (name, single_rank_ds.prognostics.current, multi_rank_ds.prognostics.current)
        for name in ("vn", "w", "exner", "theta_v", "rho")
    ] + [(name, single_rank_ds.diagnostic, multi_rank_ds.diagnostic) for name in ("u", "v")]

    # Moist experiments (e.g. APE) initialize the water-vapour tracer
    if single_rank_ds.prognostics.current.tracer.qv is not None:
        fields_to_check.append(
            (
                "qv",
                single_rank_ds.prognostics.current.tracer,
                multi_rank_ds.prognostics.current.tracer,
            )
        )

    for field_name, serial_source, local_source in fields_to_check:
        print(f"verifying field {field_name}")
        global_reference_field = process_props.comm.bcast(
            getattr(serial_source, field_name).asnumpy(),
            root=0,
        )
        local_field = getattr(local_source, field_name)
        parallel_helpers.check_local_global_field(
            decomposition_info=multi_rank_icon4py_driver.decomposition_info,
            process_props=process_props,
            dim=local_field.domain.dims[0],
            global_reference_field=global_reference_field,
            local_field=local_field.asnumpy(),
            check_halos=True,
            atol=atol,
            rtol=rtol,
        )
