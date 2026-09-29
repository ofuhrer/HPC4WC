# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    data_provider,
    decomposition_info,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    iau_wgt_dyn,
    icon_grid,
    interpolation_savepoint,
    is_iau_active,
    istep_exit,
    istep_init,
    linit,
    metrics_savepoint,
    process_props,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init,
    savepoint_dycore_30_to_38_exit,
    savepoint_dycore_30_to_38_init,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_init,
    savepoint_nonhydro_step_final,
    savepoint_velocity_exit,
    savepoint_velocity_init,
    savepoint_vertically_implicit_dycore_solver_init,
    step_date_exit,
    step_date_init,
    substep_exit,
    substep_init,
)
