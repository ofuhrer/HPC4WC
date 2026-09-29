# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from icon4py.model.common.grid import gridfile


def revert_repeated_index_to_invalid(offset: np.ndarray):
    num_elements = offset.shape[0]
    for i in range(num_elements):
        # convert repeated indices back into -1
        for val in np.flip(offset[i, :]):
            if np.count_nonzero(val == offset[i, :]) > 1:
                unique_values, counts = np.unique(offset[i, :], return_counts=True)
                rep_values = unique_values[counts > 1]
                rep_indices = np.where(np.isin(offset[i, :], rep_values))[0]
                offset[i, rep_indices[1:]] = gridfile.GridFile.INVALID_INDEX
    return offset
