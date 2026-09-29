# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Time-related type aliases (and more) shared across the model."""

import datetime
import time as _std_time
from typing import TypeAlias


RelativeTime: TypeAlias = datetime.timedelta  # noqa: UP040
AbsoluteTime: TypeAlias = datetime.datetime  # noqa: UP040
NumTimeSteps: TypeAlias = int  # noqa: UP040
EndOfSimulation: TypeAlias = RelativeTime | AbsoluteTime | NumTimeSteps  # noqa: UP040

# Re-exported from the standard library so callers use a single time module
localtime = _std_time.localtime
