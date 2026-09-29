# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import abc
import datetime
from typing import Any, Protocol


class Monitor(Protocol):
    """
    Monitor component of the model.

    Monitor is a base class for components that store or freeze state for later usage but
    don't modify it or return any new state objects.

    Named after Sympl Monitor component: https://sympl.readthedocs.io/en/latest/monitors.html
    """

    def __str__(self) -> str:
        return f"instance of {self.__class__}(Monitor)"

    @abc.abstractmethod
    def store(
        self, state: dict, model_time: datetime.datetime, *args: Any, **kwargs: dict[str, Any]
    ) -> None:
        """Store state and perform class specific actions on it.

        Args:
            state: dict  model state dictionary
            model_time: current simulation time
        """
        pass
