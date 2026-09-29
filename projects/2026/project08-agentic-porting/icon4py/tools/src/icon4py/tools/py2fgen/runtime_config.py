# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import collections
import contextlib
import os
from types import TracebackType

from gt4py import eve


def _env_flag_to_bool(name: str, default: bool) -> bool:
    """Recognize true or false signaling string values."""
    flag_value = None
    if name in os.environ:
        flag_value = os.environ[name].lower()
    match flag_value:
        case None:
            return default
        case "0" | "false" | "off":
            return False
        case "1" | "true" | "on":
            return True
        case _:
            raise ValueError(
                f"Invalid value {flag_value!r} for environment variable {name!r}: "
                "use '0 | false | off' or '1 | true | on'."
            )


def _env_to_strenum[T: eve.StrEnum](name: str, enum_type: type[T], default: T) -> T:
    """Read an enum value from an environment variable (with checking)."""
    value = os.environ.get(name, default).upper()
    if value not in enum_type.__members__:
        allowed_values = ", ".join(f"'{m}'" for m in enum_type.__members__)
        raise ValueError(
            f"Invalid value '{value}' for '{name}', allowed values are {allowed_values}."
        )
    return enum_type(value)


PROFILING: bool = _env_flag_to_bool("PY2FGEN_PROFILING", False)
"""Enable profiling for the PY2FGEN generated bindings."""


class Py2fgenLogLevels(eve.StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


LOG_LEVEL: str = _env_to_strenum("PY2FGEN_LOG_LEVEL", Py2fgenLogLevels, Py2fgenLogLevels.WARNING)
"""Set the log level for the PY2FGEN generated bindings."""


def _split_and_strip(s: str | None, sep: str = ",") -> list[str]:
    if not s:
        return []
    return [stripped_part for part in s.split(sep) if (stripped_part := part.strip())]


EXTRA_CALLABLES: list[str] = _split_and_strip(os.environ.get("PY2FGEN_EXTRA_CALLABLES", ""))
"""
Extra Python callables to be loaded and executed when the PY2FGEN bindings are initialized.

The Callables should be
- of the form `Callable[[], None]`,
- and specified as a comma-separated list of strings in the format ('my.path.to.module:object')
which will be loaded with `pkgutil.resolve_name` and executed.
"""

# CUSTOMIZATION HOOKS


class _NoopContextManager(contextlib.AbstractContextManager):
    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        pass


HOOK_BINDINGS_FUNCTION: dict[str, contextlib.AbstractContextManager] = collections.defaultdict(
    _NoopContextManager
)
"""Hook to register a ContextManager per function that wraps the function. The default is a no-op."""
