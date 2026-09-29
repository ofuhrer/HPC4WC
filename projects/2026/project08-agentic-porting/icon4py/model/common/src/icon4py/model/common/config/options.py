# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import typing
from typing import Self

from icon4py.model.common.utils import fortran_config


class NotDataclassError(Exception): ...


class MissingConfigOptionAnnotationError(Exception): ...


class MultipleOptionAnnotationsError(Exception): ...


@dataclasses.dataclass
class IconOption:
    """
    Describe how to find the equivalent option in ICON namelists.
    """

    #: option name
    name: str
    #: full path through nested namelist sections
    path: tuple[str, ...]
    list_to_value: bool = False
    read_from_icon: bool = True
    converter: typing.Callable[[typing.Any], typing.Any] | None = None


@dataclasses.dataclass
class ConfigOption:
    description: str
    icon_equivalent: IconOption | None = None

    @classmethod
    def from_type_hint(cls: type[Self], annotated: typing.Any) -> Self:
        """
        Read a ConfigOption from a (pre-evaluated) type annotation.

        Example:

        >>> @dataclasses.dataclass
        >>> class SomeConfig:
        >>>     an_option: typing.Annotated[int, ConfigOption(description="test")] = 0
        >>>
        >>> ConfigOption.from_type_hint(
        ...     typing.get_type_hints(SomeConfig, include_extras=True)["an_option"]
        ... )
        ConfigOption(description='test', icon_equivalent=None)
        """
        if not hasattr(annotated, "__metadata__"):
            raise MissingConfigOptionAnnotationError(
                f"'{annotated}' is not an annotated type hint."
            )
        opts = [a for a in annotated.__metadata__ if isinstance(a, cls)]
        if not opts:
            raise MissingConfigOptionAnnotationError(
                f"'{annotated}' is missing a ConfigOption annotation."
            )
        if len(opts) > 1:
            raise MultipleOptionAnnotationsError(
                f"'{annotated}' is annotated with multiple ConfigOptions."
            )
        (opt,) = opts
        return opt

    @classmethod
    def iter_from_config_class(
        cls: type[Self], config_cls: type
    ) -> typing.Iterator[tuple[str, Self]]:
        """
        Iterate over ConfigOption annotations of a given configuration dataclass.

        Example:

        >>> @dataclasses.dataclass
        >>> class SomeConfig:
        >>>     an_option: typing.Annotated[int, ConfigOption(description="first")] = 0
        >>>     other_option: typing.Annotated[int, ConfigOption(description="second")] = 0
        >>>
        >>> dict(ConfigOption.iter_from_config_class(SomeConfig))
        {'an_option': ConfigOption(description='first', icon_equivalent=None), 'other_option': ConfigOption(description='second', icon_equivalent=None)}
        """
        if not dataclasses.is_dataclass(config_cls):
            raise NotDataclassError(str(config_cls))
        annotations = typing.get_type_hints(config_cls, include_extras=True)
        for field in dataclasses.fields(config_cls):
            yield field.name, cls.from_type_hint(annotations[field.name])


def iter_pairs_from_icon(
    config_cls: type, icon_config: dict[str, typing.Any]
) -> typing.Iterator[tuple[str, typing.Any]]:
    """
    Iter name-value pairs for config options, reading from a fortran ICON config dict.

    Example:

    >>> @dataclasses.dataclass
    >>> class ConfigClass:
    >>>     choice: typing.Annotated[
    >>>         int,
    >>>         options.ConfigOption(
    >>>             description="A choice of methods.",
    >>>             icon_equivalent=options.IconOption(
    >>>                 name="isomchce",
    >>>                 path=("nested",)
    >>>             )
    >>>         )
    >>>     ]
    >>> next(iter_pairs_from_icon(ConfigClass, {"nested": {"isomche": 1}}))
    ("choice", 1)
    """
    annotations = typing.get_type_hints(config_cls, include_extras=True)
    if not dataclasses.is_dataclass(config_cls):
        raise NotDataclassError(str(config_cls))
    for name, opt in ConfigOption.iter_from_config_class(config_cls):
        if opt.icon_equivalent and opt.icon_equivalent.read_from_icon:
            data = icon_config
            for subsection in opt.icon_equivalent.path:
                data = data[subsection]
            raw_value = data[opt.icon_equivalent.name]
            de_listified = (
                fortran_config.list_to_value(raw_value)
                if opt.icon_equivalent.list_to_value
                else raw_value
            )
            converter = opt.icon_equivalent.converter or annotations[name]
            yield name, converter(de_listified)


def construct_config_from_icon[T](
    config_cls: type[T], icon_config: dict[str, typing.Any], **overrides: typing.Any
) -> T:
    """
    Construct a configuration instance from a fortran ICON config dict.

    Example:

    >>> @dataclasses.dataclass
    >>> class ConfigClass:
    >>>     choice: typing.Annotated[
    >>>         int,
    >>>         options.ConfigOption(
    >>>             description="A choice of methods.",
    >>>             icon_equivalent=options.IconOption(
    >>>                 name="isomchce",
    >>>                 path=()
    >>>             )
    >>>         )
    >>>     ]
    >>> construct_config_from_icon(ConfigClass, {"isomche": 1})
    ConfigClass(choice=1)
    """
    return config_cls(
        **dict(iter_pairs_from_icon(config_cls, icon_config)),
        **overrides,
    )
