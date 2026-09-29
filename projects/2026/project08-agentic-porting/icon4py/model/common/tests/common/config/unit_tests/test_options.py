# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import typing

import pytest

from icon4py.model.common.config import options


@dataclasses.dataclass
class ConfigClass:
    """A configuration class for testing."""

    choice: typing.Annotated[
        int,
        options.ConfigOption(
            description="A choice of methods.",
            icon_equivalent=options.IconOption(name="isomchce", path=("nested_1", "nested_2")),
        ),
    ]
    flag: typing.Annotated[
        bool,
        options.ConfigOption(
            description="A configuration flag.",
            icon_equivalent=options.IconOption(name="lsomflg", path=(), list_to_value=True),
        ),
    ]
    other: typing.Annotated[int, options.ConfigOption(description="Non-ICON config choice.")] = 5


def test_config_option_from_annotated_type_hint() -> None:
    class TesteeConfig:
        testee: typing.Annotated[int, options.ConfigOption(description="Just for testing.")]

    result = options.ConfigOption.from_type_hint(
        typing.get_type_hints(TesteeConfig, include_extras=True)["testee"]
    )
    assert result


def test_config_option_from_unannotated_type_hint_fails() -> None:
    class TesteeConfig:
        testee: int

    with pytest.raises(options.MissingConfigOptionAnnotationError):
        options.ConfigOption.from_type_hint(
            typing.get_type_hints(TesteeConfig, include_extras=True)["testee"]
        )


def test_config_option_from_wrongly_annotated_type_hint_fails() -> None:
    class TesteeConfig:
        no_option: typing.Annotated[int, "Just for testing."]
        more_than_one_option: typing.Annotated[
            bool,
            options.ConfigOption(description="foo"),
            options.ConfigOption(description="bar"),
        ]

    with pytest.raises(options.MissingConfigOptionAnnotationError):
        options.ConfigOption.from_type_hint(
            typing.get_type_hints(TesteeConfig, include_extras=True)["no_option"]
        )

    with pytest.raises(options.MultipleOptionAnnotationsError):
        options.ConfigOption.from_type_hint(
            typing.get_type_hints(TesteeConfig, include_extras=True)["more_than_one_option"]
        )


def test_iter_config_options_from_config_class() -> None:
    @dataclasses.dataclass
    class TesteeConfig:
        testee_choice: typing.Annotated[int, options.ConfigOption(description="Just for testing.")]
        testee_flag: typing.Annotated[bool, options.ConfigOption(description="Just for testing.")]

    result = dict(options.ConfigOption.iter_from_config_class(TesteeConfig))

    assert "testee_choice" in result
    assert "testee_flag" in result


def test_iter_pairs_from_icon() -> None:
    result = dict(
        options.iter_pairs_from_icon(
            config_cls=ConfigClass,
            icon_config={
                "nested_1": {"nested_2": {"isomchce": 42}},
                "lsomflg": [False, False, False],
            },
        )
    )

    assert result["choice"] == 42
    assert not result["flag"]


def test_construct_config_from_icon() -> None:
    result = options.construct_config_from_icon(
        config_cls=ConfigClass,
        icon_config={
            "nested_1": {"nested_2": {"isomchce": 42}},
            "lsomflg": [False, False, False],
        },
        other=3,
    )
    assert result.choice == 42
    assert result.flag is False
    assert result.other == 3


def test_default_conversion() -> None:
    @dataclasses.dataclass
    class TesteeConfig:
        testee: typing.Annotated[
            str,
            options.ConfigOption(
                description="Just for testing.",
                icon_equivalent=options.IconOption(name="testee", path=()),
            ),
        ]

    result = options.construct_config_from_icon(config_cls=TesteeConfig, icon_config={"testee": 42})
    assert result.testee == "42"


def test_custom_conversion() -> None:
    @dataclasses.dataclass
    class TesteeConfig:
        testee: typing.Annotated[
            str,
            options.ConfigOption(
                description="Just for testing.",
                icon_equivalent=options.IconOption(
                    name="testee",
                    path=(),
                    converter=lambda v: "is true" if v else "is false",
                ),
            ),
        ]

    assert (
        options.construct_config_from_icon(
            config_cls=TesteeConfig, icon_config={"testee": True}
        ).testee
        == "is true"
    )
    assert (
        options.construct_config_from_icon(
            config_cls=TesteeConfig, icon_config={"testee": False}
        ).testee
        == "is false"
    )
