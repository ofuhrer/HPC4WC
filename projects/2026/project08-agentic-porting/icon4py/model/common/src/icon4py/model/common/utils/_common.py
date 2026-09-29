# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import abc
import copy
import functools
from collections.abc import Callable, Generator
from typing import ClassVar, Concatenate, Final, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class DoubleBuffering(Protocol):
    @abc.abstractmethod
    def swap_buffers(self) -> None: ...


T = TypeVar("T")
C = TypeVar("C")


class named_property[C, T](property):
    """
    A simple extension of the built-in `property` descriptor storing
    the name of the attribute it is assigned to.

    The name is stored in the `name` attribute of the property instance.

    Examples:
        >>> class A:
        ...     @named_property
        ...     def value(self) -> int:
        ...         return self._value
        ...
        ...     @value.setter
        ...     def value(self, value: int) -> None:
        ...         self._value = value
        >>> a = A()
        >>> a.value = 1
        >>> print(A.value.name)
        value
    """

    name: str | None = None

    def __set_name__(self, owner: C, name: str) -> None:
        """
        Set the name of the attribute the property is assigned to.

        Note that if the same descriptor is assigned to multiple attributes
        or if it is reused in other classes, only the first assigned name
        is stored.
        """
        if self.name is None:
            self.name = name

    def getter(self: named_property[C, T], fget: Callable[[C], T]) -> named_property[C, T]:
        standard_property = super().getter(fget)
        result = type(self)(
            fget=standard_property.fget, fset=standard_property.fset, fdel=standard_property.fdel
        )
        result.name = getattr(self, "name", None)
        return result

    def setter(self: named_property[C, T], fset: Callable[[C, T], None]) -> named_property[C, T]:
        standard_property = super().setter(fset)
        result = type(self)(
            fget=standard_property.fget, fset=standard_property.fset, fdel=standard_property.fdel
        )
        result.name = getattr(self, "name", None)
        return result

    def deleter(self: named_property[C, T], fdel: Callable[[C], None]) -> named_property[C, T]:
        standard_property = super().deleter(fdel)
        result = type(self)(
            fget=standard_property.fget, fset=standard_property.fset, fdel=standard_property.fdel
        )
        result.name = getattr(self, "name", None)
        return result

    def __copy__(self) -> named_property[C, T]:
        result = type(self)(self.fget, self.fset, self.fdel, self.__doc__)
        result.name = self.name
        return result


class Pair[T]:  # noqa: PLW1641 [eq-without-hash]
    """
    Generic class representing a pair of values.

    The name of the pair attributes can be customized by defining new
    descriptors in the subclasses.

    See the examples below.

    Examples:
        >>> class MyPair(Pair[T]):
        ...     a: T = Pair.first
        ...     b: T = Pair.frozen_second
        >>> pair = MyPair(1, 2)
        >>> print(pair)
        MyPair(a=1, b=2)

        >>> pair.swap()
        MyPair(a=2, b=1)

        >>> pair == MyPair(2, 1)
        True

        >>> pair.a = -1
        >>> pair.a
        -1

        >>> pair.b = 3
        Traceback (most recent call last):
        ...
        AttributeError: can't set attribute

    """

    _FIRST_ACCESSOR_ID: Final = "FIRST"
    _SECOND_ACCESSOR_ID: Final = "SECOND"

    __first_attr_name: ClassVar[str] = "first"
    __second_attr_name: ClassVar[str] = "second"

    def __init_subclass__(cls) -> None:
        for key, value in {**cls.__dict__}.items():
            if (attr_id := getattr(value, "_pair_accessor_id_", None)) is not None:
                assert isinstance(value, named_property)
                if key != value.name:
                    # If one of the original descriptors from the `Pair` class has been
                    # directly assigned to another class member with a different name
                    # (likely in a subclass) instead of creating a proper copy, it is
                    # copied and initialized here with the right name.
                    descriptor_copy = copy.copy(value)
                    descriptor_copy.name = key
                    setattr(cls, key, descriptor_copy)
                if attr_id == Pair._FIRST_ACCESSOR_ID:
                    cls.__first_attr_name = key
                elif attr_id == Pair._SECOND_ACCESSOR_ID:
                    cls.__second_attr_name = key
                else:
                    raise TypeError(f"Invalid '{key}' pair accessor descriptor: {value}")

    __first: T
    __second: T

    def __init__(self, first: T, second: T, /) -> None:
        self.__first = first
        self.__second = second

    @named_property
    def first(self) -> T:
        """Property descriptor for the first element of the pair."""
        return self.__first

    @first.setter  # type: ignore[no-redef]
    def first(self, value: T) -> T:
        self.__first = value

    @named_property
    def second(self) -> T:
        """Property descriptor for the second element of the pair."""
        return self.__second

    @second.setter  # type: ignore[no-redef]
    def second(self, value: T) -> T:
        self.__second = value

    @named_property
    def frozen_first(self) -> T:
        """Read-only property descriptor for the first element of the pair (mainly for subclassing)."""
        return self.__first

    @named_property
    def frozen_second(self) -> T:
        """Read-only property descriptor for the second element of the pair (mainly for subclassing)."""
        return self.__second

    if hasattr(frozen_first, "_pair_accessor_id_") and hasattr(frozen_second, "_pair_accessor_id_"):
        first._pair_accessor_id_ = frozen_first._pair_accessor_id_ = _FIRST_ACCESSOR_ID
        second._pair_accessor_id_ = frozen_second._pair_accessor_id_ = _SECOND_ACCESSOR_ID

    def __eq__(self, other: object) -> bool:
        assert hasattr(other, "__first") and hasattr(other, "__second")
        return type(self) is type(other) and (
            self.__first == other.__first and self.__second == other.__second
        )

    # `__hash__` is implicitly set to None when `__eq__` is redefined, so instances are not hashable.

    def __iter__(self) -> Generator[T, None, None]:
        yield self.__first
        yield self.__second

    def __repr__(self) -> str:
        first_name = type(self).__first_attr_name
        second_name = type(self).__second_attr_name
        return f"{self.__class__.__name__}({first_name}={self.__first!r}, {second_name}={self.__second!r})"

    def swap(self: Pair[T]) -> Pair[T]:
        """
        Swap the values of the first and second attributes of the instance.

        Returns:
            The instance with swapped values (for fluent interfaces).
        """
        self.__first, self.__second = self.__second, self.__first
        return self

    swap_buffers = swap


class PredictorCorrectorPair(Pair[T]):
    predictor = Pair.first
    corrector = Pair.second


class TimeStepPair(Pair[T]):
    current: named_property = Pair.frozen_first
    next: named_property = Pair.frozen_second


def chainable[T, **P](
    method_fn: Callable[Concatenate[T, P], None],
) -> Callable[Concatenate[T, P], T]:
    """
    Make an instance method return the actual instance so it can be used in a chain of calls.

    Typically used for simple fluent interfaces.

    Examples:
        >>> class A:
        ...     @chainable
        ...     def set_value(self, value: int) -> None:
        ...         self.value = value
        ...
        ...     @chainable
        ...     def increment(self, value: int) -> None:
        ...         self.value += value

        >>> a = A()
        >>> a.set_value(1).increment(2)  # doctest:+ELLIPSIS
        <__main__.A object at ...>

        >>> a.value
        3
    """

    @functools.wraps(method_fn)
    def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> T:
        method_fn(self, *args, **kwargs)
        return self

    return wrapper
