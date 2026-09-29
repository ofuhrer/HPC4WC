# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import abc
import datetime
from typing import Protocol, TypeVar

from icon4py.model.common.states import model


Ins = TypeVar("Ins", bound=str)
Outs = TypeVar("Outs", bound=str)


class Component(Protocol[Ins, Outs]):
    """
    Protocol for model components.

    Components are the building blocks of the model.
    They operate on model state variables and transform them or produce new state variables.

    Each component should declare its inputs and outputs in terms of name, expected units and dimensions.

    Components are Callables upon __call__ that get passed a reference for the model state and
    select their needed inputs from there.

    Actual component definitions only need to implement this interface.

    Examples:

        >>> RequiredInputs: TypeAlias = Literal["foo", "bar"]
        >>> ProducedOutputs: TypeAlias = Literal["baz"]
        >>> class AlphaComponent(Component[RequiredInputs, ProducedOutputs]):
        ...     inputs_properties = {
        ...         "foo": {"standard_name": "Foo", "units": "m"},
        ...         "bar": {"standard_name": "BarBar", "units": "s"},
        ...     }
        ...
        ...     outputs_properties = {"baz": {"standard_name": "BAZ", "units": "s"}}
        ...
        ...     def __call__(
        ...         self, state: dict[RequiredInputs, model.DataField], time_step: datetime.datetime
        ...     ) -> dict[ProducedOutputs, model.DataField]: ...


    TODO(halungge): add more consistency checks.
     - check for mathching units and provide a hook for unit conversion for the components implementations
     - check for consistency of dimensions of state and input_properties

    """

    @property
    @abc.abstractmethod
    def inputs_properties(self) -> dict[Ins, model.FieldMetaData]:
        """Return a dictionary with the properties of the inputs expected by the component.

        Each input key contains metadata with the standard CF name, units and dimension of the associated data field.
        """
        ...

    @property
    @abc.abstractmethod
    def outputs_properties(self) -> dict[Outs, model.FieldMetaData]:
        """Return a dictionary with the properties of the outputs expected by the component.

        Each input key contains metadata with the standard CF name, units and dimension of the associated data field.

        TODO(halungge): is this too generic and we should split into separate properties for the different types of outputs: like
            tendencies, diagnostics, prognostics, etc?
            Are they different? or are they just different in the way they are used later on
            and how they are applied to the model state? Should this be made explicit in the interface?
            Along the same lines how should we track the time the produced values are valid for?
        """
        ...

    def __str__(self) -> str:
        return (
            f"instance of {self.__class__}(Component) uses inputs: "
            f"{self.inputs_properties.keys()} \n "
            f"produces : {self.outputs_properties.keys()}"
        )

    def __call__(
        self, state: dict[Ins, model.DataField], time_step: datetime.datetime
    ) -> dict[Outs, model.DataField]:
        """
        Runs the component on the input fields and returns the output fields.

        This function *must* be implemented with the real logic of the component.

        TODO(halungge): is it possible to improve this interface not haveing to pass on the entire state for example?

        Args:
            state: Model state dictionary.
            time_step: Current simulation time.
        Returns:
            A dictionary of output quantities.
        Raises:
              IncompleteStateError if the input_properties are not in the state.
        """
