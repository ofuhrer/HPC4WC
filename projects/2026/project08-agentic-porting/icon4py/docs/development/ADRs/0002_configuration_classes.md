---
tags: [config, fortran, icon]
---

# [Configuration Classes]

- **Status**: valid
- **Authors**: Rico Häuselmann (@DropD)
- **Created**: 2026-06-17
- **Updated**: 2026-06-23

While making ICON4Py user-configurable, facing code duplication between modules, we decided to provide a standard way to declaratively encode common attributes of configuration options alongside utilities that can then be reused across modules.

## Context

At the time of writing each module has a class which contains its configuration. Each of these classes knows how to construct itself from a dictionary that represents an ICON namelist. The mapping is contained implicitly in a class method. Usually the class attributes that have an equivalent in ICON are annotated with documentation comments (where to find in ICON namelists and what it does).

Given the mapping for each configuration option in each of those classes, the logic is the same for all of them.

Extrapolating from this case of duplication we find the same for:

- describing configuration options to users (currently only documentation comments exist for this)
- validation (currently partially done in monolithic, imperative validation methods)

## Decision

The chosen pattern is for configuration classes to look as follows

```python
import dataclasses
import typing

from icon4py.model.common.config import options as common_conf_opt


@dataclasses.dataclass
class MyModuleConfig:
    some_configuration_option: typing.Annotated[
        int,
        common_conf_opt.ConfigOption(
            description="Describe what `some_configuration_option` does.",
            icon_equivalent=common_conf_opt.IconOption(
                name="isomcfgop",
                path=("diffusion_nml",),  # for nested sections: ("toplevel_nml", "sublevel_nml")
                read_from_icon=True,  # optional (default: True), whether to actually read this from ICON namelists
                list_to_value=False,  # optional (default: False), whether to map a list of values in ICON namelists to a single value
            ),
        ),
    ] = 4  # default value
```

The pattern is meant to be extended in the future as the need arises. One example would be validation:

```python
def validate_some_configuration_option_with_some_flag(
    some_configuration_option: int, other_option: bool
) -> None:
    if not other_option:
        raise ConfigValidationError(
            f"'some_configuration_option' should only be set if 'other_option' is True."
        )


...


@dataclasses.dataclass
class MyModuleConfig:
    some_configuration_option: typing.Annotated[
        int,
        ConfigOption(
            ...,
            validators=(
                DependsOn("some_flag", condition=validate_some_configuration_option_with_some_flag),
                Min(0),
                Max(59000),
                ...,
            ),
        ),
    ] = 4
```

Another possible extension could be decoupling configuration choice enums from fortran ICON equivalents:

```python
# model.atmosphere.diffusion.diffusion
# CURRENTLY


class DiffusionType(int, enum.Enum):
    """
    Order of nabla operator for diffusion.

    Note: Called `hdiff_order` in `mo_diffusion_nml.f90`.
    Note: We currently only support type 5.
    """

    NO_DIFFUSION = -1  #: no diffusion
    LINEAR_2ND_ORDER = 2  #: 2nd order linear diffusion on all vertical levels
    SMAGORINSKY_NO_BACKGROUND = 3  #: Smagorinsky diffusion without background diffusion
    LINEAR_4TH_ORDER = 4  #: 4th order linear diffusion on all vertical levels
    SMAGORINSKY_4TH_ORDER = 5  #: Smagorinsky diffusion with fourth-order background diffusion


@dataclasses.dataclass(kw_only=True)
class DiffusionConfig:
    ...

    def _validate(self) -> None:
        """Apply consistency checks and validation on configuration parameters."""
        if self.diffusion_type != DiffusionType.SMAGORINSKY_4TH_ORDER:
            raise NotImplementedError(
                "Only diffusion type 5 = `Smagorinsky diffusion with fourth-order background "
                "diffusion` is implemented"
            )
```

This encodes five choices, only one of which is actually implemented. Also, the integer values for each choice are tied to the fortran ICON equivalent. Instead this could become:

```python
class DiffusionType(int, enum.Enum):
    SMAGORINSKY_4TH_ORDER = enum.auto()


def unkown_icon_diffusion_type(icon_option: common_conf_opt.IconOption, value: int) -> typing.Never:
    raise NotImplementedError(f"value {icon_diffusion_type} for "{icon_option.name} is not supported in ICON4Py.")


@dataclasses.dataclass(kw_only=True)
class DiffusionConfig:

    diffusion_type: typing.Annotated[
        DiffusionType,
        common_conf_opt.ConfigOption(
            description="Order of Nabla operator for diffusion.",
            icon_equivalent=common_conf_opt.IconOption(
                name="hdiff_order",
                path=("diffusion_nml",),
                value_map={5: DiffusionType.SMAGORINSKY_4TH_ORDER},
                unmapped_value_callback=unkown_icon_diffusion_type
            ),
        ),
    ] = DiffusionType.SMAGORINSKY_4TH_ORDER
    ...
```

## Consequences

Code for mapping from ICON inputs to ICON4Py configurations can be shared between modules.

It becomes possible to automatically provide helpful information for all available configuration options.

Different user facing utilities for configuring / inspecting ICON4Py runs will not have to duplicate logic for

- mapping to / from ICON (fortran)
- validation
- documentation
- etc

The current solution will not achieve maximum separation between internals and their mapping to and from ICON concepts. However it provides a structured path towards it.

Another downside is a loss of static type checking when using the provided utilities. This is considered minor, because the provided utilities are only envisioned to be used when dealing with user input. Example:

```python
# This might occur in a test
diff_cfg = icon4py.common.config.options.construct_config_from_icon(
    DiffusionConfig, {"diffusion_nml": {"itype_vn_diffu": 1}}
)  # no static type checking can be performed here, because the mapping logic is run-time only
```

As opposed to hand-mapping, where type checkers can infer the required types at least to some degree.

```python
# More likely usage, where static type checking is not a concern
icon_config: dict[str, Any] = read_namelists()
diffusion_config = construct_config_from_icon(icon_config)
```

## Alternatives Considered

### Separating (fortran-) ICON mapping information from the rest

It is considered a long term goal to isolate all the logic and data used to map between fortran ICON and ICON4Py concepts, such as configuration options from the ICON4Py implementation of those concepts.
The decision made here was to prioritize solving the code duplication issue. However, we think this solution will not block the long term goal.

- Pro: allow for free evolution of ICON4Py internals quicker, decoupled from ICON
- Con: takes longer, less (for now) helpful information immediately available

The declarative annotation system provides a clear path towards maximum decoupling. All the instances of `IconOption` can be moved to a separate dictionary, keyed by the name (or class AND name) of the option it belongs to. This would have to be done once per ICON4Py module.

Example:

```python
class DiffusionConfig:
    diffusion_type: Annotated[DiffusionType, ConfigOption(description="...", validation=...)]
    ...
```

And in `icon4py.model.atmosphere.diffusion.fortran_icon_mappings`:

```python
CONFIC_OPTIONS: {DiffusionConfig: {"diffusion_type": ...}}
```

To arrive at the same usability, `icon4py.common.fortan_icon_mappings` could contain some infrastructure to find the relevant mapping for each configuration class (via module path heuristics or entry points or similar), so that a different version of `icon4py.common.config.options.construct_config_from_icon` could be provided.

From there the mapping should be decoupled enough to be refactored without touching the configuration classes.

### Use `dataclasses.field` instead of `typing.Annotated`

```python
class MyModuleConfig:
    some_configuration_option: int = dataclasses.field(
        metadata={"config_option": ConfigOption(...)}, default=4
    )
```

A wrapper around `dataclasses.field` could improve the readability of this option.

This option is functionally equivalent. The one difference is that `typing.Annotated` is lazily evaluated, while `dataclasses.field` is not. It was not chosen because syntactically it groups the annotations with the default value as opposed to the type.

## References

- [Python Documentation for `typing.Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated)
