# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from typing import Any, Final, Literal, TypeGuard

from gt4py.eve import Node, codegen
from gt4py.eve.codegen import JinjaTemplate as as_jinja

from icon4py.tools.py2fgen import _definitions, _utils


# rc convention (chosen so init-fail is distinguishable from success — cffi
# forces the result slot to 0 on init-fail, see _cffi_start_and_call_python
# in _embedding.h of cffi):
#   0 -> embedding init failed
#   1 -> success
#   2 -> Python wrapper raised an exception
CFFI_DECORATOR = "@ffi.def_extern(error=2)"

BUILTIN_TO_ISO_C_TYPE: Final[dict[_definitions.ScalarKind, str]] = {
    _definitions.FLOAT64: "real(c_double)",
    _definitions.FLOAT32: "real(c_float)",
    _definitions.BOOL: "logical(c_bool)",
    _definitions.INT32: "integer(c_int)",
    _definitions.INT64: "integer(c_long)",
}
BUILTIN_TO_CPP_TYPE: Final[dict[_definitions.ScalarKind, str]] = {
    _definitions.FLOAT64: "double",
    _definitions.FLOAT32: "float",
    # NVHPC writes Fortran `.TRUE.` as `-1` (byte 0xFF), which cffi's `_Bool` validator
    # rejects ("got a _Bool of value 255, expected 0 or 1"). Use `unsigned char` on the
    # C side (same 1-byte ABI as `_Bool` and `logical(c_bool)`) to bypass that check.
    _definitions.BOOL: "unsigned char",
    _definitions.INT32: "int",
    _definitions.INT64: "long",
}


def is_array(param: _definitions.ParamDescriptor) -> TypeGuard[_definitions.ArrayParamDescriptor]:
    return isinstance(param, _definitions.ArrayParamDescriptor)


class Func(Node):
    name: str
    module_name: str
    args: dict[str, _definitions.ArrayParamDescriptor | _definitions.ScalarParamDescriptor]


class BindingsLibrary(Node):
    library_name: str
    functions: list[Func]


def to_c_type(scalar_type: _definitions.ScalarKind) -> str:
    """Convert a scalar type to its corresponding C++ type."""
    return BUILTIN_TO_CPP_TYPE[scalar_type]


def to_iso_c_type(scalar_type: _definitions.ScalarKind) -> str:
    """Convert a scalar type to its corresponding ISO C type."""
    return BUILTIN_TO_ISO_C_TYPE[scalar_type]


def as_f90_value(param: _definitions.ParamDescriptor) -> Literal["value", None]:
    """
    Return the Fortran 90 'value' keyword for scalar types.

    Args:
        param: The function parameter to check.

    Returns:
        A string containing 'value' for scalar types, otherwise an empty string.
    """
    return "value" if not is_array(param) else None


def render_c_pointer(param: _definitions.ParamDescriptor) -> Literal["*", ""]:
    """Render a C pointer symbol for array types."""
    return "*" if is_array(param) else ""


def render_fortran_array_dimensions(
    param: _definitions.ParamDescriptor, assumed_size_array: bool
) -> str | None:
    """
    Render Fortran array dimensions for array types.

    Args:
        param: The function parameter to check.

    Returns:
        A string representing Fortran array dimensions.
    """
    if is_array(param):
        if assumed_size_array:
            return "dimension(*)"
        else:
            dims = ",".join(":" for _ in range(param.rank))
        return f"dimension({dims})"

    return None


class PythonWrapperGenerator(codegen.TemplatedGenerator):
    def visit_BindingsLibrary(self, node: BindingsLibrary, **kwargs: Any) -> str:
        def render_size_args_tuple(name: str, param: _definitions.ArrayParamDescriptor) -> str:
            size_args = ",".join(f"{_size_arg_name(name, i)}" for i in range(param.rank))
            return f"({size_args},)"

        def render_params(func: Func) -> str:
            params = []
            for name, param in func.args.items():
                params.append(name)
                if is_array(param):
                    params.extend(_size_arg_name(name, i) for i in range(param.rank))
            params.append("on_gpu")
            return ", ".join(params)

        return self.generic_visit(
            node,
            ScalarKind=_definitions.ScalarKind,
            MemorySpace=_definitions.MemorySpace,
            is_array=is_array,
            render_size_args_tuple=render_size_args_tuple,
            render_params=render_params,
            cffi_decorator=CFFI_DECORATOR,
            **kwargs,
        )

    BindingsLibrary = as_jinja(
        """\
import pkgutil
from icon4py.tools.py2fgen import runtime_config
for callable_name in runtime_config.EXTRA_CALLABLES:
    pkgutil.resolve_name(callable_name)()

import logging
from {{ library_name }} import ffi
from icon4py.tools.py2fgen import _runtime, _conversion

logger = logging.getLogger(__name__)
log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
logging.basicConfig(
    level=getattr(logging, runtime_config.LOG_LEVEL),
    format=log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# embedded function imports
{% for func in _this_node.functions -%}
from {{ func.module_name }} import {{ func.name }}
{% endfor %}

{% for func in _this_node.functions %}

{{ cffi_decorator }}
def {{ func.name }}_wrapper(
{{render_params(func)}}
):
    with runtime_config.HOOK_BINDINGS_FUNCTION["{{ func.name }}"]:
        try:
            if __debug__:
                logger.info("Python execution of {{ func.name }} started.")

            if __debug__:
                if runtime_config.PROFILING:
                    unpack_start_time = _runtime.perf_counter()

            # ArrayInfos
            {% for name, arg in func.args.items() %}
            {% if is_array(arg) %}
            {{ name }} = ({{ name }}, {{ render_size_args_tuple(name, arg) }}, {% if arg.memory_space == MemorySpace.HOST %}False{% else %}on_gpu{% endif %}, {{ arg.is_optional }})
            {% endif %}
            {% endfor %}

            if __debug__:
                if runtime_config.PROFILING:
                    allocate_end_time = _runtime.perf_counter()
                    logger.info('{{ func.name }} constructing `ArrayInfos` time: %s' % str(allocate_end_time - unpack_start_time))

                    func_start_time = _runtime.perf_counter()

            if __debug__ and runtime_config.PROFILING:
                perf_counters = {}
            else:
                perf_counters = None
            {{ func.name }}(
            ffi = ffi,
            perf_counters = perf_counters,
            {%- for name, arg in func.args.items() -%}
            {{ name }} = {{ name }}{{ "," }}
            {%- endfor -%}
            )

            if __debug__:
                if runtime_config.PROFILING:
                    func_end_time = _runtime.perf_counter()
                    logger.info('{{ func.name }} convert time: %s' % str(perf_counters["convert_end_time"] - perf_counters["convert_start_time"]))
                    logger.info('{{ func.name }} execution time: %s' % str(func_end_time - func_start_time))


            {% if func.args %}
            if __debug__:
                if logger.isEnabledFor(logging.DEBUG):
                    {% for name, arg in func.args.items() %}
                    {% if is_array(arg) %}
                    {{name}}_arr = _conversion.as_array(ffi, {{ name }}) if {{ name }} is not None else None
                    msg = 'shape of {{ name }} after computation = %s' % str({{ name}}_arr.shape if {{name}} is not None else "None")
                    logger.debug(msg)
                    msg = '{{ name }} after computation: %s' % str({{name}}_arr) if {{ name }} is not None else "None"
                    logger.debug(msg)
                    {% endif %}
                    {% endfor %}
            {% endif %}

            if __debug__:
                logger.info("Python execution of {{ func.name }} completed.")

        except Exception as e:
            logger.exception(f"A Python error occurred: {e}")
            return 2

    return 1
{% endfor %}
"""
    )


class CHeaderGenerator(codegen.TemplatedGenerator):
    BindingsLibrary = as_jinja("{{'\n'.join(functions)}}")

    def visit_Func(self, func: Func) -> str:
        params = []
        for name, param in func.args.items():
            params.append(self.visit_Parameter(name, param))
            if is_array(param):
                params.extend(f"int {_size_arg_name(name, i)}" for i in range(param.rank))
        params.append(f"{to_c_type(_definitions.BOOL)} on_gpu")

        rendered_params = ", ".join(params)
        return self.generic_visit(func, rendered_params=rendered_params)

    Func = as_jinja("extern int {{ name }}_wrapper({{rendered_params}});")

    def visit_Parameter(self, name: str, param: _definitions.ParamDescriptor, **kwargs: Any) -> str:
        return f"{to_c_type(param.dtype)}{render_c_pointer(param)} {name}"


def _render_parameter_declaration(name: str, attributes: Sequence[str | None]) -> str:
    return f"{','.join(attribute for attribute in attributes if attribute is not None)} :: {name}"


def _size_arg_name(name: str, i: int) -> str:
    return f"{name}_size_{i}"


def _size_param_declaration(name: str, value: bool = True) -> str:
    return f"integer(c_int){', value' if value else ''} :: {name}"


class FortranISOCBindingsGenerator(codegen.TemplatedGenerator):
    def visit_Func(self, func: Func, **kwargs: Any) -> str:
        param_names = []
        param_declarations = []
        for name, param in func.args.items():
            param_names.append(name)
            param_declarations.append(
                _render_parameter_declaration(
                    name=name,
                    attributes=[
                        "type(c_ptr)" if is_array(param) else to_iso_c_type(param.dtype),
                        "value",
                        "target",
                    ],
                )
            )
            if is_array(param):
                for i in range(param.rank):
                    size_name = _size_arg_name(name, i)
                    param_names.append(size_name)
                    param_declarations.append(_size_param_declaration(size_name))

        # on_gpu flag
        param_declarations.append(f"{to_iso_c_type(_definitions.BOOL)}, value :: on_gpu")
        param_names.append("on_gpu")

        param_names_str = ", &\n ".join(param_names)

        return self.generic_visit(
            func,
            as_func_declaration=True,
            param_names=param_names_str,
            param_declarations=param_declarations,
        )

    Func = as_jinja(
        """
function {{name}}_wrapper({{param_names}}) bind(c, name="{{name}}_wrapper") result(rc)
   import :: c_int, c_long, c_float, c_double, c_bool, c_ptr
   integer(c_int) :: rc  ! Stores the return code
   {% for param in param_declarations %}
   {{ param }}
   {% endfor %}
end function {{name}}_wrapper
    """
    )


class FortranBindingsFunctionGenerator(codegen.TemplatedGenerator):
    def visit_Func(self, func: Func, **kwargs: Any) -> str:
        def render_args(name: str, param: _definitions.ParamDescriptor) -> str:
            if is_array(param):
                if param.is_optional:
                    return f"{name} = {name}_ptr"
                else:
                    return f"{name} = c_loc({name})"
            return f"{name} = {name}"

        param_names = ", &\n ".join([name for name in func.args] + ["rc"])
        args = []
        for name, param in func.args.items():
            args.append(render_args(name, param))
            if is_array(param):
                args.extend(
                    f"{_size_arg_name(name, i)} = {_size_arg_name(name, i)}"
                    for i in range(param.rank)
                )

        # on_gpu flag
        args.append("on_gpu = on_gpu")
        compiled_arg_names = ", &\n".join(args)

        param_declarations = [
            _render_parameter_declaration(
                name=name,
                attributes=[
                    to_iso_c_type(param.dtype),
                    render_fortran_array_dimensions(param, False),
                    as_f90_value(param),
                    # arrays are passed to C via `c_loc`, which requires contiguity
                    "contiguous" if is_array(param) else None,
                    "intent(inout)" if is_array(param) else None,
                    "pointer" if is_array(param) and param.is_optional else "target",
                ],
            )
            for name, param in func.args.items()
        ]

        # on_gpu flag
        param_declarations.append(f"{to_iso_c_type(_definitions.BOOL)} :: on_gpu")

        def get_sizes_maker(name: str, param: _definitions.ArrayParamDescriptor) -> str:
            return "\n".join(
                f"{_size_arg_name(name, i)} = SIZE({name}, {i + 1})" for i in range(param.rank)
            )

        for name, param in func.args.items():
            if is_array(param):
                for i in range(param.rank):
                    size_name = _size_arg_name(name, i)
                    param_declarations.append(_size_param_declaration(size_name, value=False))

        return self.generic_visit(
            func,
            assumed_size_array=False,
            param_names=param_names,
            args_with_size_args=compiled_arg_names,
            non_optional_arrays=[
                name
                for name, param in func.args.items()
                if is_array(param)
                and not param.is_optional
                and param.memory_space == _definitions.MemorySpace.MAYBE_DEVICE
            ],
            optional_arrays=[
                name
                for name, param in func.args.items()
                if is_array(param)
                and param.is_optional
                and param.memory_space == _definitions.MemorySpace.MAYBE_DEVICE
            ],
            as_allocatable=True,
            param_declarations=param_declarations,
            to_iso_c_type=to_iso_c_type,
            render_fortran_array_dimensions=render_fortran_array_dimensions,
            get_sizes_maker=get_sizes_maker,
            is_array=is_array,
        )

    Func = as_jinja(
        """
subroutine {{name}}({{param_names}})
   use, intrinsic :: iso_c_binding
   {% for arg in param_declarations %}
   {{ arg }}
   {% endfor %}
   integer(c_int) :: rc  ! Stores the return code
   ! ptrs
   {% for name in optional_arrays %}
   type(c_ptr) :: {{ name }}_ptr
   {% endfor %}

   {% for name in optional_arrays %}
   {{ name }}_ptr = c_null_ptr
   {% endfor %}

   {%- for arr in non_optional_arrays %}
       !$acc host_data use_device({{ arr }})
   {%- endfor %}
   {%- for arr in optional_arrays %}
       !$acc host_data use_device({{ arr }}) if(associated({{ arr }}))
   {%- endfor %}
   
   #ifdef _OPENACC
   on_gpu = .True.
   #else
   on_gpu = .False.
   #endif

   {% for name, param in _this_node.args.items() if is_array(param) and not param.is_optional %}
     {{ get_sizes_maker(name, param) }}
   {% endfor %}

   {% for name, param in _this_node.args.items() if is_array(param) and param.is_optional %}
   if(associated({{ name }})) then
   {{ name }}_ptr = c_loc({{ name }})
     {{ get_sizes_maker(name, param) }}
    endif
   {% endfor %}

   rc = {{ name }}_wrapper({{ args_with_size_args }})

   {%- for arr in non_optional_arrays %}
   !$acc end host_data
   {%- endfor %}
   {%- for arr in optional_arrays %}
   !$acc end host_data
   {%- endfor %}
end subroutine {{name}}
    """
    )


class F90InterfaceGenerator(codegen.TemplatedGenerator):
    def visit_BindingsLibrary(self, bindings_library: BindingsLibrary, **kwargs: Any) -> str:
        bindings_functions = FortranBindingsFunctionGenerator.apply(bindings_library.functions)
        iso_c_bindings_functions = FortranISOCBindingsGenerator.apply(bindings_library.functions)
        return self.generic_visit(
            bindings_library,
            bindings_functions=bindings_functions,
            iso_c_bindings_functions=iso_c_bindings_functions,
        )

    BindingsLibrary = as_jinja(
        """\
module {{ _this_node.library_name }}
    use, intrinsic :: iso_c_binding
    implicit none

    {% for func in _this_node.functions %}
    public :: {{ func.name }}
    {% endfor %}

interface
    {{ '\n'.join(iso_c_bindings_functions) }}
end interface

contains
    {{ '\n'.join(bindings_functions) }}
end module
"""
    )


def generate_c_header(bindings_library: BindingsLibrary) -> str:
    """
    Generate C header code from the given plugin.

    Args:
        plugin: The BindingsLibrary instance containing information for code generation.

    Returns:
        Formatted C header code as a string.
    """
    generated_code = CHeaderGenerator.apply(bindings_library)
    return codegen.format_source("cpp", generated_code, style="LLVM")


def add_include_guard(c_header: str, library_name: str) -> str:
    """Wrap generated C header code in an ``#ifndef`` include guard."""
    # No sanitization needed: `library_name` is also emitted as a Fortran module
    # name and a Python import, so it is already a valid identifier.
    guard = f"{library_name.upper()}_H"
    return f"#ifndef {guard}\n#define {guard}\n\n{c_header}\n\n#endif\n"


def generate_python_wrapper(bindings_library: BindingsLibrary) -> str:
    """
    Generate Python wrapper code.

    Args:
        plugin: The BindingsLibrary instance containing information for code generation.

    Returns:
        Formatted Python wrapper code as a string.
    """
    generated_code = PythonWrapperGenerator.apply(bindings_library)
    return codegen.format_source("python", generated_code)


def generate_f90_interface(bindings_library: BindingsLibrary) -> str:
    """
    Generate Fortran 90 interface code.

    Args:
        plugin: The BindingsLibrary instance containing information for code generation.
    """
    generated_code = F90InterfaceGenerator.apply(bindings_library)
    return _utils.format_fortran_code(generated_code)
