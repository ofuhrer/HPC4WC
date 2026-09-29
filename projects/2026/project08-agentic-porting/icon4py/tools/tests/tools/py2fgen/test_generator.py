# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import re
import subprocess
import tempfile

import pytest

from icon4py.tools.py2fgen import _codegen, _definitions, _generator

from tests.tools.py2fgen.wrappers import simple


def test_parse_functions_on_wrapper():
    plugin = _generator.get_cffi_description(
        [simple.square_from_function, simple.square_error], "square_plugin"
    )
    assert isinstance(plugin, _codegen.BindingsLibrary)


def test_compile_and_run_cffi_plugin_from_C():
    library_name = "test_plugin"
    c_header = "int test_function();"
    c_source_code = f"""
    #include <stdio.h>
    #include "{library_name}.h"

    int main() {{
        printf("%d\\n", test_function());
        return 0;
    }}
    """

    python_wrapper = """
    from test_plugin import ffi

    @ffi.def_extern()
    def test_function():
        return 42
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        build_path = pathlib.Path(tmpdirname)

        try:
            # Generate and compile the CFFI plugin, which creates lib{library_name}.so
            shared_library = f"{library_name}"

            _generator.generate_and_compile_cffi_plugin(
                library_name, c_header, python_wrapper, build_path
            )
            (build_path / f"{library_name}.h").write_text(
                _codegen.add_include_guard(c_header, library_name)
            )
            compiled_library_path = build_path / f"lib{shared_library}.so"

            # Verify the shared library was created
            assert compiled_library_path.exists(), (
                f"Compiled library {compiled_library_path} does not exist."
            )
            assert compiled_library_path.stat().st_size > 0, "Compiled library is empty."

            # Write the main C program to a file
            main_program_path = build_path / "main.c"
            with pathlib.Path.open(main_program_path, "w") as main_program_file:
                main_program_file.write(c_source_code)

            # Compile the main C program against the shared library
            subprocess.run(
                [
                    "gcc",
                    "-o",
                    str(build_path / "test_program"),
                    str(main_program_path),
                    "-L" + str(build_path),
                    "-l" + shared_library,
                    "-Wl,-rpath=" + str(build_path),
                ],
                check=True,
                capture_output=True,
            )

            # Execute the compiled program and capture its output
            result = subprocess.run(
                [build_path / "test_program"], stdout=subprocess.PIPE, check=True
            )
            output = result.stdout.decode().strip()

            # Assert the output of test_function called within the C program
            assert output == "42", f"Expected '42', got '{output}'"
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Execution of compiled Fortran code failed: {e}\nOutput:\n{e.stdout}")
        except Exception as e:
            pytest.fail(
                f"Unexpected error during plugin generation, compilation, or execution: {e}"
            )


@pytest.fixture
def square_plugin():
    return _generator.get_cffi_description([simple.square_from_function], "square_plugin")


def test_render_returns_all_four_sources(square_plugin):
    sources = _generator.render(square_plugin)
    assert sources.py
    assert sources.f90
    assert sources.h
    assert sources.c


def test_render_python_wrapper_contains_user_function(square_plugin):
    sources = _generator.render(square_plugin)
    # The Python wrapper drives CFFI's @ffi.def_extern bindings.
    assert "square_from_function" in sources.py
    assert "@ffi.def_extern" in sources.py


def test_render_python_wrapper_imports_from_definition_module(square_plugin):
    sources = _generator.render(square_plugin)
    # The wrapper's embedded init code must import the function from its
    # actual definition module, not from the caller's namespace.
    assert "from tests.tools.py2fgen.wrappers.simple import square_from_function" in sources.py


def test_render_c_source_embeds_python_wrapper_and_header(square_plugin):
    sources = _generator.render(square_plugin)
    # CFFI bakes the Python wrapper into the .c via embedding_init_code.
    assert "@ffi.def_extern" in sources.c
    # The C header is embedded inline as the set_source preamble, not #include'd —
    # the extern declaration must appear verbatim in sources.c.
    assert "extern int square_from_function_wrapper(" in sources.c


def test_render_fortran_module_is_named_after_library(square_plugin):
    sources = _generator.render(square_plugin)
    # The generated F90 must declare a module whose name matches library_name —
    # check the actual ``module <library_name>`` line, not just any occurrence
    # of the string elsewhere (e.g. inside a bind(c, name=...) attribute).
    assert re.search(r"(?im)^\s*module\s+square_plugin\b", sources.f90)


def test_render_c_header_declares_user_function(square_plugin):
    sources = _generator.render(square_plugin)
    assert "square_from_function" in sources.h


def test_render_is_pure(square_plugin):
    """Rendering twice with identical inputs must be deterministic."""
    a = _generator.render(square_plugin)
    b = _generator.render(square_plugin)
    assert a == b


def test_render_mixed_modules():
    """Functions from different modules each get a per-function ``from X import Y`` line."""
    args = {
        "arr": _definitions.ArrayParamDescriptor(
            rank=1,
            dtype=_definitions.FLOAT64,
            memory_space=_definitions.MemorySpace.HOST,
            is_optional=False,
        ),
    }
    fn_a = _codegen.Func(name="fn_a", module_name="pkg.mod_a", args=args)
    fn_b = _codegen.Func(name="fn_b", module_name="pkg.mod_b", args=args)
    plugin = _codegen.BindingsLibrary(library_name="mixed_plugin", functions=[fn_a, fn_b])

    sources = _generator.render(plugin)

    assert "from pkg.mod_a import fn_a" in sources.py
    assert "from pkg.mod_b import fn_b" in sources.py
