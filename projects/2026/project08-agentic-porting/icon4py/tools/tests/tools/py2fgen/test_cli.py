# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import pathlib
import subprocess
import types

import pytest
from click.testing import CliRunner

import icon4py.tools.py2fgen._utils as utils
from icon4py.tools.py2fgen._cli import main

from tests.tools.py2fgen.wrappers import simple


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def square_wrapper_module() -> types.ModuleType:
    return simple


def compile_fortran_code(
    library_name, samples_path, fortran_driver, compiler, extra_compiler_flags
):
    shared_library = f"{library_name}"
    command = [
        f"{compiler}",
        "-cpp",
        "-I.",
        "-Wl,-rpath=.",
        "-L.",
        f"{library_name}.f90",
        str(samples_path / f"{fortran_driver}.f90"),
        f"-l{shared_library}",
        "-o",
        library_name,
        *list(extra_compiler_flags),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def run_fortran_executable(library_name, env):
    try:
        result = subprocess.run(
            [f"./{library_name}"], capture_output=True, text=True, check=True, env=env
        )
    except subprocess.CalledProcessError as e:
        # If an error occurs, use the exception's `stdout` and `stderr`.
        result = e
    return result


def run_test_case(
    *,
    cli,
    module,
    function,
    library_name,
    samples_path,
    fortran_driver,
    test_temp_dir,
    compiler="gfortran",  # TODO(havogt): don't use hard-coded compiler, see gt4py.cartesian setuptools approach
    extra_compiler_flags=(),
    expected_error_code=0,
    env_vars=None,
):
    with cli.isolated_filesystem(temp_dir=test_temp_dir):
        invoke_cli(cli, module, function, library_name)
        compile_and_run_fortran(
            library_name=library_name,
            samples_path=samples_path,
            fortran_driver=fortran_driver,
            compiler=compiler,
            extra_compiler_flags=extra_compiler_flags,
            expected_error_code=expected_error_code,
            env_vars=env_vars,
        )


def invoke_cli(cli, module: types.ModuleType, function, library_name, extra_args=None):
    rpath = utils.get_prefix_lib_path()

    cli_args = [module.__name__, function, library_name, "-r", rpath]
    if extra_args:
        cli_args.extend(extra_args)
    result = cli.invoke(main, cli_args)
    assert result.exit_code == 0, result.output
    return result


def compile_and_run_fortran(
    *,
    library_name,
    samples_path,
    fortran_driver,
    compiler,
    extra_compiler_flags,
    expected_error_code,
    env_vars,
):
    try:
        compile_fortran_code(
            library_name=library_name,
            samples_path=samples_path,
            fortran_driver=fortran_driver,
            compiler=compiler,
            extra_compiler_flags=extra_compiler_flags,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Compilation failed: {e}\n{e.stderr}\n{e.stdout}")

    try:
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        fortran_result = run_fortran_executable(library_name, env)
        if expected_error_code == 0:
            assert "passed" in fortran_result.stdout, fortran_result.stderr
        else:
            assert "failed" in fortran_result.stdout, fortran_result.stderr
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Execution of compiled Fortran code failed: {e}\nOutput:\n{e.stdout}")


@pytest.mark.parametrize(
    "run_backend, extra_flags",
    [
        ("CPU", ("-DUSE_SQUARE_FROM_FUNCTION",)),
    ],
)
def test_py2fgen_compilation_and_execution_square_cpu(  # noqa: PLR0917 [too-many-positional-arguments]
    cli_runner,
    run_backend,
    samples_path,
    square_wrapper_module,
    extra_flags,
    test_temp_dir,
    fortran_subprocess_env,
):
    """Tests embedding Python functions, and GT4Py program directly.
    Also tests embedding multiple functions in one shared library.
    """
    run_test_case(
        cli=cli_runner,
        module=square_wrapper_module,
        function="square_from_function",
        library_name="square_plugin",
        samples_path=samples_path,
        fortran_driver="test_square",
        test_temp_dir=test_temp_dir,
        extra_compiler_flags=extra_flags,
    )


def test_py2fgen_python_error_propagation_to_fortran(
    cli_runner, samples_path, square_wrapper_module, test_temp_dir, fortran_subprocess_env
):
    """Tests that Exceptions triggered in Python propagate an error code (1) up to Fortran."""
    run_test_case(
        cli=cli_runner,
        module=square_wrapper_module,
        function="square_error",
        library_name="square_plugin",
        samples_path=samples_path,
        fortran_driver="test_square",
        test_temp_dir=test_temp_dir,
        extra_compiler_flags=("-DUSE_SQUARE_ERROR",),
        expected_error_code=1,
    )


def test_py2fgen_compilation_and_execution_bool_cpu(
    cli_runner, samples_path, square_wrapper_module, test_temp_dir, fortran_subprocess_env
):
    """Tests boolean scalar passing and boolean-array writeback end-to-end."""
    run_test_case(
        cli=cli_runner,
        module=square_wrapper_module,
        function="fill_mask",
        library_name="bool_plugin",
        samples_path=samples_path,
        fortran_driver="test_bool",
        test_temp_dir=test_temp_dir,
    )


@pytest.mark.skipif(os.getenv("PY2F_GPU_TESTS") is None, reason="GPU tests only run on CI.")
@pytest.mark.parametrize(
    "function_name, library_name, test_name, extra_flags",
    [
        (
            "square_from_function",
            "square_plugin",
            "test_square",
            ("-acc", "-Minfo=acc", "-DUSE_SQUARE_FROM_FUNCTION"),
        ),
    ],
)
def test_py2fgen_compilation_and_execution_gpu(  # noqa: PLR0917 [too-many-positional-arguments]
    cli_runner,
    function_name,
    library_name,
    test_name,
    samples_path,
    square_wrapper_module,
    extra_flags,
    test_temp_dir,
    fortran_subprocess_env,
):
    run_test_case(
        cli=cli_runner,
        module=square_wrapper_module,
        function=function_name,
        library_name=library_name,
        samples_path=samples_path,
        fortran_driver=test_name,
        test_temp_dir=test_temp_dir,
        compiler=os.environ["NVFORTRAN_COMPILER"],
        extra_compiler_flags=extra_flags,
        env_vars={"ICON4PY_BACKEND": "GPU"},
    )


@pytest.mark.parametrize(
    "extra_flags",
    [
        ("-DPROFILE_SQUARE_FROM_FUNCTION",),
    ],
)
def test_py2fgen_compilation_and_profiling(  # noqa: PLR0917 [too-many-positional-arguments]
    cli_runner,
    samples_path,
    square_wrapper_module,
    extra_flags,
    test_temp_dir,
    tmp_path,
    fortran_subprocess_env,
):
    """Test profiling using cProfile of the generated wrapper."""

    # TODO(havogt):
    # The ``PY2FGEN_EXTRA_CALLABLES`` env var below points at
    # ``icon4py.bindings.viztracer_plugin`` — a runtime indirection (the
    # embedded Python plugin only resolves it when the compiled .so runs),
    # not a static import edge from ``tools/tests/`` into ``bindings``.
    # Exercising the real plugin keeps the integration coverage; a stand-in
    # would duplicate ~80 lines for nothing.

    run_test_case(
        cli=cli_runner,
        module=square_wrapper_module,
        function="square_from_function",
        library_name="square_plugin",
        samples_path=samples_path,
        fortran_driver="test_square",
        test_temp_dir=test_temp_dir,
        extra_compiler_flags=extra_flags,
        env_vars={
            "PY2FGEN_EXTRA_CALLABLES": "icon4py.bindings.viztracer_plugin:init",
            "ICON4PY_TRACING_RANGE": "0:50",
            "ICON4PY_TRACING_NAMES": "square_from_function",
            "ICON4PY_TRACING_OUTPUT_DIR": str(tmp_path),
        },
    )
    assert (tmp_path / "viztracer.json").exists()


def test_py2fgen_incremental_skips_compilation_when_unchanged(
    cli_runner, square_wrapper_module, test_temp_dir, caplog
):
    """Test that running py2fgen twice without changes skips compilation on the second run."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        # First run: should generate and compile
        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(cli_runner, square_wrapper_module, "square_from_function", "square_plugin")
            first_log = caplog.text
        assert "Compiling CFFI dynamic library" in first_log

        # Second run: all files should be up to date, compilation should be skipped
        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(cli_runner, square_wrapper_module, "square_from_function", "square_plugin")
            second_log = caplog.text
        assert "Python wrapper is up to date" in second_log
        assert "Fortran interface is up to date" in second_log
        assert "Skipping compilation" in second_log
        assert "Compiling CFFI dynamic library" not in second_log


def test_py2fgen_regenerate_forces_recompilation(
    cli_runner, square_wrapper_module, test_temp_dir, caplog
):
    """Test that --regenerate forces recompilation even if files are up to date."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        # First run: generate and compile
        invoke_cli(cli_runner, square_wrapper_module, "square_from_function", "square_plugin")

        # Second run with --regenerate: should recompile
        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(
                cli_runner,
                square_wrapper_module,
                "square_from_function",
                "square_plugin",
                extra_args=["--regenerate"],
            )
            regen_log = caplog.text
        assert "Force regeneration requested" in regen_log
        assert "Compiling CFFI dynamic library" in regen_log
        assert "Skipping compilation" not in regen_log


def test_py2fgen_skip_compilation_generates_c(
    cli_runner, square_wrapper_module, test_temp_dir, caplog
):
    """Test that --skip-compilation writes .c, .py, .f90 but no shared library."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(
                cli_runner,
                square_wrapper_module,
                "square_from_function",
                "square_plugin",
                extra_args=["--skip-compilation"],
            )
            log = caplog.text

        assert "Generating C source file" in log
        assert "Compiling CFFI dynamic library" not in log

        assert pathlib.Path("square_plugin.py").exists()
        assert pathlib.Path("square_plugin.f90").exists()
        assert pathlib.Path("square_plugin.c").exists()
        assert not pathlib.Path("libsquare_plugin.so").exists()


def test_py2fgen_skip_compilation_skips_when_up_to_date(
    cli_runner, square_wrapper_module, test_temp_dir, caplog
):
    """Test that --skip-compilation skips regeneration when all files are up to date."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        # First run: generate files
        invoke_cli(
            cli_runner,
            square_wrapper_module,
            "square_from_function",
            "square_plugin",
            extra_args=["--skip-compilation"],
        )

        # Second run: should skip
        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(
                cli_runner,
                square_wrapper_module,
                "square_from_function",
                "square_plugin",
                extra_args=["--skip-compilation"],
            )
            second_log = caplog.text
        assert "Skipping C code generation" in second_log
        assert "Generating C source file" not in second_log


def test_py2fgen_skip_compilation_regenerates_if_c_file_deleted(
    cli_runner, square_wrapper_module, test_temp_dir, caplog
):
    """Test that --skip-compilation regenerates if .c file is missing."""
    with cli_runner.isolated_filesystem(temp_dir=test_temp_dir):
        # First run
        invoke_cli(
            cli_runner,
            square_wrapper_module,
            "square_from_function",
            "square_plugin",
            extra_args=["--skip-compilation"],
        )
        assert pathlib.Path("square_plugin.c").exists()

        # Delete the .c file
        pathlib.Path("square_plugin.c").unlink()

        # Second run: should regenerate since .c is missing
        with caplog.at_level(logging.INFO, logger="py2fgen"):
            caplog.clear()
            invoke_cli(
                cli_runner,
                square_wrapper_module,
                "square_from_function",
                "square_plugin",
                extra_args=["--skip-compilation"],
            )
            regen_log = caplog.text
        assert "Generating C source file" in regen_log
        assert pathlib.Path("square_plugin.c").exists()
