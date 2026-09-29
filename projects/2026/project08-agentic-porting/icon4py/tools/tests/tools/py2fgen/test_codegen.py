# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import string

import pytest

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen._codegen import (
    BindingsLibrary,
    CHeaderGenerator,
    Func,
    add_include_guard,
    as_f90_value,
    generate_c_header,
    generate_f90_interface,
    generate_python_wrapper,
)


field_2d = py2fgen.ArrayParamDescriptor(
    rank=2,
    dtype=py2fgen.FLOAT32,
    memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
    is_optional=False,
)

field_1d = py2fgen.ArrayParamDescriptor(
    rank=1,
    dtype=py2fgen.FLOAT32,
    memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
    is_optional=False,
)


simple_type = py2fgen.ScalarParamDescriptor(dtype=py2fgen.FLOAT32)


@pytest.mark.parametrize(
    ("param", "expected"), ((simple_type, "value"), (field_2d, None), (field_1d, None))
)
def test_as_target(param, expected):
    assert expected == as_f90_value(param)


foo = Func(
    name="foo",
    module_name="libtest",
    args={
        "one": py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32),
        "two": py2fgen.ArrayParamDescriptor(
            rank=2,
            dtype=py2fgen.FLOAT64,
            memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
            is_optional=False,
        ),
    },
)

bar = Func(
    name="bar",
    module_name="libtest",
    args={
        "one": py2fgen.ArrayParamDescriptor(
            rank=2,
            dtype=py2fgen.FLOAT32,
            memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
            is_optional=False,
        ),
        "two": py2fgen.ScalarParamDescriptor(dtype=py2fgen.INT32),
    },
)


def test_cheader_generation_for_single_function():
    plugin = BindingsLibrary(library_name="libtest_plugin", functions=[foo])

    header = CHeaderGenerator.apply(plugin)
    assert (
        header
        == "extern int foo_wrapper(int one, double* two, int two_size_0, int two_size_1, unsigned char on_gpu);"
    )


def test_cheader_for_pointer_args():
    plugin = BindingsLibrary(library_name="libtest_plugin", functions=[bar])

    header = CHeaderGenerator.apply(plugin)
    assert (
        header
        == "extern int bar_wrapper(float* one, int one_size_0, int one_size_1, int two, unsigned char on_gpu);"
    )


def test_add_include_guard():
    guarded = add_include_guard("extern int foo(int a);", "libtest_plugin")
    assert guarded.startswith("#ifndef LIBTEST_PLUGIN_H\n#define LIBTEST_PLUGIN_H\n")
    assert guarded.rstrip().endswith("#endif")
    assert "extern int foo(int a);" in guarded


def compare_ignore_whitespace(actual: str, expected: str):
    no_whitespace = {ord(c): None for c in string.whitespace}
    if actual.translate(no_whitespace) != expected.translate(no_whitespace):
        print("Expected:")
        print(expected)
        print("Actual:")
        print("------------------------------")
        print(actual)
        print("------------------------------")
        return False
    return True


@pytest.fixture
def dummy_plugin():
    return BindingsLibrary(
        library_name="libtest_plugin",
        functions=[foo, bar],
    )


def test_fortran_interface(dummy_plugin):
    interface = generate_f90_interface(dummy_plugin)
    expected = """
module libtest_plugin
   use, intrinsic :: iso_c_binding
   implicit none

   public :: foo

   public :: bar

   interface

      function foo_wrapper(one, &
                           two, &
                           two_size_0, &
                           two_size_1, &
                           on_gpu) bind(c, name="foo_wrapper") result(rc)
         import :: c_int, c_long, c_float, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         integer(c_int), value, target :: one

         type(c_ptr), value, target :: two

         integer(c_int), value :: two_size_0

         integer(c_int), value :: two_size_1

         logical(c_bool), value :: on_gpu

      end function foo_wrapper

      function bar_wrapper(one, &
                           one_size_0, &
                           one_size_1, &
                           two, &
                           on_gpu) bind(c, name="bar_wrapper") result(rc)
         import :: c_int, c_long, c_float, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: one

         integer(c_int), value :: one_size_0

         integer(c_int), value :: one_size_1

         integer(c_int), value, target :: two

         logical(c_bool), value :: on_gpu

      end function bar_wrapper

   end interface

contains

   subroutine foo(one, &
                  two, &
                  rc)
      use, intrinsic :: iso_c_binding

      integer(c_int), value, target :: one

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: two

      logical(c_bool) :: on_gpu

      integer(c_int) :: two_size_0

      integer(c_int) :: two_size_1

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      !$acc host_data use_device(two)

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

      two_size_0 = SIZE(two, 1)
      two_size_1 = SIZE(two, 2)

      rc = foo_wrapper(one=one, &
                       two=c_loc(two), &
                       two_size_0=two_size_0, &
                       two_size_1=two_size_1, &
                       on_gpu=on_gpu)
      !$acc end host_data
   end subroutine foo

   subroutine bar(one, &
                  two, &
                  rc)
      use, intrinsic :: iso_c_binding

      real(c_float), dimension(:, :), contiguous, intent(inout), target :: one

      integer(c_int), value, target :: two

      logical(c_bool) :: on_gpu

      integer(c_int) :: one_size_0

      integer(c_int) :: one_size_1

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      !$acc host_data use_device(one)

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

      one_size_0 = SIZE(one, 1)
      one_size_1 = SIZE(one, 2)

      rc = bar_wrapper(one=c_loc(one), &
                       one_size_0=one_size_0, &
                       one_size_1=one_size_1, &
                       two=two, &
                       on_gpu=on_gpu)
      !$acc end host_data
   end subroutine bar

end module
"""
    assert compare_ignore_whitespace(interface, expected)


def test_python_wrapper(dummy_plugin):
    interface = generate_python_wrapper(dummy_plugin)
    expected = """import pkgutil
from icon4py.tools.py2fgen import runtime_config

for callable_name in runtime_config.EXTRA_CALLABLES:
    pkgutil.resolve_name(callable_name)()

import logging
from libtest_plugin import ffi
from icon4py.tools.py2fgen import _runtime, _conversion

logger = logging.getLogger(__name__)
log_format = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
logging.basicConfig(
    level=getattr(logging, runtime_config.LOG_LEVEL),
    format=log_format,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# embedded function imports
from libtest import foo
from libtest import bar


@ffi.def_extern(error=2)
def foo_wrapper(one, two, two_size_0, two_size_1, on_gpu):
    with runtime_config.HOOK_BINDINGS_FUNCTION["foo"]:
        try:
            if __debug__:
                logger.info("Python execution of foo started.")

            if __debug__:
                if runtime_config.PROFILING:
                    unpack_start_time = _runtime.perf_counter()

            # ArrayInfos

            two = (
                two,
                (
                    two_size_0,
                    two_size_1,
                ),
                on_gpu,
                False,
            )

            if __debug__:
                if runtime_config.PROFILING:
                    allocate_end_time = _runtime.perf_counter()
                    logger.info(
                        "foo constructing `ArrayInfos` time: %s"
                        % str(allocate_end_time - unpack_start_time)
                    )

                    func_start_time = _runtime.perf_counter()

            if __debug__ and runtime_config.PROFILING:
                perf_counters = {}
            else:
                perf_counters = None
            foo(
                ffi=ffi,
                perf_counters=perf_counters,
                one=one,
                two=two,
            )

            if __debug__:
                if runtime_config.PROFILING:
                    func_end_time = _runtime.perf_counter()
                    logger.info(
                        "foo convert time: %s"
                        % str(
                            perf_counters["convert_end_time"] - perf_counters["convert_start_time"]
                        )
                    )
                    logger.info("foo execution time: %s" % str(func_end_time - func_start_time))

            if __debug__:
                if logger.isEnabledFor(logging.DEBUG):

                    two_arr = _conversion.as_array(ffi, two) if two is not None else None
                    msg = "shape of two after computation = %s" % str(
                        two_arr.shape if two is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "two after computation: %s" % str(two_arr) if two is not None else "None"
                    logger.debug(msg)

            if __debug__:
                logger.info("Python execution of foo completed.")

        except Exception as e:
            logger.exception(f"A Python error occurred: {e}")
            return 2

    return 1


@ffi.def_extern(error=2)
def bar_wrapper(one, one_size_0, one_size_1, two, on_gpu):
    with runtime_config.HOOK_BINDINGS_FUNCTION["bar"]:
        try:
            if __debug__:
                logger.info("Python execution of bar started.")

            if __debug__:
                if runtime_config.PROFILING:
                    unpack_start_time = _runtime.perf_counter()

            # ArrayInfos

            one = (
                one,
                (
                    one_size_0,
                    one_size_1,
                ),
                on_gpu,
                False,
            )

            if __debug__:
                if runtime_config.PROFILING:
                    allocate_end_time = _runtime.perf_counter()
                    logger.info(
                        "bar constructing `ArrayInfos` time: %s"
                        % str(allocate_end_time - unpack_start_time)
                    )

                    func_start_time = _runtime.perf_counter()

            if __debug__ and runtime_config.PROFILING:
                perf_counters = {}
            else:
                perf_counters = None
            bar(
                ffi=ffi,
                perf_counters=perf_counters,
                one=one,
                two=two,
            )

            if __debug__:
                if runtime_config.PROFILING:
                    func_end_time = _runtime.perf_counter()
                    logger.info(
                        "bar convert time: %s"
                        % str(
                            perf_counters["convert_end_time"] - perf_counters["convert_start_time"]
                        )
                    )
                    logger.info("bar execution time: %s" % str(func_end_time - func_start_time))

            if __debug__:
                if logger.isEnabledFor(logging.DEBUG):

                    one_arr = _conversion.as_array(ffi, one) if one is not None else None
                    msg = "shape of one after computation = %s" % str(
                        one_arr.shape if one is not None else "None"
                    )
                    logger.debug(msg)
                    msg = "one after computation: %s" % str(one_arr) if one is not None else "None"
                    logger.debug(msg)

            if __debug__:
                logger.info("Python execution of bar completed.")

        except Exception as e:
            logger.exception(f"A Python error occurred: {e}")
            return 2

    return 1

"""
    assert compare_ignore_whitespace(interface, expected)


def test_c_header(dummy_plugin):
    interface = generate_c_header(dummy_plugin)
    expected = """
    extern int foo_wrapper(int one, double *two, int two_size_0, int two_size_1, unsigned char on_gpu);
    extern int bar_wrapper(float *one, int one_size_0, int one_size_1, int two, unsigned char on_gpu);
    """
    assert compare_ignore_whitespace(interface, expected)


def test_bool_param_codegen():
    bool_func = Func(
        name="bool_fn",
        module_name="libtest",
        args={
            "flag": py2fgen.ScalarParamDescriptor(dtype=py2fgen.BOOL),
            "mask": py2fgen.ArrayParamDescriptor(
                rank=1,
                dtype=py2fgen.BOOL,
                memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
                is_optional=False,
            ),
        },
    )
    plugin = BindingsLibrary(library_name="libtest_plugin", functions=[bool_func])

    header = CHeaderGenerator.apply(plugin)
    assert "unsigned char flag" in header
    assert "unsigned char* mask" in header

    interface = generate_f90_interface(plugin)
    assert "logical(c_bool), value, target :: flag" in interface
    assert "logical(c_bool), dimension(:), contiguous, intent(inout), target :: mask" in interface
