import functools
import hashlib
import inspect
import os
import sys
import time
from pathlib import Path

from toydsl.backend.codegen import CodeGen, ModuleGen
from toydsl.backend.codegen_cpp import CodeGenCpp, format_cpp, compile_cpp, load_cpp_module
from toydsl.frontend.frontend import parse


def driver_cpp(function, hash: str, cache_dir: Path):
    """
    Driver for generating the c++ code, formatting it, compiling it, and loading the
    resulting shared object as a python module.
    """

    # we need to do this outside of the if block because we need the function name
    ir = parse(function)
    function_name = ir.name

    # We actually hash the generated C++ code as well. This is a convenience feature
    # so that changing the C++ code generation causes an update. In a real usecase
    # the generated C++ code would not be hashed, we would only need the hash of
    # the input code, because end-users would not be changing the function mapping
    # input code to C++.
    # cpp_hash = hash_string(code)

    code_dir = cache_dir / "cpp_{}".format(hash)
    so_filename = code_dir / "build" / "dslgen.so"

    if not os.path.isfile(so_filename):
        # For now we just perform all the generation steps if the .so file
        # is missing. The case where some of the steps have already been
        # performed is rare and we wouldn't save much time anyway.

        start_time = time.perf_counter()

        cmake_dir = Path(__file__).parent.parent / "cpp"
        code = CodeGenCpp.apply(ir)
        cpp_filename = code_dir / "dslgen.cpp"

        os.makedirs(code_dir, exist_ok=True)
        with open(cpp_filename, "w") as f:
            f.write(code)

        format_cpp(cpp_filename, cmake_dir)
        compile_cpp(code_dir, cmake_dir)

        end_time = time.perf_counter()
        print("\n\nGenerated, formatted, and compiled C++ code in {:.2f} seconds.".format(end_time - start_time), file=sys.stderr)

    return getattr(load_cpp_module(so_filename), function_name)

def driver_python(function, hash: str, cache_dir: Path):
    """
    Driver for generating a module from a parsable function while storing the python module
    in the given cache directory.
    """
    filename = cache_dir / "generated_{hash}.py".format(hash=hash),

    ir = parse(function)
    code = CodeGen.apply(ir)

    # print(CodeGenCpp.apply(ir))

    with open(filename, "w") as f:
        f.write(code)
    return ModuleGen.apply(ir.name, filename)


def set_up_cache_directory() -> str:
    """Searches the system for the CODE_CACHE_ROOT directory and sets it up if necessary"""
    cache_dir = os.getenv("CODE_CACHE_ROOT")
    if cache_dir is None:
        cache_dir = ".codecache"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def hash_string(input: str) -> str:
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(input.encode())
    return hash_algorithm.hexdigest()[:10]

def hash_source_code(definition_func) -> str:
    """Hashes the source code of a function to get a unique ID for a target file"""
    return hash_string(repr(inspect.getsource(definition_func)))

def computation(func):
    """Main entrypoint into the DSL.
    Decorating functions with this call will allow for calls to the generated code
    """

    @functools.wraps(func)
    def _decorator(definition_func):
        cache_dir = set_up_cache_directory()
        hash = hash_source_code(definition_func)
        stencil_call = driver_cpp(
            definition_func,
            hash,
            Path(cache_dir)
        )
        return stencil_call

    return _decorator(func)
