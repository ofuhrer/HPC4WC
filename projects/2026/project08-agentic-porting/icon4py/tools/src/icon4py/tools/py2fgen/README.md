# Py2fgen

A lightweight Fortran bindings generator for Python functions utilizing CFFI.

## Getting started

- Decorate your Python function with `py2fgen.export()`.
- Provide information about the function's parameters, in one of the following ways (see [Parameter Descriptors](#parameter-descriptors))
  - provide full `ParamDescriptors`
  - provide a `ParamDescriptor` via `Annotated` type hint for each parameter
  - provide a function that translates the type annotation to a `ParamDescriptor`
- Optional: provide a function that specifies how to convert the raw arguments to custom types.
- Finally, run the py2fgen command line tool to generate:
  - the Fortran module
  - a compiled shared library containing the wrapper code around your function

## Documentation

### How it works

For a function to be exportable with py2fgen, it needs to provide the attribute `param_descriptors`.
The easiest way is to use the `py2fgen.export()` decorator passing a `param_descriptors` dictionary,
with parameter names as keys and `ParamDescriptor` instances as values.

The parameter descriptors are used to generate the Fortran interface (and a private C interface).

**Example**

```python
@py2fgen.export(
    param_descriptors={
        "scalar": py2fgen.ScalarParamDescriptor(py2fgen.FLOAT64),
        "array": py2fgen.ArrayParamDescriptor(
            rank=2, dtype=py2fgen.FLOAT64, memory_space=py2fgen.MAYBE_DEVICE, is_optional=False
        ),
    }
)
def foo(scalar: float, array: np.ndarray): ...
```

Alternatively, the user can provide the parameter descriptors using `Annotated`.

**Example**

```python
@py2fgen.export()
def foo(
    scalar: Annotated[float, py2fgen.ScalarParamDescriptor(py2fgen.FLOAT64)],
    array: Annotated[
        np.ndarray,
        py2fgen.ArrayParamDescriptor(
            rank=2, dtype=py2fgen.FLOAT64, memory_space=py2fgen.MAYBE_DEVICE, is_optional=False
        ),
    ],
): ...
```

The most flexible, but most complicated way is to set the `annotation_descriptor_hook`,
which is a function that takes an annotation and returns a `ParamDescriptor`.
If `None` is returned we delegate to `Annotated` translation.

**Example**

```python
def my_hook(annotation: Any) -> py2fgen.ParamDescriptor | None:
    if annotation in (int, np.int64):
        return py2fgen.ScalarParamDescriptor(py2fgen.INT64)
    if annotation in (float, np.float):
        return py2fgen.ScalarParamDescriptor(py2fgen.FLOAT64)
    return None


@py2fgen.export(annotation_descriptor_hook=my_hook)
def foo(scalar: float, array: np.ndarray): ...
```

### Parameter descriptors

We provide 2 kinds of `ParamDescriptor`s:

- `ScalarParamDescriptor`: for scalar parameters with attributes
  - `dtype`: which needs to be `py2fgen.<dtype>`, with dtype `BOOL`, `INT32`, `INT64`, `FLOAT32`, `FLOAT64`.
- `ArrayParamDescriptor`: for array parameters with attributes
  - `rank`: the rank of the array
  - `dtype`: see `ScalarParamDescriptor`
  - `memory_space`: which is `py2fgen.HOST` for CPU arrays or `py2fgen.MAYBE_DEVICE` for arrays that are on GPU if compiled with OpenACC, otherwise on CPU
  - `is_optional`: whether the array is optional or not, optional arrays need to be passed as Fortran `pointer`

### Raw argument conversion hooks

By default, array arguments are translated to Numpy or CuPy arrays.
Alternatively, you can provide a hook that provides a translation function from an `ArrayInfo` to your custom type.
Note, this translation function is executed on each call and should be as efficient as possible, e.g. by caching the result.

**Example**

```python
@py2fgen.export(annotation_mapping_hook=...)
def foo(array: np.ndarray): ...
```

### Optimized Python

For production, make sure you set the environment variable `PYTHONOPTIMIZE` to `2` to enable Python optimizations,
see https://docs.python.org/3/using/cmdline.html#envvar-PYTHONOPTIMIZE.

### Debugging the bindings

For debugging, you can set the environment variable `PY2FGEN_LOGGING` to a log level value (e.g., `DEBUG`, `INFO`) to print debug information.

Additionally, you can enable profiling by setting the environment variable `PY2FGEN_PROFILE` to `1`.

Note that debugging and profiling are not available if Python is set to optimized mode.

### Booleans

Booleans use the 1-byte chain `unsigned char` (C) / `logical(kind=c_bool)`
(Fortran) / NumPy `bool_` (Python). The C side is `unsigned char` rather than
`_Bool` because some Fortran compilers (notably NVHPC) write `.TRUE.` as `-1`
(byte `0xFF`), which CFFI's strict `_Bool` validator rejects. `unsigned char`
accepts any byte value; on the Python side a non-zero byte still reads as
`True` through the `bool_` view.

Callers must declare boolean scalar and array arguments as `logical(kind=c_bool)` (not the default `logical`); converting from a default `logical` is the caller's responsibility.

### Future improvements

- Currently we require the `rank` of an array to be known at bindings generation time. We could make this more flexible, by passing an `ArrayInfo`-like struct from Fortran to Python.
- In the CLI interface the user has to provide a module and all functions. Instead we can just request the modules and export all functions that are exportable.
