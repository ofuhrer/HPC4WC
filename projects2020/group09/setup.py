# COMPILE CYTHON
# source: https://towardsdatascience.com/cython-a-speed-up-tool-for-your-python-function-9bab64364bfd

# keep html
# python setup.py build_ext --inplace

# Keep only the .c and .so
# python setup.py build_ext --inplace clean --all


from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

#Change "mult_fast" & ["mult_fast.pyx"] depends on the file name
extensions = [
    Extension(
        "cython_loop",
        ["cython_loop.pyx"],
    ),
]

# If you want html (Python-interpreter interactions with C): annotate=True
setup(
    name = "cython_loop.pyx",
    packages = find_packages(),
    ext_modules = cythonize(extensions, annotate=True)
)
