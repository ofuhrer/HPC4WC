from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

#Change "mult_fast" & ["mult_fast.pyx"] depends on the file name
extensions = [
    Extension(
        "cython_loop2",
        ["cython_loop2.pyx"],
        include_dirs = [np.get_include()]
    ),
]

module_cython = cythonize(extensions)

# If you want html (Python-interpreter interactions with C): annotate=True
setup(
    name = "cython_loop2.pyx",
    include_dirs = [np.get_include()],
    ext_modules = module_cython
)
