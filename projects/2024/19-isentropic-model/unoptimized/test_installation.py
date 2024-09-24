# -*- coding: utf-8 -*-
modules = [
    "boundary",
    "diagnostics",
    "diffusion",
    "makesetup",
    "meteo_utilities",
    "microphysics",
    "namelist",
    "output",
    "prognostics",
    "readsim",
    "solver",
    "xzplot"
]

for module in modules:
    try:
        exec("from nmwc_model import {}".format(module))
    except (ImportError, ModuleNotFoundError):
        print("Installation failed: the module {}.py cannot be imported.".format(module))
        exit(0)

print("Installation completed successfully!")
