# Toy DSL for weather and climate models

Requirements:

* cmake >= 3.12
* make
* python >= 3.8
* boost >= 1.68
* clang-format (optional)

## Building / Running

Disclaimer: the following instructions work on my machine where `wheel` is not installed, a different setup might be required if your pip doesn't fall back to the legacy `setup.py install`.

```bash
cd /path/to/toyDSL

python -m venv venv
. venv/bin/activate
pip install numpy black matplotlib
pip install .

PYTHONPATH=$PYTHONPATH:$PWD python example/basic_function.py
```

## Running on CSCS

Load up-to-date versions of our dependencies:

```bash
module load CMake/3.14.5
module load Boost/1.70.0-CrayGNU-20.11-python3
```

## Testing

To test if the generated code is working properly, one can run the example stencil_cody.py which will generate an image of the input and output data and check by themself if the result is the one expected.
