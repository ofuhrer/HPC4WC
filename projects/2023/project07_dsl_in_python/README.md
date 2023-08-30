# Project 07: DSL

## Idea

Create a simple domain specific language (DSL).

## Usage

Install the package (from this directory)

### Language

The language uses actual python code, but then parses the self.code to create custom code, using our intermediate
representation (IR) from the python abstract syntax tree (AST).

```
python -m pip install -e .
```

Run the example

1. Generate the code

```
python example/Example_Stencil.py
```

2. Run the generated code

```
python driver/driver.py
```

This will also display validation output and diagnostics.