# Coding Guidelines

## Software design

Designing a clean architecture for a software product is a difficult task and developers can easily overlook it under pressure or when they are eager to deliver exciting new features. However, choosing the quick solution instead of a clean one will likely lead to a large amount of additional work in the future (_technical debt_).

To keep technical debt at acceptable levels, design and implementation of new features should follow best practices:

1. Make sure your code is always covered by automatic testing to ensure its correctness. Where unit tests are impractical, use integration tests.
2. Adhere to the [SOLID](https://en.wikipedia.org/wiki/SOLID) principles of software design.
3. Do not reinvent the wheel: if someone solved your problem within the project or in a third party library, consider using their solution or extending it before writing a new component for the same purpose.
4. _You ain't gonna need it_ ([YAGNI](https://en.wikipedia.org/wiki/You_aren%27t_gonna_need_it)): do not design solutions for problems that might come up in the future, as chances are that you will never need that code. Focus on current problems and prepare for future requirements by writing clean code.
5. _Do not repeat yourself_ ([DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)): if you are writing the same code snippet in several places, extract it into a function.
6. Use meaningful names: the purpose of an object should be clear from its name. Usually, class names are nouns and function names are verbs.

<!--
TODO: we should add a doc folder (separate for `tools` and `model`? separate for each package?)


Remember that important design decisions should be properly documented for future reference and to share the knowledge with other developers. We decided to use lightweight _Architecture Decision Records_ (ADRs) for this purpose. The full list of ADRs and documentation for writing new ones can be found in [docs/functional/architecture/Index.md](docs/functional/architecture/Index.md).
-->

## Code Style

We follow the [Google Python Style Guide][google-style-guide] with a few minor changes (mentioned below). Since the best way to remember something is to understand the reasons behind it, make sure you go through the style guide at least once, paying special attention to the discussions in the _Pros_, _Cons_, and _Decision_ subsections.

We deviate from the [Google Python Style Guide][google-style-guide] only in the following points:

- We use [`ruff-linter`][ruff-linter] instead of [`pylint`][pylint].
- We use [`ruff-formatter`][ruff-formatter] for source code and imports formatting, which may work differently than indicated by the guidelines in section [_3. Python Style Rules_](https://google.github.io/styleguide/pyguide.html#3-python-style-rules). For example, maximum line length is set to 100 instead of 79 (although docstring lines should still be limited to 79).
- According to subsection [_2.19 Power Features_](https://google.github.io/styleguide/pyguide.html#219-power-features), direct use of _power features_ (e.g. custom metaclasses, import hacks, reflection) should be avoided, but standard library classes that internally use these power features are accepted. Following the same spirit, we allow the use of power features in infrastructure code with similar functionality and scope as the Python standard library.
- According to subsection [_3.19.12 Imports For Typing_](https://google.github.io/styleguide/pyguide.html#31912-imports-for-typing), symbols from `typing` and `collections.abc` modules used in type annotations _"can be imported directly to keep common annotations concise and match standard typing practices"_. Following the same spirit, we allow symbols to be imported directly from third-party or internal modules when they only contain a collection of frequently used typying definitions.

Further guidance and repository- or package specific guidelines can be found in the respective `docs` folders and `README.md` files.
In particular

- [naming conventions for stencils](docs/stencil_naming_convention.md)
- [liskov integration style guide ](tools/docs/ICON_Liskov_integration_style_guide.md)

### Common questions

- `pass` vs `...` (`Ellipsis`)

  `pass` is the _no-op_ statement in Python and `...` is a literal value (called _Ellipsis_) introduced for slicing collections of unknown number of dimensions. Although they are very different in nature, both of them are used in places where a statement is required purely for syntactic reasons, and there is not yet a clear standard practice in the community about when to use one or the other. We decided to align with the common pattern of using `...` in the body of empty function definitions working as placeholders for actual implementations defined somewhere else (e.g. type stubs, abstract methods and methods appearing in `Protocol` classes) and `pass` in any other place where its usage is mixed with actual statements.

  ```python
  # Correct use of `...` as the empty body of an abstract method
  class AbstractFoo:
      @abstractmethod
      def bar(self) -> Bar: ...


  # Correct use of `pass` when mixed with other statements
  try:
      resource.load(id=42)
  except ResourceException:
      pass
  ```

### Error messages

Error messages should be written as sentences, starting with a capital letter and ending with a period (avoid exclamation marks). Try to be informative without being verbose.
The message should be kept to one sentence if reasonably possible.
Code objects such as 'ClassNames' and 'function_names' should be enclosed in single quotes, and so should string values used for message interpolation.

Examples:

```python
raise ValueError(
    f"Invalid argument 'dimension': should be of type 'Dimension', got '{dimension.type}'."
)
```

Interpolated integer values do not need double quotes, if they are indicating an amount. Example:

```python
raise ValueError(f"Invalid number of arguments: expected 3 arguments, got {len(args)}.")
```

The double quotes can also be dropped when presenting a sequence of values. In this case the message should be rephrased so the sequence is separated from the text by a colon ':'.

```python
raise ValueError(f"unexpected keyword arguments: {', '.join(set(kwarg_names} - set(expected_kwarg_names)))}.")
```

### Docstrings

We encourage to add doc strings for functions, classes and modules if they help the reader understand the code and contain information that is not obvious from the code itself. While we do not yet generate API documentation from doc strings we might do so in the future using [Sphinx] and some extensions such as [Sphinx-autodoc] and [Sphinx-napoleon]. These follow the Google Python Style Guide docstring conventions to automatically format the generated documentation. A complete overview can be found here: [Example Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google).

Sphinx supports the [reStructuredText][sphinx-rest] (reST) markup language for defining additional formatting options in the generated documentation, however section [_3.8 Comments and Docstrings_](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) of the Google Python Style Guide does not specify how to use markups in docstrings. As a result, we decided to forbid reST markup in docstrings, except for the following cases:

- Cross-referencing other objects using Sphinx text roles for the [Python domain](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain) (as explained [here](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#python-roles)).
- Very basic formatting markup to improve _readability_ of the generated documentation without obscuring the source docstring (e.g. ``` ``literal`` ``` strings, bulleted lists).

We highly encourage the [doctest] format for code examples in docstrings. In fact, doctest runs code examples and makes sure they are in sync with the codebase.

### Module structure

In general, you should structure new Python modules in the following way:

1. _shebang_ line: `#! /usr/bin/env python3` (only for **executable scripts**!).
2. License header (see `.license_header.txt`, it is added automatically by `pre-commit` hook)
3. Module docstring.
4. Imports, alphabetically ordered within each block (fixed automatically by `ruff-formatter`):
   1. Block of imports from the standard library.
   2. Block of imports from general third party libraries using standard shortcuts when customary (e.g. `numpy as np`).
   3. Block of imports from specific modules of the project.
5. Definition of exported symbols (optional, mainly for re-exporting symbols from other modules):

```python
__all__ = ["func_a", "CONST_B"]
```

6. Public constants and typing definitions.
7. Module contents organized in a convenient way for understanding how the pieces of code fit together, usually defining functions before classes.

Try to keep sections and items logically ordered, add section separator comments to make section boundaries explicit when needed. If there is not a single evident logical order, pick the order you consider best or use alphabetical order.

Consider configuration files as another type of source code and apply the same criteria, using comments when possible for better readability.

### Ignoring QA errors

You may occasionally need to disable checks from _quality assurance_ (QA) tools (e.g. linters, type checkers, etc.) on specific lines as some tool might not be able to fully understand why a certain piece of code is needed. This is usually done with special comments, e.g. `# noqa: F401`, `# type: ignore`. However, you should **only** ignore QA errors when you fully understand their source and rewriting your code to pass QA checks would make it less readable. Additionally, you should add a short descriptive code if possible (check [ruff rules][ruff-rules] and [mypy error codes][mypy-error-codes] for reference; use `ruff rule <CODE>` to look up the name for a specific code):

```python
f = lambda: "empty"  # noqa: E731 [lambda-assignment]
```

and, if needed, a brief comment for future reference:

```python
...
return undeclared_symbol  # noqa: F821 [undefined-name] on purpose to trigger black-magic
```

## Testing

Testing components is a critical part of a software development project. We follow standard software engineering practices and write unit, integration, and regression tests. [Doctests][doctest] are great for documentation purposes, but they lack features and are difficult to debug, therefore they should not be used as replacement for proper unit tests except in trivial cases.

### Test suite folder structure

Each software component project in the repository should place tests inside a folder named `tests`, which should be a proper Python package structured with the following content:

- a `__init__.py` file at root level adding subfolders as members of a virtual `tests.` package
- a `<component>` folder with a **unique name** in this repository for the package being tested (e.g. `atmosphere_advection` for the `icon4py.model.atmosphere.advection` component).

The `<component>` folder should be a Python package and contain subfolders for every kind of test (e.g. `unit_tests`, `integration_tests`, ...). If needed, it may contain at any level of the file tree a `conftests.py` module for changing the pytest configuration, a `fixtures.py` module with shared fixture definitions and a `utils.py`python modules testing utilities used in the tests.

Example:

```text
/model/system/subsystem/component/
   src/
      ...
   tests/
      __init__.py          # REQUIRED to make this folder a Python package
            # Content of 'tests/__init__.py'
            # This code adds subfolders as subpackages of the `tests.` namespace package 
            __path__ = __import__("pkgutil").extend_path(__path__, __name__)

      system_subsystem_component/
         __init__.py       # REQUIRED to make this folder a Python package
         conftest.py       # pytest settings only
         fixtures.py       # fixture definitions (it might be also a package)
         utils.py          # other shared testing utilities (it might be also a package)
         stencil_tests/
            conftest.py    # specific pytest settings for this folder 
            test_foo.py
            ...
         unit_test/
            test_bar.py
            ...
```

The `scripts-cli` tool contains commands to check some of these points.

For further explanations about the trade-offs of using Python packages for tests organization, the [`pytest` import mechanisms](https://docs.pytest.org/en/stable/explanation/pythonpath.html#import-modes) documentation is a helpful reference. For further explanations about the parametrization of fixtures and tests, the [basics](https://docs.pytest.org/en/stable/how-to/parametrize.html#parametrize-basics) and [advanced](https://docs.pytest.org/en/stable/example/parametrize.html#parametrization-with-multiple-fixtures) `parametrize` examples in the pytest documentation are also very helpful.

### Stencil tests

Test suites inside a `stencil_tests` folder are generally run in integration mode with [icon-exclaim](https://github.com/C2SM/icon-exclaim) and should only contain tests for the GT4Py stencils that might be integrated into ICON.

<!--

## Further 

TODO: add test conventions:
TODO:    - to name test functions
TODO:    - to use pytest features (fixtures, markers, etc.)
TODO:    - to generate mock objects and data for tests (e.g. pytest-factoryboy, pytest-cases)
TODO:    - to use pytest plugins

-->

<!-- Reference links -->

[doctest]: https://docs.python.org/3/library/doctest.html
[google-style-guide]: https://google.github.io/styleguide/pyguide.html
[mypy-error-codes]: https://mypy.readthedocs.io/en/stable/error_code_list.html
[pylint]: https://pylint.pycqa.org/
[ruff-formatter]: https://docs.astral.sh/ruff/formatter/
[ruff-linter]: https://docs.astral.sh/ruff/linter/
[ruff-rules]: https://docs.astral.sh/ruff/rules/
[sphinx]: https://www.sphinx-doc.org
[sphinx-autodoc]: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
[sphinx-napoleon]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html#
[sphinx-rest]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
