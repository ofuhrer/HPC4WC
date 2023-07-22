from dsl.frontend.language import Horizontal, Vertical
from dsl.frontend.parser import parse_function


def example_function():
    with Vertical[1:10]:
        with Horizontal[2:8, 2:8]:
            field_1 = 10


if __name__ == "__main__":
    parse_function(example_function)
