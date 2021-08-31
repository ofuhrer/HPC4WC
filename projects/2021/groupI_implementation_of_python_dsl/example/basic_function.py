import numpy as np
import time

from toydsl.driver.driver import computation

# This import is not needed. Horizontal etc. are not used by the python interpretor, we
# just leave the import here so that VS Code is happy. :)
from toydsl.frontend.language import Horizontal, Vertical, end, start


@computation
def otherfunc(out_field, in_field):
    """
    A basic test of a funcction exercising the patterns of the toyDSL language.
    """
    with Vertical[start:end]:
        with Horizontal[start : end - 1, start : end - 1]:
            in_field[1, 0, 0] = 2
        with Horizontal[start : end - 1, start:end]:
            out_field = in_field[1, 0, 0] + 5*in_field[0,1,0]


def set_up_data():
    """
    Set up the input for the test example
    """
    i = [0, 50]
    j = [0, 50]
    k = [0, 50]
    shape = (i[-1], j[-1], k[-1])
    a = np.ones(shape)
    b = np.zeros(shape)
    return a, b, i, j, k


if __name__ == "__main__":
    input, output, i, j, k = set_up_data()

    num_runs = 10

    start = time.time_ns()
    for _ in range(num_runs):
        otherfunc(output, input, i, j, k)
    end = time.time_ns()
    print(output[:, :, 0].T)

    print("Called otherfunc {} times in {} seconds".format(num_runs, (end-start)/(10**9)))
