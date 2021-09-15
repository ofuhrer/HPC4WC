import numpy as np
from pygments.formatters import TerminalFormatter
from pygments.lexers import guess_lexer
from pygments import highlight


def print_generated_code(stencil, show_cpp=True):
    """Prints the generated code that executes when a stencil is called.
    If the stencil is generated from C++, show_cpp=True will show the C++ code
    instead of the Python wrapper.
    """
    
    backend = stencil._gt_backend_
    if backend.startswith("gt") and show_cpp:
        build_dir = os.path.splitext(stencil._file_name)[0] + "_pyext_BUILD"
        source_file = os.path.join(build_dir, "computation.cpp")
        with open(source_file, mode="r") as f:
            code = f.read()
            lexer = guess_lexer(code)
            print(highlight(code, lexer, TerminalFormatter()))
    else:
        with open(stencil._file_name, mode="r") as f:
            code = f.read()
            lexer = guess_lexer(code)
            print(highlight(code, lexer, TerminalFormatter()))


def Qcalc(D, m, l, mxGQ, mwGQ):
    """Evaluate entries in the smoothness indicator for WENO"""

    xq = mxGQ / 2
    Qelem = 0
    for i in range(1, m + 2):
        xvec = np.zeros((m - l + 1))
        for k in range(0, m - l + 1):
            xvec[k] = xq[i - 1] ** k / prod(k)

        Qelem = Qelem + matmul(xvec * D, xvec) * mwGQ[i - 1] / 2

    return Qelem


def matmul(X, Y):
    if len(Y.shape) == 1 and len(X.shape) == 1:
        result = 0
        for i in range(X.shape[0]):
            result += X[i] * Y[i]

    elif len(X.shape) == 1:
        result = np.zeros((Y.shape[1]))
        # iterate through rows of X
        for i in range(X.shape[0]):
            # iterate through columns of Y
            for j in range(Y.shape[1]):
                result[j] += X[i] * Y[i][j]

    elif len(Y.shape) == 1:
        result = np.zeros((X.shape[0]))
        # iterate through rows of X
        for i in range(X.shape[0]):
            # iterate through rows of Y
            for k in range(Y.shape[0]):
                result[i] += X[i][k] * Y[k]

    else:
        result = np.zeros((X.shape[0], Y.shape[1]))
        # iterate through rows of X
        for i in range(X.shape[0]):
            # iterate through columns of Y
            for j in range(Y.shape[1]):
                # iterate through rows of Y
                for k in range(Y.shape[0]):
                    result[i][j] += X[i][k] * Y[k][j]
    return result


def prod(x):
    if x == 0:
        return 1
    elif x == 1:
        return 1
    else:
        return x * prod(x - 1)
