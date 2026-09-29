# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import TexSoup


def tex_macros_to_mathjax(filename: str) -> dict[str, str]:
    """
    Parses a LaTeX file to extract macros and converts them to MathJax format.

    Args:
        filename: The path to the LaTeX file containing the macros.

    Returns:
        A dictionary where the keys are the macro names and the values are the
        corresponding MathJax representations. If the macro has arguments, the
        value is a list where the first element is the MathJax representation
        and the second element is the number of arguments.
    """
    latex_macros = {}
    with open(filename) as f:
        soup = TexSoup.TexSoup(f.read())
        for command in ["newcommand", "renewcommand"]:
            for macro in soup.find_all(command):
                name = macro.args[0].string.strip("\\")
                latex_macros[name] = f"{{{macro.args[-1].string}}}"
                if len(macro.args) == 3:
                    latex_macros[name] = [latex_macros[name], int(macro.args[1].string)]
    return latex_macros
