from __future__ import annotations

import importlib
import sys
from typing import List

import black

import toydsl.ir.ir as ir
from toydsl.ir.visitor import IRNodeVisitor


black_mode = black.FileMode(
    target_versions={black.TargetVersion.PY36, black.TargetVersion.PY37},
    line_length=100,
    string_normalization=True,
)


class TextBlock:
    """A block of code with indentation."""

    def __init__(
        self,
        *,
        indent_level: int = 0,
        indent_size: int = 4,
        indent_char: str = " ",
        end_line: str = "\n",
    ) -> None:
        """
        Args:
        indent_level: Initial indentation level.
        indent_size: Number of characters per indentation level.
        indent_char: Character used in the indentation.
        end_line: Character or string used as new-line separator.
        """
        self.indent_level = indent_level
        self.indent_size = indent_size
        self.indent_char = indent_char
        self.end_line = end_line
        self.lines: List[str] = []

    def append(self, new_line: str) -> None:
        self.lines.append(self.indent_str() + new_line)

    def indent_str(self) -> str:
        return self.indent_char * (self.indent_level * self.indent_size)

    def indent(self, steps: int = 1) -> None:
        self.indent_level += steps


class CodeGen(IRNodeVisitor):
    """
    The code-generation module that traverses the IR and generates code form it.
    This module generates simple python code with tripple-nested loops
    """

    @classmethod
    def apply(cls: CodeGen, ir: ir.IR) -> str:
        """
        Entrypoint for the code generation, applying this to an IR returns a formatted function for that IR
        """
        codegen = cls()
        return codegen.visit(ir)

    @staticmethod
    def offset_to_string(offset: ir.AccessOffset) -> str:
        """
        Converts the offset of a FieldAccess to a string with the proper indexing
        """
        return (
            "[idx_i + "
            + str(offset.offsets[0])
            + ", idx_j + "
            + str(offset.offsets[1])
            + ", idx_k + "
            + str(offset.offsets[2])
            + "]"
        )

    @staticmethod
    def create_vertical_loop(vertical_domain: ir.VerticalDomain) -> TextBlock:
        """
        Opens a textblock and generates the loop to start the vertical block
        """
        text_block = TextBlock()
        start_idx = int(vertical_domain.extents.start.level)
        start_string = "k[{start_idx}]+{offset}".format(
            start_idx=start_idx, offset=vertical_domain.extents.start.offset
        )
        end_idx = int(vertical_domain.extents.end.level)
        end_string = "k[{end_idx}]+{offset}".format(
            end_idx=end_idx, offset=vertical_domain.extents.end.offset
        )
        text_block.append(
            "for idx_k in range({condition}):".format(condition=start_string + "," + end_string)
        )
        text_block.indent()
        return text_block

    @staticmethod
    def create_horizontal_loop(
        loop_variable: str, horizontal_domain: ir.HorizontalDomain
    ) -> TextBlock:
        assert loop_variable in ["i", "j"]
        horizontal_loop = TextBlock()
        index = 0 if loop_variable == "i" else 1
        startidx = int(horizontal_domain.extents[index].start.level)
        start = "{loop_variable}[{startidx}]+{offset}".format(
            startidx=startidx,
            offset=horizontal_domain.extents[index].start.offset,
            loop_variable=loop_variable,
        )
        endidx = int(horizontal_domain.extents[index].end.level)
        end = "{loop_variable}[{endidx}]+{offset}".format(
            endidx=endidx,
            offset=horizontal_domain.extents[index].end.offset,
            loop_variable=loop_variable,
        )
        horizontal_loop.append(
            "for idx_{loop_variable} in range({condition}):".format(
                condition=start + "," + end, loop_variable=loop_variable
            )
        )
        horizontal_loop.indent()
        return horizontal_loop

    # ---- Visitor handlers ----
    def generic_visit(self, node: ir.Node, **kwargs) -> None:
        """
        Each visit needs to do something in code-generation, there can't be a default visit
        """
        raise RuntimeError("Invalid IR node: {}".format(node))

    def visit_LiteralExpr(self, node: ir.LiteralExpr) -> str:
        return node.value

    def visit_FieldAccessExpr(self, node: ir.FieldAccessExpr) -> str:
        return node.name + self.offset_to_string(node.offset)

    def visit_AssignmentStmt(self, node: ir.AssignmentStmt) -> str:
        return self.visit(node.left) + "=" + self.visit(node.right)

    def visit_BinaryOp(self, node: ir.BinaryOp) -> str:
        return self.visit(node.left) + node.operator + self.visit(node.right)

    def visit_VerticalDomain(self, node: ir.VerticalDomain) -> List[str]:
        vertical_loop = self.create_vertical_loop(node)
        for stmt in node.body:
            lines_of_code = self.visit(stmt)
            for line in lines_of_code:
                vertical_loop.append(line)

        return vertical_loop.lines

    def visit_HorizontalDomain(self, node: ir.HorizontalDomain) -> List[str]:
        inner_loop = self.create_horizontal_loop("j", node)
        for stmt in node.body:
            inner_loop.append(self.visit(stmt))

        outer_loop = self.create_horizontal_loop("i", node)
        for line in inner_loop.lines:
            outer_loop.append(line)

        return outer_loop.lines

    def visit_IR(self, node: ir.IR) -> str:
        scope = TextBlock()
        function_def = "def {name}({args},i,j,k):".format(
            name=node.name, args=", ".join(node.api_signature)
        )
        scope.append(function_def)
        scope.indent()

        for stmt in node.body:
            vertical_regions = self.visit(stmt)
            for line in vertical_regions:
                scope.append(line)
        code_block = "\n".join(scope.lines)

        formatted_source = black.format_str(code_block, mode=black_mode)
        return formatted_source


class ModuleGen:
    """
    Generator of the module from a given file
    """

    def __init__(self):
        pass

    @classmethod
    def apply(cls, qualified_name, file_path):
        module_gen = cls()
        return module_gen.make_function_from_file(qualified_name, file_path)

    def make_function_from_file(self, qualified_name, file_path):
        """
        Generation of the module and retrieving the function from the module
        """
        module = self.make_module_from_file(qualified_name, file_path)
        return getattr(module, qualified_name)

    def make_module_from_file(self, qualified_name, file_path, *, public_import=False):
        """Import module from file.

        References:
        https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        https://stackoverflow.com/a/43602645

        """
        try:
            spec = importlib.util.spec_from_file_location(qualified_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if public_import:
                sys.modules[qualified_name] = module
                package_name = getattr(module, "__package__", "")
                if not package_name:
                    package_name = ".".join(qualified_name.split(".")[:-1])
                    setattr(module, "__package__", package_name)
                components = package_name.split(".")
                module_name = qualified_name.split(".")[-1]

                if components[0] in sys.modules:
                    parent = sys.modules[components[0]]
                    for i, current in enumerate(components[1:]):
                        parent = getattr(parent, current, None)
                        if not parent:
                            break
                    else:
                        setattr(parent, module_name, module)

        except Exception as e:
            print(e)
            module = None

        return module
