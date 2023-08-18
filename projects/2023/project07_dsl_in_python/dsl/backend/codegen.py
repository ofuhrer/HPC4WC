import os

import dsl.ir.ir as ir
from dsl.ir.visitor import IRNodeVisitor


# hiiiii

class CodeGen(IRNodeVisitor):
    def __init__(self):
        self.indention = 0
        self.code = "import numpy as np\n\n"
        self.code += "def generated_function(in_field, out_field, num_halo, nx, ny, nz, num_iter, tmp_field, alpha):"
        self.indent()

    def indent(self):
        self.code += "\n"
        self.indention += 1

    def dedent(self):
        self.code += "\n"
        self.indention -= 1

    def get_indent(self):
        idt = "    "
        return self.indention * idt

    def set_indent(self, to):
        self.code += "\n"
        self.indention = to

    def apply(self, ir: ir.IR) -> str:
        self.visit(ir)

    def visit_FieldAccessExpr(self, node: ir.FieldAccessExpr) -> str:
        self.code += node.name

    def visit_LiteralExpr(self, node: ir.LiteralExpr) -> str:
        self.code += node.value

    def visit_AssignmentStmt(self, node: ir.AssignmentStmt) -> str:
        self.code += self.get_indent()
        self.visit(node.left)
        self.code += " = "
        self.visit(node.right)

    def visit_Vertical(self, node: ir.Vertical):
        self.generate_loop_bounds(node, "k")
        for stmt in node.body:
            self.visit(stmt)
        self.dedent()

    def visit_Horizontal(self, node: ir.Horizontal) -> str:
        outer_indention = self.indention

        self.generate_loop_bounds(node, "j")
        self.generate_loop_bounds(node, "i")

        for stmt in node.body:
            self.visit(stmt)
            self.code += "\n"

        self.set_indent(outer_indention)

    def visit_Iterations(self, node: ir.Iterations) -> str:
        self.generate_loop_bounds(node, "n")
        for stmt in node.body:
            self.visit(stmt)
            self.code += "\n"
        self.dedent()

    def visit_Function(self, node: ir.Function) -> str:
        if node.name == "lap":
            self.code += f"-4.0 *"
            self.visit(node.args[0])
            self.code += "[i, j, k] + "
            self.visit(node.args[0])
            self.code += "[i-1, j, k] + "
            self.visit(node.args[0])
            self.code += "[i+1, j, k] + "
            self.visit(node.args[0])
            self.code += "[i, j-1, k] + "
            self.visit(node.args[0])
            self.code += "[i, j+1, k]"

        elif node.name == "update_halo_bottom_edge":
            self.code += f"{self.visit(node.args[0])}[i, j+(ny - 2 * num_halo), k]"

        else:
            print("this is an unknown function!")
            self.code += "#unknown function"

    def visit_IR(self, node: ir.IR, filepath: os.PathLike = os.path.join("dsl", "generated", "main.py")) -> str:
        for stmt in node.body:
            self.visit(stmt)
            self.code += "\n"

        with open(filepath, "w") as f:
            f.write(self.code)

    def visit_FieldAccess(self, node: ir.FieldDeclaration) -> str:
        # das isch nüm wük e field declaration (eh trash ksi vorher) sondern epis wo mer chan bruche falls mers "+[i,j,k]"
        # bim assignment statment wegnehmed. En asatz zum da flexibler si und zb sege field[1,0,2]=5. Field[1:2,-3,j+1] gaht aber ned.
        # Mir bruched das im Moment ned wil mer update halo mit With glöst hend.
        self.visit(node.name)
        self.code += "["
        for n in range(0, len(node.size)):
            self.visit(node.size[n])
            if n != len(node.size) - 1:
                self.code += ","
        self.code += "]"

    def visit_BinaryOp(self, node: ir.BinaryOp) -> str:
        self.visit(node.left)
        self.code += node.operator
        self.visit(node.right)

    def visit_UnaryOp(self, node: ir.UnaryOp) -> str:
        self.code += node.operator
        self.visit(node.operand)

    def visit_SliceExpr(self, node: ir.SliceExpr) -> str:
        if node.start:
            self.visit(node.start)
        else:
            self.code += ""
        self.code += ":"
        if node.stop:
            self.visit(node.stop)
        else:
            self.code += ""

    def generate_loop_bounds(self, node, variable):
        self.code += self.get_indent()
        self.code += f"for {variable} in range("
        self.visit(node.extent[0].start)
        self.code += ","
        self.visit(node.extent[0].stop)
        self.code += "):"
        self.indent()
