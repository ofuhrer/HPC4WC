import os

import dsl.ir.ir as ir
from dsl.ir.visitor import IRNodeVisitor
#hiiiii

class CodeGen(IRNodeVisitor):
    def __init__(self):
        self.code = "import numpy as np\n\n"
        self.code += "def generated_function(field_1, field_2):"

    def apply(self, ir: ir.IR) -> str:
        self.visit(ir)

    def visit_FieldAccessExpr(self, node: ir.FieldAccessExpr) -> str:
        return node.name

    def visit_LiteralExpr(self, node: ir.LiteralExpr) -> str:
        return node.value

    def visit_AssignmentStmt(self, node: ir.AssignmentStmt) -> str:
        return self.visit(node.left) + "[i,j,k]" + " = " + self.visit(node.right)

    def visit_Vertical(self, node: ir.Vertical) -> str:
        code = f"""
    for k in range({self.visit(node.extent[0][0])}, {self.visit(node.extent[0][1])}):"""
        for stmt in node.body:
            code += self.visit(stmt)
        return code

    def visit_Horizontal(self, node: ir.Horizontal) -> str:
        code = f"""
        for i in range({self.visit(node.extent[0][0])}, {self.visit(node.extent[0][1])}):
            for j in range({self.visit(node.extent[1][0])}, {self.visit(node.extent[1][1])}):
"""
        for stmt in node.body:
            code += f"                {self.visit(stmt)}\n"

        return code

    def visit_IR(self, node: ir.IR, filepath: os.PathLike = os.path.join("dsl", "generated", "main.py")) -> str:
        for stmt in node.body:
            self.code += self.visit(stmt) + "\n"

        with open(filepath, "w") as f:
            f.write(self.code)

        return

    def visit_FieldDeclaration(self, node: ir.FieldDeclaration) -> str:
        return "    " + self.visit(node.name) + " = np.zeros([" + ", ".join([self.visit(_) for _ in node.size]) + "])"
