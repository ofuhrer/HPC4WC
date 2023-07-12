import dsl.ir.ir as ir
from dsl.ir.visitor import IRNodeVisitor


class CodeGen(IRNodeVisitor):
    def __init__(self):
        self.code = "import numpy as np\n\n"

    def apply(self, ir: ir.IR) -> str:
        self.visit(ir)

    def visit_FieldAccessExpr(self, node: ir.FieldAccessExpr) -> str:
        return node.name

    def visit_LiteralExpr(self, node: ir.LiteralExpr) -> str:
        return node.value

    def visit_AssignmentStmt(self, node: ir.AssignmentStmt) -> str:
        return """
for i in range(1):
    for j in range(2):
        for k in range(3):
""" + "            " + self.visit(node.left) + "[i,j,k]" + " = " + self.visit(
            node.right) + "\n" + "print(" + self.visit(node.left) + ")"

    def visit_IR(self, node: ir.IR) -> str:
        for stmt in node.body:
            self.code += self.visit(stmt) + "\n"

        with open("test.py", "w") as f:
            f.write(self.code)

        return

    def visit_FieldDeclaration(self, node: ir.FieldDeclaration) -> str:
        return self.visit(node.name) + " = np.zeros([" + ", ".join([self.visit(_) for _ in node.size]) + "])"
