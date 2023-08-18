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
        return "        " + self.visit(node.left) + " = " + self.visit(node.right)

    def visit_Iterations(self, node: ir.Iterations) -> str:
        code = f"""
    for n in range({self.visit(node.extent[0].start)},{self.visit(node.extent[0].stop)}):"""
        for stmt in node.body:
            code += self.visit(stmt)
        return code

    def visit_Vertical(self, node: ir.Vertical) -> str:
        code = f"""
        for k in range({self.visit(node.extent[0].start)},{self.visit(node.extent[0].stop)}):"""
        for stmt in node.body:
            code += self.visit(stmt)
        return code

    def visit_Horizontal(self, node: ir.Horizontal) -> str:
        code = f"""
            for i in range({self.visit(node.extent[0].start)},{self.visit(node.extent[0].stop)}):
                for j in range({self.visit(node.extent[1].start)},{self.visit(node.extent[1].stop)}):
"""
        for stmt in node.body:
            code += f"                {self.visit(stmt)}\n"

        return code

    def visit_Function(self, node: ir.Function) -> str:
        if node.name == "lap":
            code = f"""(
                                -4.0 * {self.visit(node.args[0])}[i,j,k]
                                + {self.visit(node.args[0])}[i-1,j,k] + {self.visit(node.args[0])}[i+1,j,k]
                                + {self.visit(node.args[0])}[i,j-1,k] + {self.visit(node.args[0])}[i,j+1,k]
                                )
                    """
        elif node.name == "update_halo_bottom_edge":
            code = f"""{self.visit(node.args[0])}[i,j+(ny-2*num_halo),k]
                """

        else:
            print("this is an unknown function!")
            code = f"""#unknown function"""

        return code

    def visit_IR(self, node: ir.IR, filepath: os.PathLike = os.path.join("dsl", "generated", "main.py")) -> str:
        for stmt in node.body:
            self.code += self.visit(stmt) + "\n"

        with open(filepath, "w") as f:
            f.write(self.code)

        return

    def visit_FieldDeclaration(self, node: ir.FieldDeclaration) -> str:
        # das isch nüm wük e field declaration (eh trash ksi vorher) sondern epis wo mer chan bruche falls mers "+[i,j,k]"
        # bim assignment statment wegnehmed. En asatz zum da flexibler si und zb sege field[1,0,2]=5. Field[1:2,-3,j+1] gaht aber ned.
        # Mir bruched das im Moment ned wil mer update halo mit With glöst hend.
        code = f"""{self.visit(node.name)}[{self.visit(node.size[0])},{self.visit(node.size[1])},{self.visit(node.size[2])}]"""
        return code

    def visit_BinaryOp(self, node: ir.BinaryOp) -> str:
        return self.visit(node.left) + ' ' + node.operator + ' ' + self.visit(node.right)

    def visit_UnaryOp(self, node: ir.UnaryOp) -> str:
        return node.operator + ' ' + self.visit(node.operand)

    def visit_SliceExpr(self, node: ir.SliceExpr) -> str:
        start = self.visit(node.start) if node.start else ''
        stop = self.visit(node.stop) if node.stop else ''
        code = f"""{start}:{stop}"""
        return code
