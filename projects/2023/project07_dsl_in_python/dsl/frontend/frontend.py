import ast

import dsl.ir.ir as ir
from dsl.ir.ir import IR


class LanguageParser(ast.NodeVisitor):
    def __init__(self):
        self._IR = IR()
        self._scope = self._IR
        self._parent = [None]

    def dump(self):
        for stmt in self._IR.body:
            print(stmt)

    def visit_Name(self, node: ast.Name) -> ir.FieldAccessExpr:
        symbol = node.id
        return ir.FieldAccessExpr(name=symbol)

    def visit_Constant(self, node: ast.Constant) -> ir.LiteralExpr:
        return ir.LiteralExpr(value=str(node.value))

    def visit_Assign(self, node: ast.Assign) -> None:
        assert len(node.targets) == 1
        lhs = self.visit(node.targets[0])
        rhs = self.visit(node.value)
        assign = ir.AssignmentStmt(left=lhs, right=rhs)
        self._scope.body.append(assign)

    def visit_Subscript(self, node: ast.Subscript) -> ir.FieldDeclaration:
        name = self.visit(node.value)
        size = [self.visit(_) for _ in node.slice.elts]

        field_declaration = ir.FieldDeclaration(name, size)
        self._scope.body.append(field_declaration)
        return field_declaration
