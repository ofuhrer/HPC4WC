import ast
from typing import List

import dsl.ir.ir as ir
from dsl.ir.ir import IR, Horizontal


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

    def visit_With(self, node: ast.With) -> ir.Horizontal:
        expr = node.items[0].context_expr
        if isinstance(expr, ast.Subscript):
            if expr.value.id == "Horizontal":
                self._parent.append(self._scope)

                assert len(node.items) == 1
                extent = []
                for slice in expr.slice.elts:
                    extent.append(self.visit(slice))

                # Create the Horizontal node
                self._scope.body.append(Horizontal(extent))

                # Set the scope to the body of the new Horizontal node
                self._scope = self._scope.body[-1]

                for stmt in node.body:
                    self.visit(stmt)
                self._scope = self._parent.pop()

    def visit_Slice(self, node: ast.Slice) -> List[int]:
        return [self.visit(node.lower), self.visit(node.upper)]
