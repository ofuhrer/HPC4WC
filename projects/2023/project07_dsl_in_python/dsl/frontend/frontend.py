import ast
from typing import List

import dsl.ir.ir as ir
from dsl.ir.ir import IR, Horizontal, Vertical, BinaryOp, SliceExpr, UnaryOp, Iterations


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

    def visit_Call(self, node: ast.Call) -> ir.Function:
        function_name = node.func.id
        args = [self.visit(arg) for arg in node.args]
        return ir.Function(name=str(function_name), args=args)

    def visit_Subscript(self, node: ast.Subscript) -> ir.FieldDeclaration:
        name = self.visit(node.value)
        size = [self.visit(_) for _ in node.slice.elts]
        self.visit(node.value)
        [self.visit(_) for _ in node.slice.elts]

        field_declaration = ir.FieldDeclaration(name, size)
        return field_declaration

    def visit_With(self, node: ast.With) -> ir.Node:
        expr = node.items[0].context_expr
        if isinstance(expr, ast.Subscript):
            if expr.value.id == "Iterations":
                self._parent.append(self._scope)

                assert len(node.items) == 1
                extent = []
                extent.append(self.visit(expr.slice))

                # Create the Iterations node
                self._scope.body.append(Iterations(extent))

                # Set the scope to the body of the new Vertical node
                self._scope = self._scope.body[-1]

                for stmt in node.body:
                    self.visit(stmt)
                self._scope = self._parent.pop()





            if expr.value.id == "Vertical":
                self._parent.append(self._scope)

                assert len(node.items) == 1
                extent = []
                extent.append(self.visit(expr.slice))

                # Create the Vertical node
                self._scope.body.append(Vertical(extent))

                # Set the scope to the body of the new Horizontal node
                self._scope = self._scope.body[-1]

                for stmt in node.body:
                    self.visit(stmt)
                self._scope = self._parent.pop()

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

    def visit_BinOp(self, node: ast.BinOp) -> ir.BinaryOp:
        left_expr = self.visit(node.left)
        right_expr = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return BinaryOp(left_expr, right_expr, '+')
        elif isinstance(node.op, ast.Sub):
            return BinaryOp(left_expr, right_expr, '-')
        elif isinstance(node.op, ast.Mult):
            return BinaryOp(left_expr, right_expr, '*')

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ir.UnaryOp:
        operand = self.visit(node.operand)
        return UnaryOp(operand, '-')

    def visit_Slice(self, node: ast.Slice) -> ir.SliceExpr:
        start = self.visit(node.lower) if node.lower else None
        stop = self.visit(node.upper) if node.upper else None
        return SliceExpr(start, stop)
