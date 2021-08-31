import ast
import inspect
import sys
from typing import List

import toydsl.ir.ir as ir
from toydsl.ir.ir import IR, AxisInterval, HorizontalDomain, LevelMarker, Offset, VerticalDomain


class IndexGen(ast.NodeVisitor):
    """Visitor to generated AxisIntervals from a slice or a list of slices"""

    def __init__(self) -> None:
        self.offset: Offset = Offset()
        self.sign: int = 1

    @classmethod
    def apply(cls, node) -> List[AxisInterval]:
        foo = cls()
        intervals: List[AxisInterval] = []
        if isinstance(node, ast.Slice):
            intervals.append(foo.visit(node))
        else:
            for dim in node.dims:
                intervals.append(foo.visit(dim))
        return intervals

    def visit_Slice(self, node: ast.Slice) -> AxisInterval:
        self.offset = Offset()
        self.visit(node.lower)
        lower = self.offset
        self.offset = Offset()
        self.visit(node.upper)
        upper = self.offset
        return AxisInterval(lower, upper)

    def visit_Name(self, node: ast.Name) -> None:
        assert node.id in ["start", "end"]
        if node.id == "end":
            self.offset.level = LevelMarker.END
        else:
            self.offset.level = LevelMarker.START

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Visits the binary operator between the offset and the levelmarker"""
        assert isinstance(node.left, ast.Name)
        assert isinstance(node.right, ast.Constant)
        if isinstance(node.op, ast.Add):
            self.sign = 1
        else:
            self.sign = -1
        self.visit(node.left)
        self.visit(node.right)

    def visit_Constant(self, node: ast.Constant) -> None:
        self.offset.offset = self.sign * node.value


class ArgumentParser(ast.NodeVisitor):
    @classmethod
    def apply(cls, node) -> str:
        parser = cls()
        return parser.visit(node)

    def visit_arg(self, node: ast.arg) -> str:
        # TODO: check the type_comment?
        return node.arg


class LanguageParser(ast.NodeVisitor):
    def __init__(self):
        self._IR = IR()
        self._scope = self._IR
        self._parent = [None]

    def visit_Constant(self, node: ast.Constant) -> ir.LiteralExpr:
        return ir.LiteralExpr(value=str(node.value))

    def visit_Name(self, node: ast.Name) -> ir.FieldAccessExpr:
        symbol = node.id
        return ir.FieldAccessExpr(name=symbol, offset=ir.AccessOffset(0, 0, 0))

    def visit_Subscript(self, node: ast.Subscript) -> ir.FieldAccessExpr:
        elts = node.slice.elts if sys.version_info >= (3,9,0) else node.slice.value.elts
        node_value = []
        for i in range(3):
            if(isinstance(elts[i],ast.Constant)):
                node_value.append(elts[i].value)
            else:
                if isinstance(elts[i].op,ast.USub):
                    out = -1*elts[i].operand.value
                else:
                    out = elts[i].operand.value
                node_value.append(out)
        offset = ir.AccessOffset(
            node_value[0],
            node_value[1],
            node_value[2],
        )
        return ir.FieldAccessExpr(name=node.value.id, offset=offset)

    def visit_Assign(self, node: ast.Assign) -> None:
        assert len(node.targets) == 1
        lhs = self.visit(node.targets[0])
        rhs = self.visit(node.value)
        assign = ir.AssignmentStmt(left=lhs, right=rhs)
        self._scope.body.append(assign)

    def visit_With(self, node: ast.With) -> None:
        if isinstance(node.items[0].context_expr, ast.Subscript):
            if node.items[0].context_expr.value.id == "Vertical":
                self._parent.append(self._scope)
                index = IndexGen.apply(node.items[0].context_expr.slice)
                self._scope.body.append(VerticalDomain())
                self._scope.body[-1].extents = index[-1]
                self._scope = self._scope.body[-1]
                for stmt in node.body:
                    self.visit(stmt)
                self._scope = self._parent.pop()
            elif node.items[0].context_expr.value.id == "Horizontal":
                index = IndexGen.apply(node.items[0].context_expr.slice)
                self._parent.append(self._scope)
                self._scope.body.append(HorizontalDomain(index))
                self._scope = self._scope.body[-1]
                for stmt in node.body:
                    self.visit(stmt)
                self._scope = self._parent.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._IR.name = node.name
        for arg in node.args.args:
            self._IR.api_signature.append(ArgumentParser.apply(arg))
        for element in node.body:
            self.visit(element)

    def visit_BinOp(self,node: ast.BinOp) -> ir.BinaryOp:
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            op_string = "+"
        elif isinstance(node.op, ast.Sub):
            op_string = "-"
        elif isinstance(node.op, ast.Mult):
            op_string = "*"
        elif isinstance(node.op, ast.Div):
            op_string = "/"
        elif isinstance(node.op, ast.Pow):
            op_string = "**"
        elif isinstance(node.op, ast.Mod):
            op_string = "%"
        else:
            op_string = ""
        return ir.BinaryOp(left=lhs,right=rhs,operator=op_string)

    def visit_UnaryOp(self,node: ast.UnaryOp) -> ir.LiteralExpr:
        if isinstance(node.op,ast.USub):
            out = "-" + str(node.operand.value)
        else:
            out = str(node.operand.value)
        return ir.LiteralExpr(value=out)


def parse(function):
    p = LanguageParser()
    source = inspect.getsource(function)
    funcAST = ast.parse(source)
    p.visit(funcAST)
    return p._IR
