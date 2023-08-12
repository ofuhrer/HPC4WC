from typing import List, Optional


class Node:
    pass


class Expr(Node):
    pass


class Stmt(Node):
    pass


class Function(Node):
    def __init__(self, name: str, args: list):
        self.name = name
        self.args = args

class FieldAccessExpr(Expr):
    def __init__(self, name: str):
        self.name = name


class FieldDeclaration(Expr):
    def __init__(self, name: str, size: list):
        self.name = name
        self.size = size


class LiteralExpr(Expr):
    value: str

    def __init__(self, value: str, dtype=float):
        self.value = value
        self.dtype = dtype


class AssignmentStmt(Stmt):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right


class Horizontal(Node):
    def __init__(self, extent: List):
        self.body: List[Stmt] = []
        self.extent = extent


class Vertical(Node):
    def __init__(self, extent: List):
        self.body: List[Stmt] = []
        self.extent = extent

class BinaryOp(Expr):
    """Any binary operator expression"""
    def __init__(self, left: Expr, right: Expr, operator: str):
        self.left = left
        self.right = right
        self.operator = operator


class UnaryOp(Expr):
    """for field[1:-1] oä"""
    def __init__(self, operand: Expr, operator: str):
        self.operand = operand
        self.operator = operator

class SliceExpr(Expr):
    def __init__(self, start: Optional[Expr]=None, stop: Optional[Expr]=None):
        self.start = start
        self.stop = stop

class IR(Node):
    def __init__(self):
        self.name = ""
        self.body = []
