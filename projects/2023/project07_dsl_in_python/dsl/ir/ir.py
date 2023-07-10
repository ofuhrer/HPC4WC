class Node:
    pass


class Expr(Node):
    pass


class Stmt(Node):
    pass


class FieldAccessExpr(Expr):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"{self.name}"

class LiteralExpr(Expr):
    value: str

    def __init__(self, value: str, dtype=float):
        self.value = value
        self.dtype = dtype

    def __str__(self):
        return f"{self.value}"

class AssignmentStmt(Stmt):
    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right

    def __str__(self):
        return f"{self.left} = {self.right}"


class IR(Node):
    def __init__(self):
        self.name = ""
        self.body = []
