from __future__ import annotations

import enum
from typing import List


@enum.unique
class LevelMarker(enum.Enum):
    START = 0
    END = -1

    def __str__(self):
        return self.name

    def __int__(self):
        return self.value


class Node:
    pass


class Offset:
    """An offset to a spacial dimension"""

    def __init__(self, level: LevelMarker = LevelMarker.START, offset: int = 0):
        self.level = level
        self.offset = offset


class AxisInterval:
    """An axis interval to be traversed in any of the horizontal dimensions"""

    def __init__(self, start: Offset = Offset(), end: Offset = Offset()):
        self.start = start
        self.end = end


class HorizontalDomain(Node):
    """A horizontal execution containing a list of statements"""

    def __init__(self, extents=[AxisInterval(), AxisInterval()]):
        self.body: List[Stmt] = []
        self.extents: List[AxisInterval] = extents


class VerticalDomain(Node):
    """A vertical execution containing a list of horizontal executions"""

    def __init__(self, extents=AxisInterval()):
        self.body: List[HorizontalDomain] = []
        self.extents: AxisInterval = extents


class AccessOffset:
    """An offset to a field access"""

    offsets: List[int]

    def __init__(self, i, j, k):
        self.offsets = []
        self.offsets.append(i)
        self.offsets.append(j)
        self.offsets.append(k)


@enum.unique
class DataType(enum.IntEnum):
    """Data type identifier."""

    # IDs from gt4py
    INVALID = -1
    AUTO = 0
    DEFAULT = 1
    BOOL = 10
    INT8 = 11
    INT16 = 12
    INT32 = 14
    INT64 = 18
    FLOAT32 = 104
    FLOAT64 = 108


class Expr(Node):
    """A generic expression"""

    pass


class Stmt(Node):
    """A generic statement"""

    pass


class LiteralExpr(Expr):
    """An access to a literal"""

    value: str
    dtype: DataType

    def __init__(self, value: str, dtype=DataType.FLOAT64):
        self.value = value
        self.dtype = dtype


class FieldAccessExpr(Expr):
    """An access to a field"""

    def __init__(self, name: str, offset: AccessOffset):
        self.name = name
        self.offset = offset


class AssignmentStmt(Stmt):
    """Assignments"""

    def __init__(self, left: Expr, right: Expr):
        self.left = left
        self.right = right


class BinaryOp(Expr):
    """Any binary operator expression"""

    def __init__(self, left: Expr, right: Expr, operator: str):
        self.left = left
        self.right = right
        self.operator = operator


class FieldDecl(Stmt):
    """Declarations of fields"""

    pass


class IR(Node):
    def __init__(self):
        self.name: str = ""
        self.body: List[VerticalDomain] = []
        self.api_signature: List[str] = []
