from __future__ import annotations

from dataclasses import dataclass

from yasmin.ir import stencil


class SymbolicExpr:
    def _as_ir(self) -> stencil.Expr:
        raise NotImplementedError

    def __add__(self, other: SymbolicExpr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.ADD,
                self._as_ir(),
                _as_ir_expr(other),
            )
        )

    def __radd__(self, other: SymbolicExpr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.ADD,
                _as_ir_expr(other),
                self._as_ir(),
            )
        )

    def __sub__(self, other: SymbolicExpr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.SUB,
                self._as_ir(),
                _as_ir_expr(other),
            )
        )

    def __rsub__(self, other: SymbolicExpr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.SUB,
                _as_ir_expr(other),
                self._as_ir(),
            )
        )

    def __mul__(self, other: SymbolicExpr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.MUL,
                self._as_ir(),
                _as_ir_expr(other),
            )
        )

    def __rmul__(self, other: SymbolicExpr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.MUL,
                _as_ir_expr(other),
                self._as_ir(),
            )
        )

    def __truediv__(self, other: SymbolicExpr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.DIV,
                self._as_ir(),
                _as_ir_expr(other),
            )
        )

    def __rtruediv__(self, other: SymbolicExpr | int | float) -> Expr:
        return Expr(
            stencil.BinaryExpr(
                stencil.BinaryOp.DIV,
                _as_ir_expr(other),
                self._as_ir(),
            )
        )


@dataclass(frozen=True, slots=True)
class Expr(SymbolicExpr):
    _ir: stencil.Expr

    def _as_ir(self) -> stencil.Expr:
        return self._ir


def _as_ir_expr(
    value: SymbolicExpr | int | float,
) -> stencil.Expr:
    if isinstance(value, SymbolicExpr):
        return value._as_ir()

    return stencil.Literal(value)


def _as_symbolic_expr(
    value: SymbolicExpr | int | float,
) -> SymbolicExpr:
    if isinstance(value, SymbolicExpr):
        return value

    return Expr(stencil.Literal(value))
