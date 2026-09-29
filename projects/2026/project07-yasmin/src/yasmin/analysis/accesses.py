from yasmin.ir.stencil import BinaryExpr, Expr, FieldAccess, Literal, ScalarRef


def collect_accesses(expr: Expr) -> tuple[FieldAccess, ...]:
    match expr:
        case FieldAccess():
            return (expr,)

        case BinaryExpr(lhs=lhs, rhs=rhs):
            return collect_accesses(lhs) + collect_accesses(rhs)

        case Literal() | ScalarRef():
            return ()

        case _:
            raise TypeError(f"Unsupported expression type: {type(expr).__name__}")
