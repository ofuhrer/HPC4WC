from dataclasses import dataclass

from yasmin.analysis import (
    Halo,
    infer_execution_domain,
    infer_halo,
    infer_offset_bounds,
)
from yasmin.core import Field, Scalar
from yasmin.ir import loop, stencil


@dataclass(frozen=True, slots=True)
class Symbols:
    fields: tuple[Field, ...]
    scalars: tuple[Scalar, ...]


def _collect_symbols(statement: stencil.Assign) -> Symbols:
    fields: list[Field] = [statement.target.field]
    scalars: list[Scalar] = []

    def visit(node: stencil.Expr) -> None:
        match node:
            case stencil.FieldAccess(field=field):
                if field not in fields:
                    fields.append(field)

            case stencil.ScalarRef(scalar=scalar):
                if scalar not in scalars:
                    scalars.append(scalar)

            case stencil.BinaryExpr(lhs=lhs, rhs=rhs):
                visit(lhs)
                visit(rhs)

            case stencil.Literal():
                pass

            case _:
                raise TypeError(
                    f"Unsupported stencil expression type: {type(node).__name__}"
                )

    visit(statement.value)

    return Symbols(
        fields=tuple(fields),
        scalars=tuple(scalars),
    )


def _lower_binary_op(op: stencil.BinaryOp) -> loop.BinaryOp:
    match op:
        case stencil.BinaryOp.ADD:
            return loop.BinaryOp.ADD
        case stencil.BinaryOp.SUB:
            return loop.BinaryOp.SUB
        case stencil.BinaryOp.MUL:
            return loop.BinaryOp.MUL
        case stencil.BinaryOp.DIV:
            return loop.BinaryOp.DIV

    raise ValueError(f"Unsupported binary operator: {op}")


def _shift_index(index: loop.Index, offset: int) -> loop.Expr:
    if offset == 0:
        return index

    if offset > 0:
        return loop.BinaryExpr(
            loop.BinaryOp.ADD,
            index,
            loop.Literal(offset),
        )

    return loop.BinaryExpr(
        loop.BinaryOp.SUB,
        index,
        loop.Literal(-offset),
    )


def _lower_expr(expr: stencil.Expr, indices: tuple[loop.Index, ...]) -> loop.Expr:
    match expr:
        case stencil.Literal(value=value):
            return loop.Literal(value)

        case stencil.ScalarRef(scalar=scalar):
            return loop.ScalarRef(scalar)

        case stencil.FieldAccess(field=field, offsets=offsets):
            if len(offsets) != len(indices):
                raise ValueError(
                    "Field access and loop indices must have "
                    "the same number of dimensions"
                )

            return loop.Load(
                field=field,
                indices=tuple(
                    _shift_index(index, offset)
                    for index, offset in zip(indices, offsets, strict=True)
                ),
            )

        case stencil.BinaryExpr(op=op, lhs=lhs, rhs=rhs):
            return loop.BinaryExpr(
                op=_lower_binary_op(op),
                lhs=_lower_expr(lhs, indices),
                rhs=_lower_expr(rhs, indices),
            )

        case _:
            raise TypeError(
                f"Unsupported stencil expression type: {type(expr).__name__}"
            )


def _lower_statement(statement: stencil.Assign) -> tuple[loop.Stmt, ...]:
    if any(offset != 0 for offset in statement.target.offsets):
        raise NotImplementedError(
            "Loop lowering currently requires centered output accesses"
        )

    bounds_by_field = infer_offset_bounds(statement.value)

    if not bounds_by_field:
        raise ValueError("Expression contains no field accesses")

    halos = tuple(infer_halo(bounds) for bounds in bounds_by_field.values())

    ndim = len(statement.target.field.dims)

    if any(len(halo) != ndim for halo in halos):
        raise ValueError("Field accesses must have the same number of dimensions")

    halo: Halo = tuple(
        (
            max(field_halo[dim][0] for field_halo in halos),
            max(field_halo[dim][1] for field_halo in halos),
        )
        for dim in range(ndim)
    )

    domain = infer_execution_domain(halo)

    indices = tuple(loop.Index(dim.name) for dim in statement.target.field.dims)

    store = loop.Store(
        field=statement.target.field,
        indices=indices,
        value=_lower_expr(statement.value, indices),
    )

    body: tuple[loop.Stmt, ...] = (store,)

    for dim in reversed(range(ndim)):
        interval = domain.intervals[dim]

        lower_bound: loop.Expr = loop.Literal(interval.lower)
        upper_bound: loop.Expr = loop.Extent(
            statement.target.field,
            dim,
        )

        if interval.upper_trim != 0:
            upper_bound = loop.BinaryExpr(
                loop.BinaryOp.SUB,
                upper_bound,
                loop.Literal(interval.upper_trim),
            )

        body = (
            loop.For(
                index=indices[dim],
                lower=lower_bound,
                upper=upper_bound,
                body=body,
            ),
        )

    return body


def _collect_operator_symbols(
    operator: stencil.Operator,
) -> Symbols:
    fields: list[Field] = []
    scalars: list[Scalar] = []

    for statement in operator.statements:
        symbols = _collect_symbols(statement)

        for field in symbols.fields:
            if field not in fields:
                fields.append(field)

        for scalar in symbols.scalars:
            if scalar not in scalars:
                scalars.append(scalar)

    return Symbols(
        fields=tuple(fields),
        scalars=tuple(scalars),
    )


def lower(
    operator: stencil.Operator,
    *,
    name: str = "kernel",
) -> loop.Function:
    if not operator.statements:
        raise ValueError("Operator requires at least one assignment")

    body: list[loop.Stmt] = []

    for statement in operator.statements:
        body.extend(_lower_statement(statement))

    symbols = _collect_operator_symbols(operator)

    return loop.Function(
        name=name,
        fields=symbols.fields,
        scalars=symbols.scalars,
        body=tuple(body),
    )
