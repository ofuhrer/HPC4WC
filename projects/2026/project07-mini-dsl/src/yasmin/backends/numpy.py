from collections.abc import Mapping
from typing import Any

import numpy.typing as npt

from yasmin.analysis import (
    Halo,
    infer_execution_domain,
    infer_halo,
    infer_offset_bounds,
)
from yasmin.backends.base import ScalarValue
from yasmin.core import Field, Scalar
from yasmin.ir.stencil import (
    BinaryExpr,
    BinaryOp,
    Expr,
    FieldAccess,
    Literal,
    Operator,
    ScalarRef,
)

Array = npt.NDArray[Any]


class NumPyBackend:
    name = "numpy"

    def execute(
        self,
        operator: Operator,
        *,
        fields: Mapping[Field, Array],
        scalars: Mapping[Scalar, ScalarValue] | None = None,
    ) -> None:
        scalar_bindings = scalars or {}

        for statement in operator.statements:
            bounds_by_field = infer_offset_bounds(statement.value)

            if not bounds_by_field:
                raise ValueError("Expression contains no field accesses")

            halos = tuple(infer_halo(bounds) for bounds in bounds_by_field.values())
            halo = self._merge_halos(halos)
            domain = infer_execution_domain(halo)

            target = statement.target
            target_array = fields[target.field]

            target_slices = tuple(
                slice(
                    interval.lower,
                    target_array.shape[dim] - interval.upper_trim,
                )
                for dim, interval in enumerate(domain.intervals)
            )

            target_array[target_slices] = self._eval_expr(
                statement.value,
                fields=fields,
                scalars=scalar_bindings,
                halo=halo,
            )

    def _eval_expr(
        self,
        expr: Expr,
        *,
        fields: Mapping[Field, Array],
        scalars: Mapping[Scalar, ScalarValue],
        halo: Halo,
    ) -> Array | ScalarValue:
        match expr:
            case Literal(value=value):
                return value

            case ScalarRef(scalar=scalar):
                return scalars[scalar]

            case FieldAccess(field=field, offsets=offsets):
                array = fields[field]
                slices = self._slice_for_offset(
                    array.shape,
                    halo,
                    offsets,
                )
                return array[slices]

            case BinaryExpr(op=op, lhs=lhs, rhs=rhs):
                lhs_value = self._eval_expr(
                    lhs,
                    fields=fields,
                    scalars=scalars,
                    halo=halo,
                )
                rhs_value = self._eval_expr(
                    rhs,
                    fields=fields,
                    scalars=scalars,
                    halo=halo,
                )

                match op:
                    case BinaryOp.ADD:
                        return lhs_value + rhs_value
                    case BinaryOp.SUB:
                        return lhs_value - rhs_value
                    case BinaryOp.MUL:
                        return lhs_value * rhs_value
                    case BinaryOp.DIV:
                        return lhs_value / rhs_value

                raise ValueError(f"Unsupported binary operator: {op}")

            case _:
                raise TypeError(f"Unsupported expression type: {type(expr).__name__}")

    @staticmethod
    def _slice_for_offset(
        shape: tuple[int, ...],
        halo: Halo,
        offsets: tuple[int, ...],
    ) -> tuple[slice, ...]:
        if not (len(shape) == len(halo) == len(offsets)):
            raise ValueError(
                "Shape, halo, and offsets must have the same dimensionality"
            )

        return tuple(
            slice(
                left + offset,
                size - right + offset,
            )
            for size, (left, right), offset in zip(
                shape,
                halo,
                offsets,
                strict=True,
            )
        )

    @staticmethod
    def _merge_halos(halos: tuple[Halo, ...]) -> Halo:
        if not halos:
            raise ValueError("Cannot merge empty set of halos")

        ndim = len(halos[0])

        if any(len(halo) != ndim for halo in halos):
            raise ValueError("All halos must have the same number of dimensions")

        return tuple(
            (
                max(halo[dim][0] for halo in halos),
                max(halo[dim][1] for halo in halos),
            )
            for dim in range(ndim)
        )
