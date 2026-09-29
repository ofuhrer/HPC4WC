from __future__ import annotations

from dataclasses import dataclass

from yasmin.analysis.omp_heuristic import parallel_config
from yasmin.analysis.system import available_cores
from yasmin.compiler.config import CppOptions, FieldShapes, OpenMPOptions
from yasmin.ir import loop


@dataclass(frozen=True, slots=True)
class OpenMPLoopConfig:
    parallelize: bool
    collapse: bool
    num_threads: int | None
    schedule: str | None
    schedule_chunk: int | None


@dataclass(frozen=True, slots=True)
class OpenMPConfig:
    cpp: CppOptions
    loop_configs: tuple[OpenMPLoopConfig | None, ...]
    extra_compile_flags: tuple[str, ...]


def _extent_of(
    expr: loop.Expr,
    *,
    shapes: FieldShapes,
) -> int | None:
    if isinstance(expr, loop.Literal):
        return int(expr.value)

    if isinstance(expr, loop.Extent):
        shape = shapes.get(expr.field)
        if shape is None:
            return None
        return shape[expr.dim]

    if isinstance(expr, loop.BinaryExpr):
        lhs = _extent_of(expr.lhs, shapes=shapes)
        rhs = _extent_of(expr.rhs, shapes=shapes)

        if lhs is None or rhs is None:
            return None

        if expr.op is loop.BinaryOp.SUB:
            return lhs - rhs

        if expr.op is loop.BinaryOp.ADD:
            return lhs + rhs

        if expr.op is loop.BinaryOp.MUL:
            return lhs * rhs

    return None


def _trip_count(
    statement: loop.For,
    *,
    shapes: FieldShapes,
) -> int | None:
    lower = _extent_of(statement.lower, shapes=shapes)
    upper = _extent_of(statement.upper, shapes=shapes)

    if lower is None or upper is None:
        return None

    return upper - lower


def resolve_openmp_config(
    function: loop.Function,
    *,
    options: OpenMPOptions,
    shapes: FieldShapes | None = None,
) -> OpenMPConfig:
    loop_configs: list[OpenMPLoopConfig | None] = []

    for statement in function.body:
        if not isinstance(statement, loop.For):
            loop_configs.append(None)
            continue

        is_perfectly_nested = len(statement.body) == 1 and isinstance(
            statement.body[0], loop.For
        )

        collapse = options.use_collapse and is_perfectly_nested
        num_threads = options.num_threads
        schedule_chunk = options.schedule_chunk
        parallelize = True

        if options.adaptive and shapes is not None:
            outer_extent = _trip_count(
                statement,
                shapes=shapes,
            )

            inner = statement.body[0] if is_perfectly_nested else None
            inner_extent = (
                _trip_count(inner, shapes=shapes)
                if isinstance(inner, loop.For)
                else None
            )

            if outer_extent is not None:
                heuristic = parallel_config(
                    outer_extent,
                    inner_extent,
                    max_threads=options.num_threads or available_cores(),
                    min_iters_per_thread=options.min_iters_per_thread,
                    collapse_enabled=collapse,
                    schedule_kind=options.schedule,
                )

                parallelize = heuristic.parallelize
                collapse = heuristic.collapse
                num_threads = heuristic.num_threads

                if schedule_chunk is None:
                    schedule_chunk = heuristic.schedule_chunk

        loop_configs.append(
            OpenMPLoopConfig(
                parallelize=parallelize,
                collapse=collapse,
                num_threads=num_threads,
                schedule=options.schedule,
                schedule_chunk=schedule_chunk,
            )
        )

    return OpenMPConfig(
        cpp=CppOptions(
            use_restrict=options.use_restrict,
        ),
        loop_configs=tuple(loop_configs),
        extra_compile_flags=options.extra_compile_flags,
    )
