from yasmin.compiler.config import CppOptions
from yasmin.core import DType, Field, float32, float64, int32, int64
from yasmin.ir import loop
from yasmin.runtime.native import CompiledFunction, NativeArtifact, compile_cpp


class CppBackend:
    name = "cpp"

    def __init__(self, options: CppOptions | None = None) -> None:
        self.options = options or CppOptions()

    def source(self, function: loop.Function) -> str:
        lines: list[str] = []

        params = self._emit_parameters(function)

        lines.append("#include <cstdint>")
        lines.append("")
        lines.append(f'extern "C" void {function.name}({params}) {{')

        for top_level_index, statement in enumerate(function.body):
            lines.extend(
                self._emit_stmt(
                    statement,
                    indent=1,
                    top_level_index=top_level_index,
                )
            )

        lines.append("}")

        return "\n".join(lines)

    def compile(
        self,
        function: loop.Function,
    ) -> NativeArtifact:
        source = self.source(function)

        shared_library = compile_cpp(source)

        compiled_function = CompiledFunction(
            function=function,
            shared_library=shared_library,
        )

        return NativeArtifact(
            function=compiled_function,
            source=source,
        )

    def _emit_loop_prefix(
        self,
        statement: loop.For,
        *,
        indent: int,
        loop_depth: int,
        top_level_index: int,
    ) -> list[str]:
        return []

    def _emit_parameters(self, function: loop.Function) -> str:
        params: list[str] = []

        for field in function.fields:
            cpp_type = self._cpp_type(field.dtype)
            if not self.options.use_restrict:
                params.append(f"{cpp_type}* {field.name}")
            else:
                params.append(f"{cpp_type}* __restrict__ {field.name}")

        for field in function.fields:
            for dim in range(len(field.dims)):
                params.append(f"int {field.name}_shape_{dim}")

        for scalar in function.scalars:
            cpp_type = self._cpp_type(scalar.dtype)
            params.append(f"{cpp_type} {scalar.name}")

        return ", ".join(params)

    def _emit_stmt(
        self,
        statement: loop.Stmt,
        *,
        indent: int,
        loop_depth: int = 0,
        top_level_index: int,
    ) -> list[str]:
        prefix = "    " * indent

        match statement:
            case loop.Store(field=field, indices=indices, value=value):
                index = self._emit_flat_index(field, indices)
                return [f"{prefix}{field.name}[{index}] = {self._emit_expr(value)};"]

            case loop.For(index=index, lower=lower, upper=upper, body=body):
                lines = self._emit_loop_prefix(
                    statement,
                    indent=indent,
                    loop_depth=loop_depth,
                    top_level_index=top_level_index,
                )

                lines.append(
                    f"{prefix}for (int {index.name} = "
                    f"{self._emit_expr(lower)}; "
                    f"{index.name} < {self._emit_expr(upper)}; "
                    f"++{index.name}) {{"
                )

                for child in body:
                    lines.extend(
                        self._emit_stmt(
                            child,
                            indent=indent + 1,
                            loop_depth=loop_depth + 1,
                            top_level_index=top_level_index,
                        )
                    )

                lines.append(f"{prefix}}}")
                return lines

            case _:
                raise TypeError(
                    f"Unsupported loop statement type: {type(statement).__name__}"
                )

    def _emit_expr(self, expr: loop.Expr) -> str:
        match expr:
            case loop.Literal(value=value):
                return repr(value)

            case loop.ScalarRef(scalar=scalar):
                return scalar.name

            case loop.Index(name=name):
                return name

            case loop.Extent(field=field, dim=dim):
                return f"{field.name}_shape_{dim}"

            case loop.Load(field=field, indices=indices):
                index = self._emit_flat_index(field, indices)
                return f"{field.name}[{index}]"

            case loop.BinaryExpr(op=op, lhs=lhs, rhs=rhs):
                return f"({self._emit_expr(lhs)} {op.value} {self._emit_expr(rhs)})"

            case _:
                raise TypeError(
                    f"Unsupported loop expression type: {type(expr).__name__}"
                )

    def _emit_flat_index(
        self,
        field: Field,
        indices: tuple[loop.Expr, ...],
    ) -> str:
        if not indices:
            raise ValueError("Cannot flatten an empty index tuple")

        result = self._emit_expr(indices[0])

        for axis, index in enumerate(indices[1:], start=1):
            result = (
                f"({result} * {field.name}_shape_{axis} + {self._emit_expr(index)})"
            )

        return result

    def _cpp_type(self, dtype: DType) -> str:
        if dtype == float32:
            return "float"

        if dtype == float64:
            return "double"

        if dtype == int32:
            return "std::int32_t"

        if dtype == int64:
            return "std::int64_t"

        raise TypeError(f"Unsupported native dtype: {dtype.name}")
