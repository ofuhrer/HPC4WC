from yasmin.backends.cpp import CppBackend
from yasmin.compiler.openmp import OpenMPConfig
from yasmin.ir import loop
from yasmin.runtime.native import CompiledFunction, NativeArtifact, compile_cpp


class OpenMPBackend(CppBackend):
    name = "openmp"

    def __init__(self, config: OpenMPConfig) -> None:
        super().__init__(options=config.cpp)
        self.config = config

    def compile(
        self,
        function: loop.Function,
    ) -> NativeArtifact:
        source = self.source(function)

        shared_library = compile_cpp(
            source,
            extra_flags=("-fopenmp", *self.config.extra_compile_flags),
        )

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
        if loop_depth != 0:
            return []

        config = self.config.loop_configs[top_level_index]

        if config is None or not config.parallelize:
            return []

        prefix = "    " * indent

        clauses = ["parallel", "for"]

        if config.collapse:
            clauses.append("collapse(2)")

        if config.schedule is not None:
            if config.schedule_chunk is not None:
                clauses.append(f"schedule({config.schedule}, {config.schedule_chunk})")
            else:
                clauses.append(f"schedule({config.schedule})")

        if config.num_threads is not None:
            clauses.append(f"num_threads({config.num_threads})")

        return [f"{prefix}#pragma omp {' '.join(clauses)}"]
