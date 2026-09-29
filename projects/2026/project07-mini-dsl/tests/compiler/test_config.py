from yasmin.compiler.config import CompileConfig, CppOptions, OpenMPOptions


def test_cpp_compile_config() -> None:
    options = CppOptions()

    config = CompileConfig(
        backend="cpp",
        options=options,
    )

    assert config.backend == "cpp"
    assert config.options is options


def test_openmp_compile_config() -> None:
    options = OpenMPOptions()

    config = CompileConfig(
        backend="openmp",
        options=options,
    )

    assert config.backend == "openmp"
    assert config.options is options
