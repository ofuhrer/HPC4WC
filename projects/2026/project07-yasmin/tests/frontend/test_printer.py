import yasmin as yasi
from yasmin.core import float64


def test_print_operator() -> None:
    x = yasi.Dimension("x")
    y = yasi.Dimension("y")

    u = yasi.Field("u", dims=(x, y), dtype=float64)
    out = yasi.Field("out", dims=(x, y), dtype=float64)

    laplacian = yasi.Stencil(u[-1, 0] + u[1, 0] + u[0, -1] + u[0, 1] - 4 * u[0, 0])

    op = yasi.Operator(
        target=out[0, 0],
        value=laplacian,
    )

    yasi.print_stencil_ir(laplacian)
    yasi.print_stencil_ir(op)
    yasi.print_loop_ir(op)
