from yasmin.core import Dimension, Field, Scalar, float64
from yasmin.ir.loop import For, Function, Index, Literal


def test_function() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)
    alpha = Scalar("alpha", dtype=float64)

    i = Index("i")

    loop = For(index=i, lower=Literal(1), upper=Literal(15), body=())

    function = Function(
        name="my_function", fields=(u, out), scalars=(alpha,), body=(loop,)
    )

    assert function.name == "my_function"
    assert function.fields == (u, out)
    assert function.scalars == (alpha,)
    assert function.body == (loop,)
