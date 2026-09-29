from yasmin.core import Dimension, Field, Scalar, float64


def test_core_declarations() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field(
        name="u",
        dims=(x, y),
        dtype=float64,
    )

    alpha = Scalar(
        name="alpha",
        dtype=float64,
    )

    assert u.name == "u"
    assert u.dims == (x, y)
    assert u.dtype is float64

    assert alpha.name == "alpha"
    assert alpha.dtype is float64
