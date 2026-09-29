import yasmin as yasi
from yasmin.core import Dimension as CoreDimension


def test_dimension_creates_single_dimension() -> None:
    x = yasi.Dimension("x")

    assert isinstance(x, CoreDimension)
    assert x.name == "x"


def test_dimension_creates_multiple_dimensions() -> None:
    x, y = yasi.Dimension("x", "y")

    assert isinstance(x, CoreDimension)
    assert isinstance(y, CoreDimension)
    assert x.name == "x"
    assert y.name == "y"


def test_dimensions_can_be_shared_between_fields() -> None:
    x, y = yasi.Dimension("x", "y")

    u = yasi.Field(
        "u",
        dims=(x, y),
        dtype=yasi.float64,
    )
    v = yasi.Field(
        "v",
        dims=(x, y),
        dtype=yasi.float64,
    )

    assert u.dims == (x, y)
    assert v.dims == (x, y)
