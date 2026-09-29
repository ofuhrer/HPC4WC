import pytest

from yasmin.core import Dimension, float64
from yasmin.frontend import Field
from yasmin.ir import stencil as stencil_ir


def test_frontend_field_builds_field_access() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)

    access = u[-1, 0]

    assert access._ir == stencil_ir.FieldAccess(
        field=u._core,
        offsets=(-1, 0),
    )


def test_frontend_field_exposes_core_properties() -> None:
    x = Dimension("x")
    u = Field("u", dims=(x,), dtype=float64)

    assert u.name == "u"
    assert u.dims == (x,)
    assert u.dtype == float64


def test_frontend_field_validates_dimensionality() -> None:
    x = Dimension("x")
    y = Dimension("y")

    u = Field("u", dims=(x, y), dtype=float64)

    with pytest.raises(ValueError):
        _ = u[0]
