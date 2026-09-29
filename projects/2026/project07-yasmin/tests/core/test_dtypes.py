from yasmin.core import dtype, float64


def test_dtype_from_string() -> None:
    assert dtype("float64") is float64


def test_dtype_identity() -> None:
    assert dtype(float64) is float64
