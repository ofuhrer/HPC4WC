from yasmin.core import Dimension, Field, float64
from yasmin.ir.loop import For, Index, Literal, Load, Store


def test_loop_with_store() -> None:
    x = Dimension("x")

    u = Field("u", dims=(x,), dtype=float64)
    out = Field("out", dims=(x,), dtype=float64)

    i = Index("i")

    load = Load(field=u, indices=(i,))
    store = Store(field=out, indices=(i,), value=load)

    loop = For(index=i, lower=Literal(1), upper=Literal(15), body=(store,))

    assert loop.index == i
    assert loop.lower == Literal(1)
    assert loop.upper == Literal(15)
    assert loop.body == (store,)
