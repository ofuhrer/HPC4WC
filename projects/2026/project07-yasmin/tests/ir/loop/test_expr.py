from yasmin.core import Dimension, Field, float64
from yasmin.ir.loop import BinaryExpr, BinaryOp, Index, Literal, Load


def test_load_with_index_expression() -> None:
    x = Dimension("x")
    u = Field("u", dims=(x,), dtype=float64)

    i = Index("i")

    index_expr = BinaryExpr(
        BinaryOp.ADD,
        i,
        Literal(1),
    )

    load = Load(field=u, indices=(index_expr,))

    assert load.field == u
    assert load.indices == (index_expr,)
