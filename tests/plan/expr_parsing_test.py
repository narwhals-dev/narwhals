from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterable

import pytest

import narwhals as nw
import narwhals._plan.demo as nwd
from narwhals._plan.common import ExprIR

if TYPE_CHECKING:
    from narwhals._plan.common import IntoExpr
    from narwhals._plan.common import Seq


@pytest.mark.parametrize(
    ("exprs", "named_exprs"),
    [
        ([nwd.col("a")], {}),
        (["a"], {}),
        ([], {"a": "b"}),
        ([], {"a": nwd.col("b")}),
        (["a", "b", nwd.col("c", "d", "e")], {"g": nwd.lit(1)}),
        ([["a", "b", "c"]], {"q": nwd.lit(5, nw.Int8())}),
        (
            [[nwd.nth(1), nwd.nth(2, 3, 4)]],
            {"n": nwd.col("p").count(), "other n": nwd.len()},
        ),
    ],
)
def test_parsing(
    exprs: Seq[IntoExpr | Iterable[IntoExpr]], named_exprs: dict[str, IntoExpr]
) -> None:
    assert all(
        isinstance(node, ExprIR) for node in nwd.select_context(*exprs, **named_exprs)
    )
