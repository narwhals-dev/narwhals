from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Iterable

import pytest

import narwhals as nw
import narwhals._plan.demo as nwd
from narwhals._plan import boolean
from narwhals._plan import functions as F  # noqa: N812
from narwhals._plan.common import ExprIR
from narwhals._plan.common import Function
from narwhals._plan.dummy import DummyExpr
from narwhals._plan.expr import FunctionExpr

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


@pytest.mark.parametrize(
    ("function", "ir_node"),
    [
        (nwd.all_horizontal, boolean.AllHorizontal),
        (nwd.any_horizontal, boolean.AnyHorizontal),
        (nwd.sum_horizontal, F.SumHorizontal),
        (nwd.min_horizontal, F.MinHorizontal),
        (nwd.max_horizontal, F.MaxHorizontal),
        (nwd.mean_horizontal, F.MeanHorizontal),
    ],
)
@pytest.mark.parametrize(
    "args",
    [
        ("a", "b", "c"),
        (["a", "b", "c"]),
        (nwd.col("d", "e", "f"), nwd.col("g"), "q", nwd.nth(9)),
        ((nwd.lit(1),)),
        ([nwd.lit(1), nwd.lit(2), nwd.lit(3)]),
    ],
)
def test_function_expr_horizontal(
    function: Callable[..., DummyExpr],
    ir_node: type[Function],
    args: Seq[IntoExpr | Iterable[IntoExpr]],
) -> None:
    variadic = function(*args)
    sequence = function(args)
    assert isinstance(variadic, DummyExpr)
    assert isinstance(sequence, DummyExpr)
    variadic_node = variadic._ir
    sequence_node = sequence._ir
    unrelated_node = nwd.lit(1)._ir
    assert isinstance(variadic_node, FunctionExpr)
    assert isinstance(variadic_node.function, ir_node)
    assert variadic_node == sequence_node
    assert sequence_node != unrelated_node
