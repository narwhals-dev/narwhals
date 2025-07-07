from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._plan import demo as nwd, selectors as ndcs
from narwhals._plan.common import is_expr
from narwhals.exceptions import ComputeError
from narwhals.utils import Version
from tests.namespace_test import backends

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals._namespace import BackendName
    from narwhals._plan.dummy import DummyExpr


@pytest.fixture
def data_small() -> dict[str, Any]:
    return {"a": ["A", "B", "A"], "b": [1, 2, 3], "c": [9, 2, 4], "d": [8, 7, 8]}


def _ids_ir(expr: DummyExpr | Any) -> str:
    if is_expr(expr):
        return repr(expr._ir)
    return repr(expr)


@pytest.mark.parametrize(
    ("expr"),
    [
        nwd.col("a"),
        nwd.col("a", "b"),
        nwd.lit(1),
        nwd.lit(2.0),
        nwd.lit(None, nw.String()),
    ],
    ids=_ids_ir,
)
@backends
def test_to_compliant(backend: BackendName, expr: DummyExpr) -> None:
    pytest.importorskip(backend)
    namespace = Version.MAIN.namespace.from_backend(backend).compliant
    compliant_expr = expr._to_compliant(namespace)
    assert isinstance(compliant_expr, namespace._expr)


XFAIL_REQUIRES_BINARY_EXPR = pytest.mark.xfail(
    reason="Requires `BinaryExpr` implementation.", raises=NotImplementedError
)
XFAIL_REWRITE_SPECIAL_ALIASES = pytest.mark.xfail(
    reason="Bug in `meta` namespace impl", raises=ComputeError
)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwd.col("a"), {"a": ["A", "B", "A"]}),
        (nwd.col("a", "b"), {"a": ["A", "B", "A"], "b": [1, 2, 3]}),
        (nwd.lit(1), {"literal": [1]}),
        (nwd.lit(2.0), {"literal": [2.0]}),
        (nwd.lit(None, nw.String()), {"literal": [None]}),
        (nwd.col("a", "b").first(), {"a": ["A"], "b": [1]}),
        (nwd.col("d").max(), {"d": [8]}),
        ([nwd.len(), nwd.nth(3).last()], {"len": [3], "d": [8]}),
        (
            [nwd.len().alias("e"), nwd.nth(3).last(), nwd.nth(2)],
            {"e": [3, 3, 3], "d": [8, 8, 8], "c": [9, 2, 4]},
        ),
        (nwd.col("b").sort(descending=True).alias("b_desc"), {"b_desc": [3, 2, 1]}),
        pytest.param(
            nwd.col("c").filter(a="B"), {"c": [2]}, marks=XFAIL_REQUIRES_BINARY_EXPR
        ),
        (nwd.col("b").cast(nw.Float64()), {"b": [1.0, 2.0, 3.0]}),
        (nwd.lit(1).cast(nw.Float64()).alias("literal_cast"), {"literal_cast": [1.0]}),
        pytest.param(
            nwd.lit(1).cast(nw.Float64()).name.suffix("_cast"),
            {"literal_cast": [1.0]},
            marks=XFAIL_REWRITE_SPECIAL_ALIASES,
        ),
        ([ndcs.string().first(), nwd.col("b")], {"a": ["A", "A", "A"], "b": [1, 2, 3]}),
        (
            nwd.col("c", "d")
            .sort_by("a", "b", descending=[True, False])
            .cast(nw.Float32())
            .name.to_uppercase(),
            {"C": [2.0, 9.0, 4.0], "D": [7.0, 8.0, 8.0]},
        ),
    ],
    ids=_ids_ir,
)
def test_select(
    expr: DummyExpr | Sequence[DummyExpr],
    expected: dict[str, Any],
    data_small: dict[str, Any],
) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    from narwhals._plan.dummy import DummyFrame

    frame = pa.table(data_small)
    df = DummyFrame.from_native(frame)
    result = df.select(expr).to_dict(as_series=False)
    assert result == expected
