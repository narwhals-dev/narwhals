from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals._plan.demo as nwd
from narwhals._plan.common import is_expr
from narwhals.utils import Version
from tests.namespace_test import backends

if TYPE_CHECKING:
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
    ],
    ids=_ids_ir,
)
def test_select(
    expr: DummyExpr, expected: dict[str, Any], data_small: dict[str, Any]
) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    from narwhals._plan.dummy import DummyFrame

    frame = pa.table(data_small)
    df = DummyFrame.from_native(frame)
    result = df.select(expr).to_dict(as_series=False)
    assert result == expected
