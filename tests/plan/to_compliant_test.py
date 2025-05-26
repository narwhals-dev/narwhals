from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals._plan.demo as nwd
from narwhals._plan.common import is_expr
from narwhals._plan.impl_arrow import evaluate as evaluate_pyarrow
from narwhals.utils import Version
from tests.namespace_test import backends

if TYPE_CHECKING:
    from narwhals._namespace import BackendName
    from narwhals._plan.dummy import DummyExpr


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
        (nwd.col("a"), ["A", "B", "A"]),
        (nwd.col("a", "b"), [["A", "B", "A"], [1, 2, 3]]),
        (nwd.lit(1), [1]),
        (nwd.lit(2.0), [2.0]),
        (nwd.lit(None, nw.String()), [None]),
    ],
    ids=_ids_ir,
)
def test_evaluate_pyarrow(expr: DummyExpr, expected: Any) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    data: dict[str, Any] = {
        "a": ["A", "B", "A"],
        "b": [1, 2, 3],
        "c": [9, 2, 4],
        "d": [8, 7, 8],
    }
    frame = pa.table(data)
    result = evaluate_pyarrow(expr._ir, frame)
    if len(result) == 1:
        assert result[0].to_pylist() == expected
    else:
        results = [col.to_pylist() for col in result]
        assert results == expected
