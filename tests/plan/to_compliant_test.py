from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan.demo as nwd
from narwhals.utils import Version
from tests.namespace_test import backends

if TYPE_CHECKING:
    from narwhals._namespace import BackendName
    from narwhals._plan.dummy import DummyExpr


def _ids_ir(expr: DummyExpr) -> str:
    return repr(expr._ir)


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
