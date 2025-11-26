from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals._plan as nwp
from narwhals.exceptions import DuplicateError
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tests.conftest import Data

pytest.importorskip("pyarrow")


@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        pytest.param(
            nwp.col("user").struct.field("id"), {"id": ["0", "1"]}, id="field-single"
        ),
        pytest.param(
            [nwp.col("user").struct.field("id"), nwp.col("user").struct.field("name")],
            {"id": ["0", "1"], "name": ["john", "jane"]},
            marks=pytest.mark.xfail(
                raises=DuplicateError,
                reason="TODO: Handle `FieldByName` correctly during `Expr` expansion",
            ),
            id="multiple-fields-same-root",
        ),
        pytest.param(
            nwp.col("user").struct.field("id").name.keep(),
            {"user": ["0", "1"]},
            marks=pytest.mark.xfail(
                raises=KeyError,
                reason="TODO: Handle `FieldByName` correctly during `Expr` expansion",
            ),
            id="field-single-keep-root",
        ),
    ],
)
def test_struct_field(exprs: nwp.Expr | Iterable[nwp.Expr], expected: Data) -> None:
    data = {"user": [{"id": "0", "name": "john"}, {"id": "1", "name": "jane"}]}
    result = dataframe(data).select(exprs)
    assert_equal_data(result, expected)  # pragma: no cover
