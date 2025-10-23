"""Tests adapted from [upstream].

[upstream]: https://github.com/pola-rs/polars/blob/84d66e960e3d462811f0575e0a6e4e78e34c618c/py-polars/tests/unit/test_selectors.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import expressions as ir, selectors as ncs
from narwhals._plan._expansion import prepare_projection
from narwhals._plan._parse import parse_into_seq_of_expr_ir

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._plan.schema import IntoFrozenSchema
    from narwhals._plan.typing import IntoExpr, Seq
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType


XFAIL_NESTED_INNER_SELECTOR = pytest.mark.xfail(
    reason="Bug causing the inner selector to always be falsy?", raises=AssertionError
)


def _schema(mapping: Mapping[str, IntoDType]) -> nw.Schema:
    # NOTE: Runtime isn't as strict as the annotation which requires instantiated `DType`
    return nw.Schema(mapping)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def schema_nested_1() -> nw.Schema:
    return _schema(
        {
            "a": nw.Int32(),
            "b": nw.List(nw.Int32()),
            "c": nw.List(nw.UInt32),
            "d": nw.Array(nw.Int32, 3),
            "e": nw.List(nw.String),
            "f": nw.Struct({"x": nw.Int32}),
        }
    )


@pytest.fixture(scope="module")
def schema_nested_2() -> nw.Schema:
    return _schema(
        {
            "a": nw.Int32(),
            "b": nw.Array(nw.Int32, 4),
            "c": nw.Array(nw.UInt32, 4),
            "d": nw.Array(nw.Int32, 3),
            "e": nw.List(nw.Int32),
            "f": nw.Array(nw.String, 4),
            "g": nw.Struct({"x": nw.Int32}),
        }
    )


@pytest.fixture(scope="module")
def schema_non_nested() -> nw.Schema:
    return _schema(
        {
            "abc": nw.UInt16(),
            "bbb": nw.UInt32(),
            "cde": nw.Float64(),
            "def": nw.Float32(),
            "eee": nw.Boolean(),
            "fgg": nw.Boolean(),
            "ghi": nw.Time(),
            "JJK": nw.Date(),
            "Lmn": nw.Duration(),
            "opp": nw.Datetime("ms"),
            "qqR": nw.String(),
        }
    )


def project_named_irs(*exprs: IntoExpr, schema: IntoFrozenSchema) -> Seq[ir.NamedIR]:
    named_irs, _ = prepare_projection(parse_into_seq_of_expr_ir(*exprs), schema=schema)
    return named_irs


def project_names(*exprs: IntoExpr, schema: IntoFrozenSchema) -> Seq[str]:
    named_irs = project_named_irs(*exprs, schema=schema)
    return tuple(e.name for e in named_irs)


def test_selector_all(schema_non_nested: nw.Schema) -> None:
    schema = schema_non_nested
    names = tuple(schema.names())

    assert project_names(ncs.all(), schema=schema) == names
    assert project_names(~ncs.all(), schema=schema) == ()
    assert project_names(~(~ncs.all()), schema=schema) == names
    assert project_names(ncs.all() & nwp.col("abc"), schema=schema) == ("abc",)


def test_selector_by_dtype(schema_non_nested: nw.Schema) -> None:
    schema = schema_non_nested

    assert project_names(ncs.boolean() | ncs.by_dtype(nw.UInt16), schema=schema) == (
        "abc",
        "eee",
        "fgg",
    )

    assert project_names(
        ~ncs.by_dtype(
            nw.Int8,
            nw.Int16,
            nw.Int32,
            nw.Int64,
            nw.Int128,
            nw.UInt8,
            nw.UInt16,
            nw.UInt32,
            nw.UInt64,
            nw.UInt128,
            nw.Date,
            nw.Datetime,
            nw.Duration,
            nw.Time,
        ),
        schema=schema,
    ) == ("cde", "def", "eee", "fgg", "qqR")

    assert project_names(
        ncs.by_dtype(nw.Datetime("ns"), nw.Float32, nw.UInt32, nw.Date), schema=schema
    ) == ("bbb", "def", "JJK")

    # cover timezones and decimal
    schema_x = _schema(
        {
            "idx": nw.Decimal(),
            "dt1": nw.Datetime("ms"),
            "dt2": nw.Datetime(time_zone="Asia/Tokyo"),
        }
    )
    assert project_names(ncs.by_dtype(nw.Decimal), schema=schema_x) == ("idx",)

    assert project_names(
        ncs.by_dtype(nw.Datetime(time_zone="Asia/Tokyo")), schema=schema_x
    ) == ("dt2",)

    assert project_names(ncs.by_dtype(nw.Datetime("ms", None)), schema=schema_x) == (
        "dt1",
    )

    assert project_names(ncs.by_dtype(nw.Datetime), schema=schema_x) == ("dt1", "dt2")


@pytest.mark.xfail(
    reason="Bug: Forgot to handle this during construction", raises=StopIteration
)
def test_selector_by_dtype_empty(
    schema_non_nested: nw.Schema,
) -> None:  # pragma: no cover
    # empty selection selects nothing
    assert project_names(ncs.by_dtype(), schema=schema_non_nested) == ()
    assert project_names(ncs.by_dtype([]), schema=schema_non_nested) == ()


@pytest.mark.xfail(reason="Bug: Forgot to handle this during construction")
def test_selector_by_dtype_invalid_input() -> None:
    with pytest.raises(TypeError):
        ncs.by_dtype(999)  # type: ignore[arg-type]


@XFAIL_NESTED_INNER_SELECTOR
def test_list_selector(schema_nested_1: Mapping[str, DType]) -> None:  # pragma: no cover
    schema = schema_nested_1
    assert project_names(ncs.list(), schema=schema) == ("b", "c", "e")

    # NOTE: bug here
    assert project_names(ncs.list(inner=ncs.numeric()), schema=schema) == ("b", "c")
    assert project_names(ncs.list(inner=ncs.string()), schema=schema) == ("e",)

    # NOTE: Not implemented
    with pytest.raises(
        TypeError, match=r"expected datatype based expression got.+by_name\("
    ):
        project_named_irs(ncs.list(inner=ncs.by_name("???")), schema=schema)


@XFAIL_NESTED_INNER_SELECTOR
def test_array_selector(schema_nested_2: Mapping[str, DType]) -> None:  # pragma: no cover
    schema = schema_nested_2
    assert project_names(ncs.array(), schema=schema) == ("b", "c", "d", "f")
    assert project_names(ncs.array(size=4), schema=schema) == ("b", "c", "f")

    # NOTE: bug here
    assert project_names(ncs.array(inner=ncs.numeric()), schema=schema) == ("b", "c", "d")
    assert project_names(ncs.array(inner=ncs.string()), schema=schema) == ("f",)

    # NOTE: Not implemented
    with pytest.raises(
        TypeError, match=r"expected datatype based expression got.+by_name\("
    ):
        project_named_irs(ncs.array(inner=ncs.by_name("???")), schema=schema)
