"""Adapted from upstream tests.

- https://github.com/pola-rs/polars/blob/1c11555550f8772dd4378b729069fd8c19e2d2dc/py-polars/tests/unit/expr/test_serde.py
- https://github.com//pola-rs/polars/blob/1c11555550f8772dd4378b729069fd8c19e2d2dc/py-polars/tests/unit/io/test_serde.py
"""

from __future__ import annotations

# ruff: noqa: S301
import io
import pickle
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from narwhals.exceptions import ComputeError
from tests.plan.utils import assert_expr_ir_equal

if TYPE_CHECKING:
    from narwhals._plan.meta import MetaNamespace

XFAIL_NOT_IMPL_SERDE = pytest.mark.xfail(
    reason="TODO @dangotbanned: Add `Expr.meta.serialize`",
    raises=(AttributeError, NotImplementedError),
)
XFAIL_NOT_IMPL_JSON = pytest.mark.xfail(
    reason="`deserialize(format='json')` is not yet implemented",
    raises=(NotImplementedError),
)


def cases() -> pytest.MarkDecorator:
    """Split out to get unique instances for the 2x shared tests.

    Same cases are duplicated upstream
    - https://github.com/pola-rs/polars/blob/1c11555550f8772dd4378b729069fd8c19e2d2dc/py-polars/tests/unit/expr/test_serde.py#L9-L18
    - https://github.com/pola-rs/polars/blob/1c11555550f8772dd4378b729069fd8c19e2d2dc/py-polars/tests/unit/expr/test_serde.py#L25-L34
    """
    return pytest.mark.parametrize(
        "expr",
        [
            nwp.col("foo").sum().over("bar"),
            nwp.col("foo").sum().over("bar", order_by=ncs.string()),
            nwp.col("foo").replace_strict({3: 1}, return_dtype=nw.UInt32),
            nwp.col("foo").replace_strict(
                [5, 6, 7], [8, 9, 10], default=nwp.nth(-1).last()
            ),
            nwp.col("foo").rolling_var(window_size=4, ddof=2),
            nwp.col("foo").rolling_mean(window_size=2),
            nwp.col("foo").sort_by(
                "bar", descending=[False, True], nulls_last=[True, False]
            ),
            nwp.col("foo").ewm_mean(alpha=0.5, min_samples=2, ignore_nulls=True),
        ],
    )


# TODO @dangotbanned: Add `Expr.meta.__eq__`
# See https://github.com/narwhals-dev/narwhals/blob/c3f00c85945230c945ac2eb90e4b9049949a0313/src/narwhals/_plan/meta.py#L124-L141
def meta_eq(actual: nwp.Expr, expected: nwp.Expr | MetaNamespace) -> None:
    assert_expr_ir_equal(actual._ir, expected._ir)


@cases()
@XFAIL_NOT_IMPL_SERDE
def test_expr_serde_roundtrip_binary(expr: nwp.Expr) -> None:  # pragma: no cover
    json = expr.meta.serialize(format="binary")  # type: ignore[attr-defined]
    round_tripped = nwp.Expr.deserialize(io.BytesIO(json), format="binary")
    meta_eq(round_tripped, expr)


@cases()
@XFAIL_NOT_IMPL_SERDE
def test_expr_serde_roundtrip_json(expr: nwp.Expr) -> None:  # pragma: no cover
    expr = nwp.col("foo").sum().over("bar")
    json = expr.meta.serialize(format="json")  # type: ignore[attr-defined]
    round_tripped = nwp.Expr.deserialize(io.StringIO(json), format="json")  # type: ignore[arg-type]
    meta_eq(round_tripped, expr)


def test_expr_deserialize_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        nwp.Expr.deserialize("abcdef")


@XFAIL_NOT_IMPL_JSON
def test_expr_deserialize_invalid_json() -> None:
    with pytest.raises(
        ComputeError, match="could not deserialize input into an expression"
    ):
        nwp.Expr.deserialize(io.StringIO("abcdef"), format="json")  # type: ignore[arg-type]


@XFAIL_NOT_IMPL_SERDE
def test_expression_json_13991() -> None:  # pragma: no cover
    expr = nwp.col("foo").cast(nw.Decimal(38, 10))
    json = expr.meta.serialize(format="json")  # type: ignore[attr-defined]

    round_tripped = nwp.Expr.deserialize(io.StringIO(json), format="json")  # type: ignore[arg-type]
    meta_eq(round_tripped, expr)


def test_serde_expression_5461() -> None:
    e = nwp.col("a").sqrt() / nwp.col("b").alias("c")

    roundtrip = pickle.loads(pickle.dumps(e))
    meta_eq(roundtrip, e.meta)


def test_pickling_simple_expression() -> None:
    e = nwp.col("foo").sum()
    roundtrip = pickle.loads(pickle.dumps(e))
    meta_eq(roundtrip, e)


def test_pickling_as_struct_11100() -> None:
    e = nwp.struct("a")
    roundtrip = pickle.loads(pickle.dumps(e))
    meta_eq(roundtrip, e)
