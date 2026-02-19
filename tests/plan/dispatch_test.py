from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("pyarrow")
import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import expressions as ir, selectors as ncs
from narwhals._plan._dispatch import get_dispatch_name
from tests.plan.utils import assert_equal_data, dataframe, named_ir, re_compile

if TYPE_CHECKING:
    import pyarrow as pa

    from narwhals._plan.dataframe import DataFrame


@pytest.fixture
def data() -> dict[str, Any]:
    return {
        "a": [12.1, None, 4.0],
        "b": [42, 10, None],
        "c": [4, 5, 6],
        "d": ["play", "swim", "walk"],
    }


@pytest.fixture
def df(data: dict[str, Any]) -> DataFrame[pa.Table, pa.ChunkedArray[Any]]:
    return dataframe(data)


def test_dispatch(df: DataFrame[pa.Table, pa.ChunkedArray[Any]]) -> None:
    implemented_full = nwp.col("a").is_null()
    forgot_to_expand = (named_ir("howdy", nwp.nth(3, 4).first()),)
    aliased_after_expand: tuple[ir.NamedIR] = (
        ir.NamedIR.from_ir(ir.col("a").alias("b")),
    )

    assert_equal_data(df.select(implemented_full), {"a": [False, True, False]})

    missing_backend = r"ewm_mean.+is not yet implemented for"
    with pytest.raises(NotImplementedError, match=missing_backend):
        df.select(nwp.col("c").ewm_mean())

    missing_protocol = re_compile(
        r"dt\.offset_by.+has not been implemented.+compliant.+"
        r"Hint.+try adding.+CompliantExpr\.dt\.offset_by\(\)"
    )
    with pytest.raises(NotImplementedError, match=missing_protocol):
        df.select(nwp.col("d").dt.offset_by("1d"))

    with pytest.raises(
        TypeError,
        match=re_compile(r"RootSelector.+not.+appear.+compliant.+expand.+expr.+first"),
    ):
        df._compliant.select(forgot_to_expand)

    bad = re.escape("col('a').alias('b')")
    with pytest.raises(TypeError, match=re_compile(rf"Alias.+not.+appear.+got.+{bad}")):
        df._compliant.select(aliased_after_expand)

    # Not a narwhals method, to make sure this doesn't allow arbitrary calls
    with pytest.raises(AttributeError):
        nwp.col("a").max().to_physical()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.col("a"), "col"),
        (nwp.col("a").min().over("b"), "over"),
        (nwp.col("a").first().over(order_by="b"), "over_ordered"),
        (nwp.all_horizontal("a", "b", nwp.nth(4, 5, 6)), "all_horizontal"),
        (nwp.int_range(10), "int_range"),
        (nwp.col("a") + nwp.col("b") + 10, "binary_expr"),
        (nwp.when(nwp.col("c")).then(5).when(nwp.col("d")).then(20), "ternary_expr"),
        (nwp.col("a").rolling_sum(2), "rolling_expr"),
        (nwp.col("a").cum_sum(), "cum_sum"),
        (nwp.col("a").cat.get_categories(), "cat.get_categories"),
        (nwp.col("a").dt.timestamp(), "dt.timestamp"),
        (nwp.col("a").dt.replace_time_zone(None), "dt.replace_time_zone"),
        (nwp.col("a").list.len(), "list.len"),
        (nwp.col("a").cast(nw.String).str.starts_with("something"), ("str.starts_with")),
        (nwp.col("a").str.slice(1), ("str.slice")),
        (nwp.col("a").str.head(), ("str.slice")),
        (nwp.col("a").str.tail(), ("str.slice")),
        (nwp.col("a").struct.field("b"), "struct.field"),
        (nwp.mean("a"), "mean"),
        (nwp.nth(1).first(), "first"),
        (nwp.col("a").sum(), "sum"),
        (~nwp.col("a"), "not_"),
        (nwp.col("a").drop_nulls().arg_min(), "arg_min"),
        (nwp.col("a").map_batches(lambda x: x), "map_batches"),
        pytest.param(nwp.col("a").alias("b"), "Alias", id="no_dispatch-Alias"),
        pytest.param(ncs.string(), "RootSelector", id="no_dispatch-RootSelector"),
    ],
)
def test_dispatch_name(expr: nwp.Expr, expected: str) -> None:
    assert get_dispatch_name(expr._ir) == expected
