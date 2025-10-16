from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("pyarrow")
import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import expressions as ir, selectors as ncs
from narwhals._plan._dispatch import get_dispatch_name
from tests.plan.utils import assert_equal_data, dataframe, named_ir

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
    only_at_compliant_level = nwp.col("c").ewm_mean()
    only_at_narwhals_level = nwp.col("d").str.contains("a")
    forgot_to_expand = (named_ir("howdy", nwp.nth(3, 4).first()),)
    aliased_after_expand: tuple[ir.NamedIR[Any]] = (
        ir.NamedIR.from_ir(ir.col("a").alias("b")),
    )

    pattern_expand = re.compile(
        r"IndexColumns.+not.+appear.+compliant.+expand.+expr.+first",
        re.DOTALL | re.IGNORECASE,
    )
    bad = re.escape("col('a').alias('b')")
    pattern_aliased_after_expand = re.compile(
        rf"Alias.+not.+appear.+got.+{bad}", re.DOTALL | re.IGNORECASE
    )

    assert_equal_data(df.select(implemented_full), {"a": [False, True, False]})

    with pytest.raises(NotImplementedError, match=r"ewm_mean"):
        df.select(only_at_compliant_level)

    with pytest.raises(NotImplementedError, match=r"str\.contains"):
        df.select(only_at_narwhals_level)

    with pytest.raises(TypeError, match=pattern_expand):
        df._compliant.select(forgot_to_expand)

    with pytest.raises(TypeError, match=pattern_aliased_after_expand):
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
        (nwp.col("a").cast(nw.String).str.starts_with("something"), ("str.starts_with")),
        (nwp.mean("a"), "mean"),
        (nwp.nth(1).first(), "first"),
        (nwp.col("a").sum(), "sum"),
        (nwp.col("a").drop_nulls().arg_min(), "arg_min"),
        pytest.param(nwp.col("a").alias("b"), "Alias", id="no_dispatch-Alias"),
        pytest.param(ncs.string(), "RootSelector", id="no_dispatch-RootSelector"),
    ],
)
def test_dispatch_name(expr: nwp.Expr, expected: str) -> None:
    assert get_dispatch_name(expr._ir) == expected
