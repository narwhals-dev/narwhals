from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from tests.plan.utils import DataFrame, Series, assert_equal_data, assert_equal_series

if TYPE_CHECKING:
    from narwhals._plan.typing import OneOrIterable
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"int": [-1, 1, None]}


@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        (
            [nwp.col("no_nan").fill_nan(None), ncs.last().fill_nan(None)],
            [None, 1.0, None],
        ),
        (
            [nwp.col("no_nan").fill_nan(3.0), nwp.col("float_nan").fill_nan(3.0)],
            [3.0, 1.0, None],
        ),
        (nwp.all().fill_nan(None), [None, 1.0, None]),
        (ncs.all().fill_nan(3.0), [3.0, 1.0, None]),
        (
            [
                nwp.col("no_nan"),
                nwp.col("float_nan").fill_nan(nwp.col("no_nan").max() * 6),
            ],
            [6.0, 1.0, None],
        ),
    ],
)
def test_fill_nan_expr(
    data: Data, exprs: OneOrIterable[nwp.Expr], expected: list[Any], dataframe: DataFrame
) -> None:
    base = nwp.col("int")
    df = dataframe(data).select(
        base.cast(nw.Float64).alias("no_nan"), (base**0.5).alias("float_nan")
    )
    result = df.select(exprs)
    assert_equal_data(result, {"no_nan": [-1.0, 1.0, None], "float_nan": expected})
    assert result.get_column("float_nan").null_count() == expected.count(None)


def test_fill_nan_expr_series(data: Data, dataframe: DataFrame) -> None:
    expected = [55.5, 1.0, None]
    base = nwp.col("int")
    df = dataframe(data).select(
        base.cast(nw.Float64).alias("no_nan"), (base**0.5).alias("float_nan")
    )
    ser = df.to_series().from_iterable(
        [55.5, -100, -200], backend=dataframe.implementation
    )

    result = df.select(ncs.numeric().fill_nan(nwp.lit(ser)))
    assert_equal_data(result, {"no_nan": [-1.0, 1.0, None], "float_nan": expected})
    assert result.get_column("float_nan").null_count() == expected.count(None)


def test_fill_nan_series(data: Data, series: Series) -> None:
    fill = series([1.23, None, None])
    ser = (
        fill.to_frame()
        .from_dict(data, backend=series.implementation)
        .select(float_nan=nwp.col("int") ** 0.5)
        .get_column("float_nan")
    )
    result = ser.fill_nan(999)
    assert_equal_series(result, [999.0, 1.0, None], "float_nan")
    result = ser.fill_nan(fill)
    assert_equal_series(result, [1.23, 1.0, None], "float_nan")
