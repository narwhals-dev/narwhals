# Testing dataframe-agnostic code

[`narwhals.testing`](../api-reference/testing.md) provides `assert_frame_equal` and
`assert_series_equal`. Both take Narwhals objects, not native ones, and raise an
`AssertionError` showing each side of the comparison.

## Comparing frames and series

Say this is the function under test:

```python exec="yes" source="above" session="how-to-testing"
import narwhals as nw
from narwhals.typing import IntoFrameT


def with_total(df_native: IntoFrameT) -> IntoFrameT:
    return (
        nw.from_native(df_native)
        .with_columns(total=nw.col("price") * nw.col("quantity"))
        .to_native()
    )
```

Wrap both the result and the expectation with `nw.from_native`, then compare:

```python exec="yes" source="material-block" result="python" session="how-to-testing"
import polars as pl
from narwhals.testing import assert_frame_equal

data = {"price": [1.5, 2.0], "quantity": [2, 3]}

result = nw.from_native(with_total(pl.DataFrame(data)))
expected = nw.from_native(pl.DataFrame({**data, "total": [3.0, 6.0]}))

assert_frame_equal(result, expected)
print("frames match")
```

When they differ, the error names the column:

```python exec="yes" source="material-block" result="python" session="how-to-testing"
wrong = nw.from_native(pl.DataFrame({**data, "total": [3.0, 6.5]}))

try:
    assert_frame_equal(result, wrong)
except AssertionError as exc:
    print(exc)
```

`assert_series_equal` compares a single column:

```python exec="yes" source="material-block" result="python" session="how-to-testing"
from narwhals.testing import assert_series_equal

assert_series_equal(result["total"], expected["total"])
print("series match")
```

## Running a test against several backends

Narwhals parametrises its own suite over a dict of constructors: functions which take a
column-oriented `dict` and return a native frame. Each one imports its backend lazily, so
a backend which isn't installed is skipped rather than collected. The same pattern works
downstream:

```py
# conftest.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import pytest

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame

    Data = dict[str, list[Any]]
    Constructor = Callable[[Data], IntoFrame]


def pandas_constructor(obj: Data) -> IntoFrame:
    pytest.importorskip("pandas")
    import pandas as pd

    return pd.DataFrame(obj)


def polars_eager_constructor(obj: Data) -> IntoFrame:
    pytest.importorskip("polars")
    import polars as pl

    return pl.DataFrame(obj)


def polars_lazy_constructor(obj: Data) -> IntoFrame:
    pytest.importorskip("polars")
    import polars as pl

    return pl.LazyFrame(obj)


def duckdb_lazy_constructor(obj: Data) -> IntoFrame:
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    import duckdb
    import pyarrow as pa

    _df = pa.table(obj)
    return duckdb.sql("select * from _df")


CONSTRUCTORS: dict[str, Constructor] = {
    "pandas": pandas_constructor,
    "polars[eager]": polars_eager_constructor,
    "polars[lazy]": polars_lazy_constructor,
    "duckdb": duckdb_lazy_constructor,
}


@pytest.fixture(params=list(CONSTRUCTORS))
def constructor(request: pytest.FixtureRequest) -> Constructor:
    return CONSTRUCTORS[request.param]
```

The test body then never mentions a specific library:

```py
# test_with_total.py
from __future__ import annotations

import narwhals as nw
from narwhals.testing import assert_frame_equal

from my_library import with_total

data = {"price": [1.5, 2.0], "quantity": [2, 3]}


def test_with_total(constructor: Constructor) -> None:
    result = nw.from_native(with_total(constructor(data)))
    expected = nw.from_native(constructor({**data, "total": [3.0, 6.0]}))
    assert_frame_equal(result, expected)
```

Because the constructors return native frames, lazy backends stay lazy until
`assert_frame_equal` collects them, which is what exercises the lazy code paths in
`with_total`.

Narwhals' own version of this lives in
[`tests/conftest.py`](https://github.com/narwhals-dev/narwhals/blob/main/tests/conftest.py).
It adds pandas' nullable and PyArrow dtype backends, Modin, cuDF, Dask, PySpark, SQLFrame
and Ibis, plus a `--constructors` command-line option to select a subset.

## Comparison options

`assert_frame_equal` and `assert_series_equal` share these keyword arguments:

| Argument | Default | Effect |
| --- | --- | --- |
| `check_dtypes` | `True` | Requires data types to match. |
| `check_exact` | `False` | When `False`, float columns are compared within `rel_tol` and `abs_tol`. |
| `rel_tol` | `1e-5` | Relative tolerance, as a fraction of the values in `right`. |
| `abs_tol` | `1e-8` | Absolute tolerance. |
| `categorical_as_str` | `False` | Casts categorical columns to string before comparing, for columns which don't share a string cache. |

`assert_frame_equal` also takes `check_row_order`, `check_column_order` and `backend`.
`assert_series_equal` also takes `check_names` and `check_order`.

Two checks happen before any of that:

- Both arguments must be `narwhals.DataFrame` or `narwhals.LazyFrame` instances. Native
  frames raise a `TypeError`.
- Both must have the same `implementation`. A pandas frame compared against a Polars one
  raises an `AssertionError` reporting an implementation mismatch.

Comparing two `LazyFrame`s calls `collect()` on each, and `backend` chooses which eager
backend to collect into. DuckDB, Ibis, PySpark and SQLFrame guarantee no row order, so for
those `check_row_order` is ignored and both frames are sorted by all their non-nested
columns before comparing.

## Namespaces

`narwhals.testing` exists only in the main `narwhals` namespace. It accepts objects from
`narwhals.stable.v1` and `narwhals.stable.v2` too:

```python exec="yes" source="material-block" result="python" session="how-to-testing"
import narwhals.stable.v2 as nw_v2

left = nw_v2.from_native(pl.DataFrame(data))
right = nw_v2.from_native(pl.DataFrame(data))
assert_frame_equal(left, right)
print("stable.v2 frames match")
```
