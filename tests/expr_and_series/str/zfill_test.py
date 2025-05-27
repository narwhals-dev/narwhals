from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["-1", "+1", "1", "12", "123", "99999", "+9999", None]}
polars_expected = {"a": ["-01", "0+1", "001", "012", "123", "99999", "+9999", None]}
pandas_and_dask_expected = {
    "a": ["-01", "+01", "001", "012", "123", "99999", "+9999", None]
}


def skip_pandas_pyarrow(constructor: Constructor | ConstructorEager) -> None:
    name: str = constructor.__name__
    if name in {"pandas_pyarrow_constructor", "modin_pyarrow_constructor"}:
        reason = (
            "pandas with pyarrow backend doesn't support str.zfill, see "
            "https://github.com/pandas-dev/pandas/issues/61485"
        )
        raise pytest.skip(reason=reason)


def get_expected(constructor_name: str) -> dict[str, list[str | None]]:
    if "pandas" in constructor_name or "dask" in constructor_name:
        return pandas_and_dask_expected

    return polars_expected


def test_str_zfill(constructor: Constructor) -> None:
    skip_pandas_pyarrow(constructor)

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.zfill(3))
    expected = get_expected(constructor.__name__)
    assert_equal_data(result, expected)


def test_str_zfill_series(constructor_eager: ConstructorEager) -> None:
    skip_pandas_pyarrow(constructor_eager)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].str.zfill(3))
    expected = get_expected(constructor_eager.__name__)
    assert_equal_data(result, expected)
