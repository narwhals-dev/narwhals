from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": ["-1", "+1", "1", "12", "123", "99999", "+9999", None]}
expected = {"a": ["-01", "+01", "001", "012", "123", "99999", "+9999", None]}


def skip_pandas_pyarrow(constructor: Constructor | ConstructorEager) -> None:
    name: str = constructor.__name__
    if name in {"pandas_pyarrow_constructor", "modin_pyarrow_constructor"}:
        reason = (
            "pandas with pyarrow backend doesn't support str.zfill, see "
            "https://github.com/pandas-dev/pandas/issues/61485"
        )
        raise pytest.skip(reason=reason)

    if "polars" in name:
        reason = "Polars zfill behavior is different from pandas at the moment."
        raise pytest.skip(reason=reason)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="different zfill behavior")
def test_str_zfill(constructor: Constructor) -> None:
    skip_pandas_pyarrow(constructor)

    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.zfill(3))
    assert_equal_data(result, expected)


def test_str_zfill_series(constructor_eager: ConstructorEager) -> None:
    skip_pandas_pyarrow(constructor_eager)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].str.zfill(3))
    assert_equal_data(result, expected)
