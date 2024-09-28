import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts

data = {"a": [1, 4, 2, 5]}


def test_expr_is_in(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").is_in([4, 5]))
    expected = {"a": [False, True, False, True]}

    compare_dicts(result, expected)


def test_ser_is_in(constructor_eager: ConstructorEager) -> None:
    ser = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = {"a": ser.is_in([4, 5])}
    expected = {"a": [False, True, False, True]}

    compare_dicts(result, expected)


def test_is_in_other(constructor: Constructor) -> None:
    df_raw = constructor(data)
    with pytest.raises(
        NotImplementedError,
        match=(
            "Narwhals `is_in` doesn't accept expressions as an argument, as opposed to Polars. You should provide an iterable instead."
        ),
    ):
        nw.from_native(df_raw).with_columns(contains=nw.col("a").is_in("sets"))
