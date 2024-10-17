import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts

data = {"a": ["foo", "foobar", "Café", "345", "東京"]}


def test_str_len_chars(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").str.len_chars())
    expected = {
        "a": [3, 6, 4, 3, 2],
    }
    compare_dicts(result, expected)


def test_str_len_chars_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = {
        "a": [3, 6, 4, 3, 2],
    }
    result = df.select(df["a"].str.len_chars())
    compare_dicts(result, expected)
