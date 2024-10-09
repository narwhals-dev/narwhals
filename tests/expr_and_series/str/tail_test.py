import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts

data = {"a": ["foo", "bars"]}


def test_str_tail(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    expected = {"a": ["foo", "ars"]}

    result_frame = df.select(nw.col("a").str.tail(3))
    compare_dicts(result_frame, expected)


def test_str_tail_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = {"a": ["foo", "ars"]}

    result_series = df["a"].str.tail(3)
    compare_dicts({"a": result_series}, expected)
