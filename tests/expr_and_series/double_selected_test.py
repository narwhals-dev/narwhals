import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_double_selected(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))

    result = df.select(nw.col("a", "b") * 2)
    expected = {"a": [2, 6, 4], "b": [8, 8, 12]}
    assert_equal_data(result, expected)

    result = df.select("z", nw.col("a", "b") * 2)
    expected = {"z": [7, 8, 9], "a": [2, 6, 4], "b": [8, 8, 12]}
    assert_equal_data(result, expected)

    result = df.select("a").select(nw.col("a") + nw.all())
    expected = {"a": [2, 6, 4]}
    assert_equal_data(result, expected)
