import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts


def test_clip(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3, -4, 5]}))
    result = df.select(
        lower_only=nw.col("a").clip(lower_bound=3),
        upper_only=nw.col("a").clip(upper_bound=4),
        both=nw.col("a").clip(3, 4),
    )
    expected = {
        "lower_only": [3, 3, 3, 3, 5],
        "upper_only": [1, 2, 3, -4, 4],
        "both": [3, 3, 3, 3, 4],
    }
    compare_dicts(result, expected)


def test_clip_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3, -4, 5]}), eager_only=True)
    result = {
        "lower_only": df["a"].clip(lower_bound=3),
        "upper_only": df["a"].clip(upper_bound=4),
        "both": df["a"].clip(3, 4),
    }

    expected = {
        "lower_only": [3, 3, 3, 3, 5],
        "upper_only": [1, 2, 3, -4, 4],
        "both": [3, 3, 3, 3, 4],
    }
    compare_dicts(result, expected)
