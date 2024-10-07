import narwhals as nw
from tests.utils import Constructor
from tests.utils import compare_dicts


def test_alias_rename(constructor_eager: Constructor) -> None:
    data = [1, 2, 3]
    expected = {"bar": data}
    series = nw.from_native(constructor_eager({"foo": data}), eager_only=True)["foo"]
    result = series.alias("bar").to_frame()
    compare_dicts(result, expected)
    result = series.rename("bar").to_frame()
    compare_dicts(result, expected)
