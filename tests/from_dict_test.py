from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_from_dict(constructor_eager: Any) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [4, 5, 6]}), eager_only=True
    )
    native_namespace = nw.get_native_namespace(df)
    result = nw.from_dict({"c": [1, 2], "d": [5, 6]}, native_namespace=native_namespace)
    expected = {"c": [1, 2], "d": [5, 6]}
    compare_dicts(result, expected)
    assert isinstance(result, nw.DataFrame)


def test_from_dict_schema(constructor_eager: Any) -> None:
    schema = {"c": nw.Int16(), "d": nw.Float32()}
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [4, 5, 6]}), eager_only=True
    )
    native_namespace = nw.get_native_namespace(df)
    result = nw.from_dict(
        {"c": [1, 2], "d": [5, 6]},
        native_namespace=native_namespace,
        schema=schema,  # type: ignore[arg-type]
    )
    assert result.collect_schema() == schema
