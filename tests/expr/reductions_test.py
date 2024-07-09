def test_expr_min_max(constructor_with_pyarrow: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor_with_pyarrow(data), eager_only=True)
    result_min = nw.to_native(df.select(nw.min("a", "b", "z")))
    result_max = nw.to_native(df.select(nw.max("a", "b", "z")))
    expected_min = {"a": [1], "b": [4], "z": [7]}
    expected_max = {"a": [3], "b": [6], "z": [9]}
    compare_dicts(result_min, expected_min)
    compare_dicts(result_max, expected_max)
