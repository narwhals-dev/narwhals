import pandas as pd
import pytest

import narwhals as nw


def test_narwhalify() -> None:
    @nw.narwhalify
    def func(df: nw.DataFrame) -> nw.DataFrame:
        return df.with_columns(nw.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))
    result = func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))


def test_narwhalify_called() -> None:
    @nw.narwhalify()
    def func(df: nw.DataFrame) -> nw.DataFrame:
        return df.with_columns(nw.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))
    result = func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))


def test_narwhalify_method() -> None:
    class Foo:
        @nw.narwhalify_method
        def func(self, df: nw.DataFrame) -> nw.DataFrame:
            return df.with_columns(nw.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))
    result = Foo().func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))


def test_narwhalify_method_called() -> None:
    class Foo:
        @nw.narwhalify_method(eager_only=True)
        def func(self, df: nw.DataFrame) -> nw.DataFrame:
            return df.with_columns(nw.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))
    result = Foo().func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))


def test_narwhalify_method_invalid() -> None:
    class Foo:
        @nw.narwhalify_method(eager_only=True)
        def func(self) -> nw.DataFrame:
            return self  # type: ignore[return-value]

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(TypeError):
        Foo().func(df)
    with pytest.raises(TypeError):
        Foo().func(df=df)


def test_narwhalify_invalid() -> None:
    @nw.narwhalify(eager_only=True)
    def func() -> nw.DataFrame:
        return None  # type: ignore[return-value]

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(TypeError):
        func(df)
    with pytest.raises(TypeError):
        func(df=df)
