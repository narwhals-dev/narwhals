from __future__ import annotations

from typing import Any

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


def test_narwhalify_method() -> None:
    class Foo:
        @nw.narwhalify
        def func(self, df: nw.DataFrame, a: int = 1) -> nw.DataFrame:
            return df.with_columns(nw.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))


def test_narwhalify_method_called() -> None:
    class Foo:
        @nw.narwhalify
        def func(self, df: nw.DataFrame, a: int = 1) -> nw.DataFrame:
            return df.with_columns(nw.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))
    result = Foo().func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"a": [2, 3, 4]}))


def test_narwhalify_method_invalid() -> None:
    class Foo:
        @nw.narwhalify(strict=True, eager_only=True)
        def func(self) -> nw.DataFrame:  # pragma: no cover
            return self  # type: ignore[return-value]

        @nw.narwhalify(strict=True, eager_only=True)
        def fun2(self, df: Any) -> nw.DataFrame:  # pragma: no cover
            return df  # type: ignore[no-any-return]

    with pytest.raises(TypeError):
        Foo().func()


def test_narwhalify_invalid() -> None:
    @nw.narwhalify(strict=True)
    def func() -> nw.DataFrame:  # pragma: no cover
        return None  # type: ignore[return-value]

    with pytest.raises(TypeError):
        func()
