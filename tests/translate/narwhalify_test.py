from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw

data = {"a": [2, 3, 4]}


def test_narwhalify() -> None:
    @nw.narwhalify
    def func(df: nw.DataFrame) -> nw.DataFrame:
        return df.with_columns(nw.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method() -> None:
    class Foo:
        @nw.narwhalify
        def func(self, df: nw.DataFrame, a: int = 1) -> nw.DataFrame:
            return df.with_columns(nw.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method_called() -> None:
    class Foo:
        @nw.narwhalify
        def func(self, df: nw.DataFrame, a: int = 1) -> nw.DataFrame:
            return df.with_columns(nw.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


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


@pytest.mark.parametrize(
    ("arg1", "arg2", "context"),
    [
        (pd.DataFrame(data), pd.Series(data["a"]), does_not_raise()),
        (pl.DataFrame(data), pl.Series(data["a"]), does_not_raise()),
        (
            pd.DataFrame(data),
            pl.DataFrame(data),
            pytest.raises(
                ValueError,
                match="Found multiple backends. Make sure that all dataframe/series inputs come from the same backend.",
            ),
        ),
        (
            pl.DataFrame(data),
            pd.Series(data["a"]),
            pytest.raises(
                ValueError,
                match="Found multiple backends. Make sure that all dataframe/series inputs come from the same backend.",
            ),
        ),
    ],
)
def test_narwhalify_backends(arg1: Any, arg2: Any, context: Any) -> None:
    @nw.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    with context:
        func(arg1, arg2)
