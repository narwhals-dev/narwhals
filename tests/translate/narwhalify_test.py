from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING
from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.typing import IntoDataFrameT

data = {"a": [2, 3, 4]}


def test_narwhalify() -> None:
    @nw.narwhalify
    def func(df: nw.DataFrame[IntoDataFrameT]) -> nw.DataFrame[IntoDataFrameT]:
        return df.with_columns(nw.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method() -> None:
    class Foo:
        @nw.narwhalify
        def func(
            self: Self, df: nw.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method_called() -> None:
    class Foo:
        @nw.narwhalify
        def func(
            self: Self, df: nw.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method_invalid() -> None:
    with pytest.deprecated_call(match="please use `pass_through` instead"):

        class Foo:
            @nw.narwhalify(strict=True, eager_only=True)
            def func(self) -> Foo:  # pragma: no cover
                return self

            @nw.narwhalify(strict=True, eager_only=True)
            def fun2(self, df: Any) -> Any:  # pragma: no cover
                return df

        with pytest.raises(TypeError):
            Foo().func()


def test_narwhalify_invalid() -> None:
    @nw.narwhalify(pass_through=False)
    def func() -> None:  # pragma: no cover
        return None

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
