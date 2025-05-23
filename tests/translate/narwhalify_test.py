from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrameT

data = {"a": [2, 3, 4]}


def test_narwhalify() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    @nw.narwhalify
    def func(df: nw.DataFrame[IntoDataFrameT]) -> nw.DataFrame[IntoDataFrameT]:
        return df.with_columns(nw.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    class Foo:
        @nw.narwhalify
        def func(
            self, df: nw.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method_called() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    class Foo:
        @nw.narwhalify
        def func(
            self, df: nw.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


@pytest.mark.filterwarnings("ignore:.*distutils Version classes are deprecated")
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


def test_narwhalify_backends_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    @nw.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    func(pd.DataFrame(data), pd.Series(data["a"]))


def test_narwhalify_backends_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    @nw.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    func(pl.DataFrame(data), pl.Series(data["a"]))


def test_narwhalify_backends_cross() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")

    import pandas as pd
    import polars as pl

    @nw.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    with pytest.raises(
        ValueError,
        match="Found multiple backends. Make sure that all dataframe/series inputs come from the same backend.",
    ):
        func(pd.DataFrame(data), pl.DataFrame(data))


def test_narwhalify_backends_cross2() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")

    import pandas as pd
    import polars as pl

    @nw.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    with pytest.raises(
        ValueError,
        match="Found multiple backends. Make sure that all dataframe/series inputs come from the same backend.",
    ):
        func(pl.DataFrame(data), pd.Series(data["a"]))
