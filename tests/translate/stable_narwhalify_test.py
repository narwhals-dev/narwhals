from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytest

import narwhals.stable.v1 as nw_v1

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrameT

data = {"a": [2, 3, 4]}


def test_narwhalify() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    @nw_v1.narwhalify
    def func(df: nw_v1.DataFrame[IntoDataFrameT]) -> nw_v1.DataFrame[IntoDataFrameT]:
        return df.with_columns(nw_v1.all() + 1)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    class Foo:
        @nw_v1.narwhalify
        def func(
            self, df: nw_v1.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw_v1.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw_v1.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method_called() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    class Foo:
        @nw_v1.narwhalify
        def func(
            self, df: nw_v1.DataFrame[IntoDataFrameT], a: int = 1
        ) -> nw_v1.DataFrame[IntoDataFrameT]:
            return df.with_columns(nw_v1.all() + a)

    df = pd.DataFrame({"a": [1, 2, 3]})
    result = Foo().func(df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))
    result = Foo().func(a=1, df=df)
    pd.testing.assert_frame_equal(result, pd.DataFrame(data))


def test_narwhalify_method_invalid() -> None:
    class Foo:
        @nw_v1.narwhalify(strict=True, eager_only=True)
        def func(self) -> Foo:  # pragma: no cover
            return self

        @nw_v1.narwhalify(strict=True, eager_only=True)
        def fun2(self, df: Any) -> Any:  # pragma: no cover
            return df

    with pytest.raises(TypeError):
        Foo().func()


def test_narwhalify_invalid() -> None:
    @nw_v1.narwhalify(strict=True)
    def func() -> None:  # pragma: no cover
        return None

    with pytest.raises(TypeError):
        func()


def test_narwhalify_backends_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    @nw_v1.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    func(pd.DataFrame(data), pd.Series(data["a"]))


def test_narwhalify_backends_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    @nw_v1.narwhalify
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

    @nw_v1.narwhalify
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

    @nw_v1.narwhalify
    def func(
        arg1: Any, arg2: Any, extra: int = 1
    ) -> tuple[Any, Any, int]:  # pragma: no cover
        return arg1, arg2, extra

    with pytest.raises(
        ValueError,
        match="Found multiple backends. Make sure that all dataframe/series inputs come from the same backend.",
    ):
        func(pl.DataFrame(data), pd.Series(data["a"]))
