from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Sequence
from typing import TypeVar
from typing import overload

from narwhals.dependencies import get_polars
from narwhals.dependencies import is_numpy_array
from narwhals.schema import Schema
from narwhals.utils import flatten
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from io import BytesIO
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals.group_by import GroupBy
    from narwhals.group_by import LazyGroupBy
    from narwhals.series import Series
    from narwhals.typing import IntoDataFrame
    from narwhals.typing import IntoExpr

FrameT = TypeVar("FrameT", bound="IntoDataFrame")


class BaseFrame(Generic[FrameT]):
    _compliant_frame: Any
    _level: Literal["full", "interchange"]

    def __len__(self) -> Any:
        return self._compliant_frame.__len__()

    def __native_namespace__(self) -> Any:
        return self._compliant_frame.__native_namespace__()

    def __narwhals_namespace__(self) -> Any:
        return self._compliant_frame.__narwhals_namespace__()

    def _from_compliant_dataframe(self, df: Any) -> Self:
        # construct, preserving properties
        return self.__class__(  # type: ignore[call-arg]
            df,
            level=self._level,
        )

    def _flatten_and_extract(self, *args: Any, **kwargs: Any) -> Any:
        """Process `args` and `kwargs`, extracting underlying objects as we go."""
        args = [self._extract_compliant(v) for v in flatten(args)]  # type: ignore[assignment]
        kwargs = {k: self._extract_compliant(v) for k, v in kwargs.items()}
        return args, kwargs

    def _extract_compliant(self, arg: Any) -> Any:
        from narwhals.expr import Expr
        from narwhals.series import Series

        if isinstance(arg, BaseFrame):
            return arg._compliant_frame
        if isinstance(arg, Series):
            return arg._compliant_series
        if isinstance(arg, Expr):
            return arg._call(self.__narwhals_namespace__())
        if get_polars() is not None and "polars" in str(type(arg)):
            msg = (
                f"Expected Narwhals object, got: {type(arg)}.\n\n"
                "Perhaps you:\n"
                "- Forgot a `nw.from_native` somewhere?\n"
                "- Used `pl.col` instead of `nw.col`?"
            )
            raise TypeError(msg)
        return arg

    @property
    def schema(self) -> Schema:
        return Schema(self._compliant_frame.schema.items())

    def collect_schema(self) -> Schema:
        native_schema = dict(self._compliant_frame.collect_schema())

        return Schema(native_schema)

    def pipe(self, function: Callable[[Any], Self], *args: Any, **kwargs: Any) -> Self:
        return function(self, *args, **kwargs)

    def with_row_index(self, name: str = "index") -> Self:
        return self._from_compliant_dataframe(
            self._compliant_frame.with_row_index(name),
        )

    def drop_nulls(self: Self, subset: str | list[str] | None = None) -> Self:
        return self._from_compliant_dataframe(
            self._compliant_frame.drop_nulls(subset=subset),
        )

    @property
    def columns(self) -> list[str]:
        return self._compliant_frame.columns  # type: ignore[no-any-return]

    def lazy(self) -> LazyFrame[Any]:
        return LazyFrame(
            self._compliant_frame.lazy(),
            level=self._level,
        )

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        exprs, named_exprs = self._flatten_and_extract(*exprs, **named_exprs)
        return self._from_compliant_dataframe(
            self._compliant_frame.with_columns(*exprs, **named_exprs),
        )

    def select(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        exprs, named_exprs = self._flatten_and_extract(*exprs, **named_exprs)
        return self._from_compliant_dataframe(
            self._compliant_frame.select(*exprs, **named_exprs),
        )

    def rename(self, mapping: dict[str, str]) -> Self:
        return self._from_compliant_dataframe(self._compliant_frame.rename(mapping))

    def head(self, n: int) -> Self:
        return self._from_compliant_dataframe(self._compliant_frame.head(n))

    def tail(self, n: int) -> Self:
        return self._from_compliant_dataframe(self._compliant_frame.tail(n))

    def drop(self, *columns: Iterable[str], strict: bool) -> Self:
        return self._from_compliant_dataframe(
            self._compliant_frame.drop(columns, strict=strict)
        )

    def unique(
        self,
        subset: str | list[str] | None = None,
        *,
        keep: Literal["any", "first", "last", "none"] = "any",
        maintain_order: bool = False,
    ) -> Self:
        return self._from_compliant_dataframe(
            self._compliant_frame.unique(
                subset=subset, keep=keep, maintain_order=maintain_order
            )
        )

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        predicates, _ = self._flatten_and_extract(*predicates)
        return self._from_compliant_dataframe(
            self._compliant_frame.filter(*predicates),
        )

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        return self._from_compliant_dataframe(
            self._compliant_frame.sort(by, *more_by, descending=descending)
        )

    def join(
        self,
        other: Self,
        *,
        how: Literal["inner", "left", "cross", "semi", "anti"] = "inner",
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
    ) -> Self:
        _supported_joins = ("inner", "left", "cross", "anti", "semi")

        if how not in _supported_joins:
            msg = f"Only the following join stragies are supported: {_supported_joins}; found '{how}'."
            raise NotImplementedError(msg)

        if how == "cross" and (left_on or right_on):
            msg = "Can not pass left_on, right_on for cross join"
            raise ValueError(msg)

        return self._from_compliant_dataframe(
            self._compliant_frame.join(
                self._extract_compliant(other),
                how=how,
                left_on=left_on,
                right_on=right_on,
            )
        )

    def clone(self) -> Self:
        return self._from_compliant_dataframe(self._compliant_frame.clone())

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        return self._from_compliant_dataframe(
            self._compliant_frame.gather_every(n=n, offset=offset)
        )


class DataFrame(BaseFrame[FrameT]):
    """
    Narwhals DataFrame, backed by a native dataframe.

    The native dataframe might be pandas.DataFrame, polars.DataFrame, ...

    This class is not meant to be instantiated directly - instead, use
    `narwhals.from_native`.
    """

    def __init__(
        self,
        df: Any,
        *,
        level: Literal["full", "interchange"],
    ) -> None:
        self._level: Literal["full", "interchange"] = level
        if hasattr(df, "__narwhals_dataframe__"):
            self._compliant_frame: Any = df.__narwhals_dataframe__()
        else:  # pragma: no cover
            msg = f"Expected an object which implements `__narwhals_dataframe__`, got: {type(df)}"
            raise AssertionError(msg)

    def __array__(self) -> np.ndarray:
        return self._compliant_frame.to_numpy()

    def __repr__(self) -> str:  # pragma: no cover
        header = " Narwhals DataFrame                            "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Use `narwhals.to_native` to see native output |\n"
            + "└"
            + "─" * length
            + "┘"
        )

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        """
        Export a DataFrame via the Arrow PyCapsule Interface.

        - if the underlying dataframe implements the interface, it'll return that
        - else, it'll call `to_arrow` and then defer to PyArrow's implementation

        See [PyCapsule Interface](https://arrow.apache.org/docs/dev/format/CDataInterface/PyCapsuleInterface.html)
        for more.
        """
        native_frame = self._compliant_frame._native_frame
        if hasattr(native_frame, "__arrow_c_stream__"):
            return native_frame.__arrow_c_stream__(requested_schema=requested_schema)
        try:
            import pyarrow as pa  # ignore-banned-import
        except ModuleNotFoundError as exc:  # pragma: no cover
            msg = f"PyArrow>=14.0.0 is required for `DataFrame.__arrow_c_stream__` for object of type {type(native_frame)}"
            raise ModuleNotFoundError(msg) from exc
        if parse_version(pa.__version__) < (14, 0):  # pragma: no cover
            msg = f"PyArrow>=14.0.0 is required for `DataFrame.__arrow_c_stream__` for object of type {type(native_frame)}"
            raise ModuleNotFoundError(msg) from None
        pa_table = self.to_arrow()
        return pa_table.__arrow_c_stream__(requested_schema=requested_schema)

    def lazy(self) -> LazyFrame[Any]:
        """
        Lazify the DataFrame (if possible).

        If a library does not support lazy execution, then this is a no-op.

        Examples:
            Construct pandas and Polars DataFrames:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.lazy()

            Note that then, pandas dataframe stay eager, but Polars DataFrame becomes a Polars LazyFrame:

            >>> func(df_pd)
               foo  bar ham
            0    1  6.0   a
            1    2  7.0   b
            2    3  8.0   c
            >>> func(df_pl)
            <LazyFrame ...>
        """
        return super().lazy()

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert this DataFrame to a pandas DataFrame.

        Examples:
            Construct pandas and Polars DataFrames:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.to_pandas()

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo  bar ham
            0    1  6.0   a
            1    2  7.0   b
            2    3  8.0   c
            >>> func(df_pl)
               foo  bar ham
            0    1  6.0   a
            1    2  7.0   b
            2    3  8.0   c
        """
        return self._compliant_frame.to_pandas()

    def write_parquet(self, file: str | Path | BytesIO) -> Any:
        """
        Write dataframe to parquet file.

        Examples:
            Construct pandas and Polars DataFrames:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> def func(df):
            ...     df = nw.from_native(df)
            ...     df.write_parquet("foo.parquet")

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest:+SKIP
            >>> func(df_pl)  # doctest:+SKIP
        """
        self._compliant_frame.write_parquet(file)

    def to_numpy(self) -> np.ndarray:
        """
        Convert this DataFrame to a NumPy ndarray.

        Examples:
            Construct pandas and polars DataFrames:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6.5, 7.0, 8.5], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.to_numpy()

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
            array([[1, 6.5, 'a'],
                   [2, 7.0, 'b'],
                   [3, 8.5, 'c']], dtype=object)
            >>> func(df_pl)
            array([[1, 6.5, 'a'],
                   [2, 7.0, 'b'],
                   [3, 8.5, 'c']], dtype=object)
        """
        return self._compliant_frame.to_numpy()

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the DataFrame.

        Examples:
            Construct pandas and polars DataFrames:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3, 4, 5]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.shape

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
            (5, 1)
            >>> func(df_pl)
            (5, 1)
        """
        return self._compliant_frame.shape  # type: ignore[no-any-return]

    def get_column(self, name: str) -> Series:
        """
        Get a single column by name.

        Notes:
            Although `name` is typed as `str`, pandas does allow non-string column
            names, and they will work when passed to this function if the
            `narwhals.DataFrame` is backed by a pandas dataframe with non-string
            columns. This function can only be used to extract a column by name, so
            there is no risk of ambiguity.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {"a": [1, 2], "b": [3, 4]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a library agnostic function:

            >>> @nw.narwhalify(eager_only=True)
            ... def func(df):
            ...     name = df.columns[0]
            ...     return df.get_column(name)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
            0    1
            1    2
            Name: a, dtype: int64
            >>> func(df_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (2,)
            Series: 'a' [i64]
            [
                1
                2
            ]
        """
        from narwhals.series import Series

        return Series(
            self._compliant_frame.get_column(name),
            level=self._level,
        )

    @overload
    def __getitem__(self, item: tuple[Sequence[int], Sequence[int]]) -> Self: ...
    @overload
    def __getitem__(self, item: tuple[Sequence[int], str]) -> Series: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(self, item: tuple[Sequence[int], Sequence[str]]) -> Self: ...
    @overload
    def __getitem__(self, item: tuple[Sequence[int], int]) -> Series: ...  # type: ignore[overload-overlap]

    @overload
    def __getitem__(self, item: Sequence[int]) -> Self: ...

    @overload
    def __getitem__(self, item: str) -> Series: ...

    @overload
    def __getitem__(self, item: slice) -> Self: ...

    def __getitem__(
        self,
        item: str
        | slice
        | Sequence[int]
        | tuple[Sequence[int], str | int]
        | tuple[Sequence[int], Sequence[int] | Sequence[str]],
    ) -> Series | Self:
        """
        Extract column or slice of DataFrame.

        Arguments:
            item: how to slice dataframe:

                - str: extract column
                - slice or Sequence of integers: slice rows from dataframe.
                - tuple of Sequence of integers and str or int: slice rows and extract column at the same time.
                - tuple of Sequence of integers and Sequence of integers: slice rows and extract columns at the same time.
        Notes:
            Integers are always interpreted as positions, and strings always as column names.

            In contrast with Polars, pandas allows non-string column names.
            If you don't know whether the column name you're trying to extract
            is definitely a string (e.g. `df[df.columns[0]]`) then you should
            use `DataFrame.get_column` instead.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {"a": [1, 2], "b": [3, 4]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a library agnostic function:

            >>> @nw.narwhalify(eager_only=True)
            ... def func(df):
            ...     return df["a"]

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
            0    1
            1    2
            Name: a, dtype: int64
            >>> func(df_pl)  # doctest:+NORMALIZE_WHITESPACE
            shape: (2,)
            Series: 'a' [i64]
            [
                1
                2
            ]
        """
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and (isinstance(item[0], (str, int)))
        ):
            msg = (
                f"Expected str or slice, got: {type(item)}.\n\n"
                "Hint: if you were trying to get a single element out of a "
                "dataframe, use `DataFrame.item`."
            )
            raise TypeError(msg)
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[1], (list, tuple))
        ):
            return self._from_compliant_dataframe(self._compliant_frame[item])
        if isinstance(item, str) or (isinstance(item, tuple) and len(item) == 2):
            from narwhals.series import Series

            return Series(
                self._compliant_frame[item],
                level=self._level,
            )

        elif isinstance(item, (Sequence, slice)) or (
            is_numpy_array(item) and item.ndim == 1
        ):
            return self._from_compliant_dataframe(self._compliant_frame[item])

        else:
            msg = f"Expected str or slice, got: {type(item)}"
            raise TypeError(msg)

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    @overload
    def to_dict(self, *, as_series: Literal[True] = ...) -> dict[str, Series]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(self, *, as_series: bool) -> dict[str, Series] | dict[str, list[Any]]: ...
    def to_dict(
        self, *, as_series: bool = True
    ) -> dict[str, Series] | dict[str, list[Any]]:
        """
        Convert DataFrame to a dictionary mapping column name to values.

        Arguments:
            as_series: If set to true ``True``, then the values are Narwhals Series,
                        otherwise the values are Any.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {
            ...     "A": [1, 2, 3, 4, 5],
            ...     "fruits": ["banana", "banana", "apple", "apple", "banana"],
            ...     "B": [5, 4, 3, 2, 1],
            ...     "animals": ["beetle", "fly", "beetle", "beetle", "beetle"],
            ...     "optional": [28, 300, None, 2, -30],
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.to_dict(as_series=False)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
            {'A': [1, 2, 3, 4, 5], 'fruits': ['banana', 'banana', 'apple', 'apple', 'banana'], 'B': [5, 4, 3, 2, 1], 'animals': ['beetle', 'fly', 'beetle', 'beetle', 'beetle'], 'optional': [28.0, 300.0, nan, 2.0, -30.0]}
            >>> func(df_pl)
            {'A': [1, 2, 3, 4, 5], 'fruits': ['banana', 'banana', 'apple', 'apple', 'banana'], 'B': [5, 4, 3, 2, 1], 'animals': ['beetle', 'fly', 'beetle', 'beetle', 'beetle'], 'optional': [28, 300, None, 2, -30]}
        """
        from narwhals.series import Series

        if as_series:
            return {
                key: Series(
                    value,
                    level=self._level,
                )
                for key, value in self._compliant_frame.to_dict(
                    as_series=as_series
                ).items()
            }
        return self._compliant_frame.to_dict(as_series=as_series)  # type: ignore[no-any-return]

    # inherited
    def pipe(self, function: Callable[[Any], Self], *args: Any, **kwargs: Any) -> Self:
        """
        Pipe function call.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {"a": [1, 2, 3], "ba": [4, 5, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.pipe(
            ...         lambda _df: _df.select([x for x in _df.columns if len(x) == 1])
            ...     )

            We can then pass either pandas or Polars:

            >>> func(df_pd)
               a
            0  1
            1  2
            2  3
            >>> func(df_pl)
            shape: (3, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            │ 2   │
            │ 3   │
            └─────┘
        """
        return super().pipe(function, *args, **kwargs)

    def drop_nulls(self: Self, subset: str | list[str] | None = None) -> Self:
        """
        Drop null values.

        Arguments:
            subset: Column name(s) for which null values are considered. If set to None
                (default), use all columns.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {"a": [1.0, 2.0, None], "ba": [1.0, None, 2.0]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.drop_nulls()

            We can then pass either pandas or Polars:

            >>> func(df_pd)
                 a   ba
            0  1.0  1.0
            >>> func(df_pl)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ ba  │
            │ --- ┆ --- │
            │ f64 ┆ f64 │
            ╞═════╪═════╡
            │ 1.0 ┆ 1.0 │
            └─────┴─────┘
        """
        return super().drop_nulls(subset=subset)

    def with_row_index(self, name: str = "index") -> Self:
        """
        Insert column which enumerates rows.

        Examples:
            Construct pandas as polars DataFrames:

            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_row_index()

            We can then pass either pandas or Polars:

            >>> func(df_pd)
               index  a  b
            0      0  1  4
            1      1  2  5
            2      2  3  6
            >>> func(df_pl)
            shape: (3, 3)
            ┌───────┬─────┬─────┐
            │ index ┆ a   ┆ b   │
            │ ---   ┆ --- ┆ --- │
            │ u32   ┆ i64 ┆ i64 │
            ╞═══════╪═════╪═════╡
            │ 0     ┆ 1   ┆ 4   │
            │ 1     ┆ 2   ┆ 5   │
            │ 2     ┆ 3   ┆ 6   │
            └───────┴─────┴─────┘
        """
        return super().with_row_index(name)

    @property
    def schema(self) -> Schema:
        r"""
        Get an ordered mapping of column names to their data type.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6.0, 7.0, 8.0],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.schema

            You can pass either pandas or Polars to `func`:

            >>> df_pd_schema = func(df_pd)
            >>> df_pd_schema  # doctest:+SKIP
            Schema({'foo': Int64, 'bar': Float64, 'ham', String})

            >>> df_pl_schema = func(df_pl)
            >>> df_pl_schema  # doctest:+SKIP
            Schema({'foo': Int64, 'bar': Float64, 'ham', String})
        """
        return super().schema

    def collect_schema(self: Self) -> Schema:
        r"""
        Get an ordered mapping of column names to their data type.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6.0, 7.0, 8.0],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.collect_schema()

            You can pass either pandas or Polars to `func`:

            >>> df_pd_schema = func(df_pd)
            >>> df_pd_schema  # doctest:+SKIP
            Schema({'foo': Int64, 'bar': Float64, 'ham', String})

            >>> df_pl_schema = func(df_pl)
            >>> df_pl_schema  # doctest:+SKIP
            Schema({'foo': Int64, 'bar': Float64, 'ham', String})
        """
        return super().collect_schema()

    @property
    def columns(self) -> list[str]:
        """
        Get column names.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.columns

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
            ['foo', 'bar', 'ham']
            >>> func(df_pl)
            ['foo', 'bar', 'ham']
        """
        return super().columns

    @overload
    def rows(
        self,
        *,
        named: Literal[False],
    ) -> list[tuple[Any, ...]]: ...

    @overload
    def rows(
        self,
        *,
        named: Literal[True],
    ) -> list[dict[str, Any]]: ...

    @overload
    def rows(
        self,
        *,
        named: bool,
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]: ...

    def rows(
        self,
        *,
        named: bool = False,
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """
        Returns all data in the DataFrame as a list of rows of python-native values.

        Arguments:
            named: By default, each row is returned as a tuple of values given
                in the same order as the frame columns. Setting named=True will
                return rows of dictionaries instead.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df, *, named):
            ...     return df.rows(named=named)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd, named=False)
            [(1, 6.0, 'a'), (2, 7.0, 'b'), (3, 8.0, 'c')]
            >>> func(df_pd, named=True)
            [{'foo': 1, 'bar': 6.0, 'ham': 'a'}, {'foo': 2, 'bar': 7.0, 'ham': 'b'}, {'foo': 3, 'bar': 8.0, 'ham': 'c'}]
            >>> func(df_pl, named=False)
            [(1, 6.0, 'a'), (2, 7.0, 'b'), (3, 8.0, 'c')]
            >>> func(df_pl, named=True)
            [{'foo': 1, 'bar': 6.0, 'ham': 'a'}, {'foo': 2, 'bar': 7.0, 'ham': 'b'}, {'foo': 3, 'bar': 8.0, 'ham': 'c'}]
        """
        return self._compliant_frame.rows(named=named)  # type: ignore[no-any-return]

    @overload
    def iter_rows(
        self, *, named: Literal[False], buffer_size: int = ...
    ) -> Iterator[tuple[Any, ...]]: ...

    @overload
    def iter_rows(
        self, *, named: Literal[True], buffer_size: int = ...
    ) -> Iterator[dict[str, Any]]: ...

    @overload
    def iter_rows(
        self, *, named: bool, buffer_size: int = ...
    ) -> Iterator[tuple[Any, ...]] | Iterator[dict[str, Any]]: ...

    def iter_rows(
        self, *, named: bool = False, buffer_size: int = 512
    ) -> Iterator[tuple[Any, ...]] | Iterator[dict[str, Any]]:
        """
        Returns an iterator over the DataFrame of rows of python-native values.

        Arguments:
            named: By default, each row is returned as a tuple of values given
                in the same order as the frame columns. Setting named=True will
                return rows of dictionaries instead.
            buffer_size: Determines the number of rows that are buffered
                internally while iterating over the data.
                See https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.iter_rows.html

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df, *, named):
            ...     return df.iter_rows(named=named)

            We can then pass either pandas or Polars to `func`:

            >>> [row for row in func(df_pd, named=False)]
            [(1, 6.0, 'a'), (2, 7.0, 'b'), (3, 8.0, 'c')]
            >>> [row for row in func(df_pd, named=True)]
            [{'foo': 1, 'bar': 6.0, 'ham': 'a'}, {'foo': 2, 'bar': 7.0, 'ham': 'b'}, {'foo': 3, 'bar': 8.0, 'ham': 'c'}]
            >>> [row for row in func(df_pl, named=False)]
            [(1, 6.0, 'a'), (2, 7.0, 'b'), (3, 8.0, 'c')]
            >>> [row for row in func(df_pl, named=True)]
            [{'foo': 1, 'bar': 6.0, 'ham': 'a'}, {'foo': 2, 'bar': 7.0, 'ham': 'b'}, {'foo': 3, 'bar': 8.0, 'ham': 'c'}]
        """
        return self._compliant_frame.iter_rows(named=named, buffer_size=buffer_size)  # type: ignore[no-any-return]

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        r"""
        Add columns to this DataFrame.

        Added columns will replace existing columns with the same name.

        Arguments:
            *exprs: Column(s) to add, specified as positional arguments.
                     Accepts expression input. Strings are parsed as column names, other
                     non-expression inputs are parsed as literals.

            **named_exprs: Additional columns to add, specified as keyword arguments.
                            The columns will be renamed to the keyword used.

        Returns:
            DataFrame: A new DataFrame with the columns added.

        Note:
            Creating a new DataFrame using this method does not create a new copy of
            existing data.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {
            ...     "a": [1, 2, 3, 4],
            ...     "b": [0.5, 4, 10, 13],
            ...     "c": [True, True, False, True],
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            Let's define a dataframe-agnostic function in which we pass an expression
            to add it as a new column:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns((nw.col("a") * 2).alias("a*2"))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a     b      c  a*2
            0  1   0.5   True    2
            1  2   4.0   True    4
            2  3  10.0  False    6
            3  4  13.0   True    8
            >>> func(df_pl)
            shape: (4, 4)
            ┌─────┬──────┬───────┬─────┐
            │ a   ┆ b    ┆ c     ┆ a*2 │
            │ --- ┆ ---  ┆ ---   ┆ --- │
            │ i64 ┆ f64  ┆ bool  ┆ i64 │
            ╞═════╪══════╪═══════╪═════╡
            │ 1   ┆ 0.5  ┆ true  ┆ 2   │
            │ 2   ┆ 4.0  ┆ true  ┆ 4   │
            │ 3   ┆ 10.0 ┆ false ┆ 6   │
            │ 4   ┆ 13.0 ┆ true  ┆ 8   │
            └─────┴──────┴───────┴─────┘
        """
        return super().with_columns(*exprs, **named_exprs)

    def select(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        r"""
        Select columns from this DataFrame.

        Arguments:
            *exprs: Column(s) to select, specified as positional arguments.
                     Accepts expression input. Strings are parsed as column names,
                     other non-expression inputs are parsed as literals.

            **named_exprs: Additional columns to select, specified as keyword arguments.
                            The columns will be renamed to the keyword used.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6, 7, 8],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            Let's define a dataframe-agnostic function in which we pass the name of a
            column to select that column.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select("foo")

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo
            0    1
            1    2
            2    3
            >>> func(df_pl)
            shape: (3, 1)
            ┌─────┐
            │ foo │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            │ 2   │
            │ 3   │
            └─────┘

            Multiple columns can be selected by passing a list of column names.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(["foo", "bar"])
            >>> func(df_pd)
               foo  bar
            0    1    6
            1    2    7
            2    3    8
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 6   │
            │ 2   ┆ 7   │
            │ 3   ┆ 8   │
            └─────┴─────┘

            Multiple columns can also be selected using positional arguments instead of a
            list. Expressions are also accepted.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.col("foo"), nw.col("bar") + 1)
            >>> func(df_pd)
               foo  bar
            0    1    7
            1    2    8
            2    3    9
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 7   │
            │ 2   ┆ 8   │
            │ 3   ┆ 9   │
            └─────┴─────┘

            Use keyword arguments to easily name your expression inputs.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(threshold=nw.col("foo") * 2)
            >>> func(df_pd)
               threshold
            0          2
            1          4
            2          6
            >>> func(df_pl)
            shape: (3, 1)
            ┌───────────┐
            │ threshold │
            │ ---       │
            │ i64       │
            ╞═══════════╡
            │ 2         │
            │ 4         │
            │ 6         │
            └───────────┘
        """
        return super().select(*exprs, **named_exprs)

    def rename(self, mapping: dict[str, str]) -> Self:
        """
        Rename column names.

        Arguments:
            mapping: Key value pairs that map from old name to new name.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.rename({"foo": "apple"})

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               apple  bar ham
            0      1    6   a
            1      2    7   b
            2      3    8   c
            >>> func(df_pl)
            shape: (3, 3)
            ┌───────┬─────┬─────┐
            │ apple ┆ bar ┆ ham │
            │ ---   ┆ --- ┆ --- │
            │ i64   ┆ i64 ┆ str │
            ╞═══════╪═════╪═════╡
            │ 1     ┆ 6   ┆ a   │
            │ 2     ┆ 7   ┆ b   │
            │ 3     ┆ 8   ┆ c   │
            └───────┴─────┴─────┘
        """
        return super().rename(mapping)

    def head(self, n: int = 5) -> Self:
        """
        Get the first `n` rows.

        Arguments:
            n: Number of rows to return. If a negative value is passed, return all rows
                except the last `abs(n)`.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {
            ...     "foo": [1, 2, 3, 4, 5],
            ...     "bar": [6, 7, 8, 9, 10],
            ...     "ham": ["a", "b", "c", "d", "e"],
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            Let's define a dataframe-agnostic function that gets the first 3 rows.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.head(3)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo  bar ham
            0    1    6   a
            1    2    7   b
            2    3    8   c
            >>> func(df_pl)
            shape: (3, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            │ 2   ┆ 7   ┆ b   │
            │ 3   ┆ 8   ┆ c   │
            └─────┴─────┴─────┘
        """
        return super().head(n)

    def tail(self, n: int = 5) -> Self:
        """
        Get the last `n` rows.

        Arguments:
            n: Number of rows to return. If a negative value is passed, return all rows
                except the first `abs(n)`.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {
            ...     "foo": [1, 2, 3, 4, 5],
            ...     "bar": [6, 7, 8, 9, 10],
            ...     "ham": ["a", "b", "c", "d", "e"],
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            Let's define a dataframe-agnostic function that gets the last 3 rows.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.tail(3)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo  bar ham
            2    3    8   c
            3    4    9   d
            4    5   10   e
            >>> func(df_pl)
            shape: (3, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 3   ┆ 8   ┆ c   │
            │ 4   ┆ 9   ┆ d   │
            │ 5   ┆ 10  ┆ e   │
            └─────┴─────┴─────┘
        """
        return super().tail(n)

    def drop(self, *columns: str | Iterable[str], strict: bool = True) -> Self:
        """
        Remove columns from the dataframe.

        Arguments:
            *columns: Names of the columns that should be removed from the dataframe.
            strict: Validate that all column names exist in the schema and throw an
                exception if a column name does not exist in the schema.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.drop("ham")

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo  bar
            0    1  6.0
            1    2  7.0
            2    3  8.0
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ i64 ┆ f64 │
            ╞═════╪═════╡
            │ 1   ┆ 6.0 │
            │ 2   ┆ 7.0 │
            │ 3   ┆ 8.0 │
            └─────┴─────┘

            Use positional arguments to drop multiple columns.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.drop("foo", "ham")

            >>> func(df_pd)
               bar
            0  6.0
            1  7.0
            2  8.0
            >>> func(df_pl)
            shape: (3, 1)
            ┌─────┐
            │ bar │
            │ --- │
            │ f64 │
            ╞═════╡
            │ 6.0 │
            │ 7.0 │
            │ 8.0 │
            └─────┘
        """
        return super().drop(*flatten(columns), strict=strict)

    def unique(
        self,
        subset: str | list[str] | None = None,
        *,
        keep: Literal["any", "first", "last", "none"] = "any",
        maintain_order: bool = False,
    ) -> Self:
        """
        Drop duplicate rows from this dataframe.

        Arguments:
            subset: Column name(s) to consider when identifying duplicate rows.
            keep: {'first', 'last', 'any', 'none'}
                Which of the duplicate rows to keep.

                * 'any': Does not give any guarantee of which row is kept.
                        This allows more optimizations.
                * 'none': Don't keep duplicate rows.
                * 'first': Keep first unique row.
                * 'last': Keep last unique row.
            maintain_order: Keep the same order as the original DataFrame. This is more
                expensive to compute. Settings this to `True` blocks the possibility
                to run on the streaming engine for polars.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {
            ...     "foo": [1, 2, 3, 1],
            ...     "bar": ["a", "a", "a", "a"],
            ...     "ham": ["b", "b", "b", "b"],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.unique(["bar", "ham"])

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo bar ham
            0    1   a   b
            >>> func(df_pl)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ str ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ a   ┆ b   │
            └─────┴─────┴─────┘
        """
        return super().unique(subset, keep=keep, maintain_order=maintain_order)

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        r"""
        Filter the rows in the DataFrame based on one or more predicate expressions.

        The original order of the remaining rows is preserved.

        Arguments:
            predicates: Expression(s) that evaluates to a boolean Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6, 7, 8],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            Let's define a dataframe-agnostic function in which we filter on
            one condition.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.filter(nw.col("foo") > 1)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo  bar ham
            1    2    7   b
            2    3    8   c
            >>> func(df_pl)
            shape: (2, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 2   ┆ 7   ┆ b   │
            │ 3   ┆ 8   ┆ c   │
            └─────┴─────┴─────┘

            Filter on multiple conditions, combined with and/or operators:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.filter((nw.col("foo") < 3) & (nw.col("ham") == "a"))
            >>> func(df_pd)
               foo  bar ham
            0    1    6   a
            >>> func(df_pl)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            └─────┴─────┴─────┘

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.filter((nw.col("foo") == 1) | (nw.col("ham") == "c"))
            >>> func(df_pd)
               foo  bar ham
            0    1    6   a
            2    3    8   c
            >>> func(df_pl)
            shape: (2, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            │ 3   ┆ 8   ┆ c   │
            └─────┴─────┴─────┘

            Provide multiple filters using `*args` syntax:

            >>> @nw.narwhalify
            ... def func(df):
            ...     dframe = df.filter(
            ...         nw.col("foo") <= 2,
            ...         ~nw.col("ham").is_in(["b", "c"]),
            ...     )
            ...     return dframe
            >>> func(df_pd)
               foo  bar ham
            0    1    6   a
            >>> func(df_pl)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            └─────┴─────┴─────┘
        """
        return super().filter(*predicates)

    def group_by(self, *keys: str | Iterable[str]) -> GroupBy[Self]:
        r"""
        Start a group by operation.

        Arguments:
            *keys: Column(s) to group by. Accepts multiple columns names as a list.

        Returns:
            GroupBy: Object which can be used to perform aggregations.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {
            ...     "a": ["a", "b", "a", "b", "c"],
            ...     "b": [1, 2, 1, 3, 3],
            ...     "c": [5, 4, 3, 2, 1],
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            Let's define a dataframe-agnostic function in which we group by one column
            and call `agg` to compute the grouped sum of another column.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.group_by("a").agg(nw.col("b").sum()).sort("a")

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a  b
            0  a  2
            1  b  5
            2  c  3
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ str ┆ i64 │
            ╞═════╪═════╡
            │ a   ┆ 2   │
            │ b   ┆ 5   │
            │ c   ┆ 3   │
            └─────┴─────┘

            Group by multiple columns by passing a list of column names.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.group_by(["a", "b"]).agg(nw.max("c")).sort("a", "b")
            >>> func(df_pd)
               a  b  c
            0  a  1  5
            1  b  2  4
            2  b  3  2
            3  c  3  1
            >>> func(df_pl)
            shape: (4, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ a   ┆ 1   ┆ 5   │
            │ b   ┆ 2   ┆ 4   │
            │ b   ┆ 3   ┆ 2   │
            │ c   ┆ 3   ┆ 1   │
            └─────┴─────┴─────┘
        """
        from narwhals.group_by import GroupBy

        return GroupBy(self, *flatten(keys))

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        r"""
        Sort the dataframe by the given columns.

        Arguments:
            by: Column(s) names to sort by.

            *more_by: Additional columns to sort by, specified as positional
                       arguments.

            descending: Sort in descending order. When sorting by multiple
                         columns, can be specified per column by passing a
                         sequence of booleans.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {
            ...     "a": [1, 2, None],
            ...     "b": [6.0, 5.0, 4.0],
            ...     "c": ["a", "c", "b"],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function in which we sort by multiple
            columns in different orders

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.sort("c", "a", descending=[False, True])

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                 a    b  c
            0  1.0  6.0  a
            2  NaN  4.0  b
            1  2.0  5.0  c
            >>> func(df_pl)
            shape: (3, 3)
            ┌──────┬─────┬─────┐
            │ a    ┆ b   ┆ c   │
            │ ---  ┆ --- ┆ --- │
            │ i64  ┆ f64 ┆ str │
            ╞══════╪═════╪═════╡
            │ 1    ┆ 6.0 ┆ a   │
            │ null ┆ 4.0 ┆ b   │
            │ 2    ┆ 5.0 ┆ c   │
            └──────┴─────┴─────┘
        """
        return super().sort(by, *more_by, descending=descending)

    def join(
        self,
        other: Self,
        *,
        how: Literal["inner", "left", "cross", "semi", "anti"] = "inner",
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
    ) -> Self:
        r"""
        Join in SQL-like fashion.

        Arguments:
            other: DataFrame to join with.

            how: Join strategy.

                  * *inner*: Returns rows that have matching values in both tables.
                  * *cross*: Returns the Cartesian product of rows from both tables.
                  * *semi*: Filter rows that have a match in the right table.
                  * *anti*: Filter rows that do not have a match in the right table.

            left_on: Name(s) of the left join column(s).

            right_on: Name(s) of the right join column(s).

        Returns:
            A new joined DataFrame

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6.0, 7.0, 8.0],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> data_other = {
            ...     "apple": ["x", "y", "z"],
            ...     "ham": ["a", "b", "d"],
            ... }

            >>> df_pd = pd.DataFrame(data)
            >>> other_pd = pd.DataFrame(data_other)

            >>> df_pl = pl.DataFrame(data)
            >>> other_pl = pl.DataFrame(data_other)

            Let's define a dataframe-agnostic function in which we join over "ham" column:

            >>> @nw.narwhalify
            ... def join_on_ham(df, other_any):
            ...     return df.join(other_any, left_on="ham", right_on="ham")

            We can now pass either pandas or Polars to the function:

            >>> join_on_ham(df_pd, other_pd)
               foo  bar ham apple
            0    1  6.0   a     x
            1    2  7.0   b     y

            >>> join_on_ham(df_pl, other_pl)
            shape: (2, 4)
            ┌─────┬─────┬─────┬───────┐
            │ foo ┆ bar ┆ ham ┆ apple │
            │ --- ┆ --- ┆ --- ┆ ---   │
            │ i64 ┆ f64 ┆ str ┆ str   │
            ╞═════╪═════╪═════╪═══════╡
            │ 1   ┆ 6.0 ┆ a   ┆ x     │
            │ 2   ┆ 7.0 ┆ b   ┆ y     │
            └─────┴─────┴─────┴───────┘
        """
        return super().join(other, how=how, left_on=left_on, right_on=right_on)

    # --- descriptive ---
    def is_duplicated(self: Self) -> Series:
        r"""
        Get a mask of all duplicated rows in this DataFrame.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> df_pd = pd.DataFrame(
            ...     {
            ...         "a": [1, 2, 3, 1],
            ...         "b": ["x", "y", "z", "x"],
            ...     }
            ... )
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "a": [1, 2, 3, 1],
            ...         "b": ["x", "y", "z", "x"],
            ...     }
            ... )

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.is_duplicated()

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest: +NORMALIZE_WHITESPACE
            0     True
            1    False
            2    False
            3     True
            dtype: bool

            >>> func(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
                true
                false
                false
                true
            ]
        """
        from narwhals.series import Series

        return Series(
            self._compliant_frame.is_duplicated(),
            level=self._level,
        )

    def is_empty(self: Self) -> bool:
        r"""
        Check if the dataframe is empty.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl

            Let's define a dataframe-agnostic function that filters rows in which "foo"
            values are greater than 10, and then checks if the result is empty or not:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.filter(nw.col("foo") > 10).is_empty()

            We can then pass either pandas or Polars to `func`:

            >>> df_pd = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
            >>> df_pl = pl.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
            >>> func(df_pd), func(df_pl)
            (True, True)

            >>> df_pd = pd.DataFrame({"foo": [100, 2, 3], "bar": [4, 5, 6]})
            >>> df_pl = pl.DataFrame({"foo": [100, 2, 3], "bar": [4, 5, 6]})
            >>> func(df_pd), func(df_pl)
            (False, False)
        """
        return self._compliant_frame.is_empty()  # type: ignore[no-any-return]

    def is_unique(self: Self) -> Series:
        r"""
        Get a mask of all unique rows in this DataFrame.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> df_pd = pd.DataFrame(
            ...     {
            ...         "a": [1, 2, 3, 1],
            ...         "b": ["x", "y", "z", "x"],
            ...     }
            ... )
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "a": [1, 2, 3, 1],
            ...         "b": ["x", "y", "z", "x"],
            ...     }
            ... )

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.is_unique()

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest: +NORMALIZE_WHITESPACE
            0    False
            1     True
            2     True
            3    False
            dtype: bool

            >>> func(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
                false
                 true
                 true
                false
            ]
        """
        from narwhals.series import Series

        return Series(
            self._compliant_frame.is_unique(),
            level=self._level,
        )

    def null_count(self: Self) -> Self:
        r"""
        Create a new DataFrame that shows the null counts per column.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> df_pd = pd.DataFrame(
            ...     {
            ...         "foo": [1, None, 3],
            ...         "bar": [6, 7, None],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... )
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "foo": [1, None, 3],
            ...         "bar": [6, 7, None],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... )

            Let's define a dataframe-agnostic function that returns the null count of
            each columns:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.null_count()

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo  bar  ham
            0    1    1    0

            >>> func(df_pl)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ u32 ┆ u32 ┆ u32 │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 1   ┆ 0   │
            └─────┴─────┴─────┘
        """
        return self._from_compliant_dataframe(self._compliant_frame.null_count())

    def item(self: Self, row: int | None = None, column: int | str | None = None) -> Any:
        r"""
        Return the DataFrame as a scalar, or return the element at the given row/column.

        Notes:
            If row/col not provided, this is equivalent to df[0,0], with a check that the shape is (1,1).
            With row/col, this is equivalent to df[row,col].

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function that returns item at given row/column

            >>> @nw.narwhalify
            ... def func(df, row, column):
            ...     return df.item(row, column)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd, 1, 1), func(df_pd, 2, "b")  # doctest:+SKIP
            (5, 6)

            >>> func(df_pl, 1, 1), func(df_pl, 2, "b")
            (5, 6)
        """
        return self._compliant_frame.item(row=row, column=column)

    def clone(self) -> Self:
        r"""
        Create a copy of this DataFrame.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {"a": [1, 2], "b": [3, 4]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function in which we clone the DataFrame:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.clone()

            >>> func(df_pd)
               a  b
            0  1  3
            1  2  4

            >>> func(df_pl)
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 3   │
            │ 2   ┆ 4   │
            └─────┴─────┘
        """
        return super().clone()

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        r"""
        Take every nth row in the DataFrame and return as a new DataFrame.

        Arguments:
            n: Gather every *n*-th row.
            offset: Starting index.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function in which gather every 2 rows,
            starting from a offset of 1:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.gather_every(n=2, offset=1)

            >>> func(df_pd)
               a  b
            1  2  6
            3  4  8

            >>> func(df_pl)
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 2   ┆ 6   │
            │ 4   ┆ 8   │
            └─────┴─────┘
        """
        return super().gather_every(n=n, offset=offset)

    def to_arrow(self: Self) -> pa.Table:
        r"""
        Convert to arrow table.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {"foo": [1, 2, 3], "bar": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function that converts to arrow table:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.to_arrow()

            >>> func(df_pd)  # doctest:+SKIP
            pyarrow.Table
            foo: int64
            bar: string
            ----
            foo: [[1,2,3]]
            bar: [["a","b","c"]]

            >>> func(df_pl)  # doctest:+NORMALIZE_WHITESPACE
            pyarrow.Table
            foo: int64
            bar: large_string
            ----
            foo: [[1,2,3]]
            bar: [["a","b","c"]]
        """
        return self._compliant_frame.to_arrow()


class LazyFrame(BaseFrame[FrameT]):
    """
    Narwhals DataFrame, backed by a native dataframe.

    The native dataframe might be pandas.DataFrame, polars.LazyFrame, ...

    This class is not meant to be instantiated directly - instead, use
    `narwhals.from_native`.
    """

    def __init__(
        self,
        df: Any,
        *,
        level: Literal["full", "interchange"],
    ) -> None:
        self._level = level
        if hasattr(df, "__narwhals_lazyframe__"):
            self._compliant_frame: Any = df.__narwhals_lazyframe__()
        else:  # pragma: no cover
            msg = f"Expected Polars LazyFrame or an object that implements `__narwhals_lazyframe__`, got: {type(df)}"
            raise AssertionError(msg)

    def __repr__(self) -> str:  # pragma: no cover
        header = " Narwhals LazyFrame                            "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Use `narwhals.to_native` to see native output |\n"
            + "└"
            + "─" * length
            + "┘"
        )

    def __getitem__(self, item: str | slice) -> Series | Self:
        msg = "Slicing is not supported on LazyFrame"
        raise TypeError(msg)

    def collect(self) -> DataFrame[Any]:
        r"""
        Materialize this LazyFrame into a DataFrame.

        Returns:
            DataFrame

        Examples:
            >>> import narwhals as nw
            >>> import polars as pl
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "a": ["a", "b", "a", "b", "b", "c"],
            ...         "b": [1, 2, 3, 4, 5, 6],
            ...         "c": [6, 5, 4, 3, 2, 1],
            ...     }
            ... )
            >>> lf = nw.from_native(lf_pl)
            >>> lf
            ┌───────────────────────────────────────────────┐
            | Narwhals LazyFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> df = lf.group_by("a").agg(nw.all().sum()).collect()
            >>> nw.to_native(df).sort("a")
            shape: (3, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ a   ┆ 4   ┆ 10  │
            │ b   ┆ 11  ┆ 10  │
            │ c   ┆ 6   ┆ 1   │
            └─────┴─────┴─────┘
        """
        return DataFrame(
            self._compliant_frame.collect(),
            level=self._level,
        )

    # inherited
    def pipe(self, function: Callable[[Any], Self], *args: Any, **kwargs: Any) -> Self:
        """
        Pipe function call.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {"a": [1, 2, 3], "ba": [4, 5, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.pipe(lambda _df: _df.select("a"))

            We can then pass either pandas or Polars:

            >>> func(df_pd)
               a
            0  1
            1  2
            2  3
            >>> func(df_pl).collect()
            shape: (3, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            │ 2   │
            │ 3   │
            └─────┘
        """
        return super().pipe(function, *args, **kwargs)

    def drop_nulls(self: Self, subset: str | list[str] | None = None) -> Self:
        """
        Drop null values.

        Arguments:
            subset: Column name(s) for which null values are considered. If set to None
                (default), use all columns.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {"a": [1.0, 2.0, None], "ba": [1.0, None, 2.0]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.drop_nulls()

            We can then pass either pandas or Polars:

            >>> func(df_pd)
                 a   ba
            0  1.0  1.0
            >>> func(df_pl).collect()
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ ba  │
            │ --- ┆ --- │
            │ f64 ┆ f64 │
            ╞═════╪═════╡
            │ 1.0 ┆ 1.0 │
            └─────┴─────┘
        """
        return super().drop_nulls(subset=subset)

    def with_row_index(self, name: str = "index") -> Self:
        """
        Insert column which enumerates rows.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_row_index()

            We can then pass either pandas or Polars:

            >>> func(df_pd)
               index  a  b
            0      0  1  4
            1      1  2  5
            2      2  3  6
            >>> func(df_pl).collect()
            shape: (3, 3)
            ┌───────┬─────┬─────┐
            │ index ┆ a   ┆ b   │
            │ ---   ┆ --- ┆ --- │
            │ u32   ┆ i64 ┆ i64 │
            ╞═══════╪═════╪═════╡
            │ 0     ┆ 1   ┆ 4   │
            │ 1     ┆ 2   ┆ 5   │
            │ 2     ┆ 3   ┆ 6   │
            └───────┴─────┴─────┘
        """
        return super().with_row_index(name)

    @property
    def schema(self) -> Schema:
        r"""
        Get an ordered mapping of column names to their data type.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "foo": [1, 2, 3],
            ...         "bar": [6.0, 7.0, 8.0],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... )
            >>> lf = nw.from_native(lf_pl)
            >>> lf.schema  # doctest:+SKIP
            Schema({'foo': Int64, 'bar': Float64, 'ham', String})
        """
        return super().schema

    def collect_schema(self: Self) -> Schema:
        r"""
        Get an ordered mapping of column names to their data type.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "foo": [1, 2, 3],
            ...         "bar": [6.0, 7.0, 8.0],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... )
            >>> lf = nw.from_native(lf_pl)
            >>> lf.collect_schema()  # doctest:+SKIP
            Schema({'foo': Int64, 'bar': Float64, 'ham', String})
        """
        return super().collect_schema()

    @property
    def columns(self) -> list[str]:
        r"""
        Get column names.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> lf_pl = pl.LazyFrame(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.columns

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
            ['foo', 'bar', 'ham']
            >>> func(lf_pl)  # doctest: +SKIP
            ['foo', 'bar', 'ham']
        """
        return super().columns

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        r"""
        Add columns to this LazyFrame.

        Added columns will replace existing columns with the same name.

        Arguments:
            *exprs: Column(s) to add, specified as positional arguments.
                     Accepts expression input. Strings are parsed as column names, other
                     non-expression inputs are parsed as literals.

            **named_exprs: Additional columns to add, specified as keyword arguments.
                            The columns will be renamed to the keyword used.

        Returns:
            LazyFrame: A new LazyFrame with the columns added.

        Note:
            Creating a new LazyFrame using this method does not create a new copy of
            existing data.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {
            ...     "a": [1, 2, 3, 4],
            ...     "b": [0.5, 4, 10, 13],
            ...     "c": [True, True, False, True],
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)
            >>> lf_pl = pl.LazyFrame(df)

            Let's define a dataframe-agnostic function in which we pass an expression
            to add it as a new column:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.with_columns((nw.col("a") * 2).alias("2a"))

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a     b      c  2a
            0  1   0.5   True   2
            1  2   4.0   True   4
            2  3  10.0  False   6
            3  4  13.0   True   8
            >>> func(df_pl)
            shape: (4, 4)
            ┌─────┬──────┬───────┬─────┐
            │ a   ┆ b    ┆ c     ┆ 2a  │
            │ --- ┆ ---  ┆ ---   ┆ --- │
            │ i64 ┆ f64  ┆ bool  ┆ i64 │
            ╞═════╪══════╪═══════╪═════╡
            │ 1   ┆ 0.5  ┆ true  ┆ 2   │
            │ 2   ┆ 4.0  ┆ true  ┆ 4   │
            │ 3   ┆ 10.0 ┆ false ┆ 6   │
            │ 4   ┆ 13.0 ┆ true  ┆ 8   │
            └─────┴──────┴───────┴─────┘
            >>> func(lf_pl).collect()
            shape: (4, 4)
            ┌─────┬──────┬───────┬─────┐
            │ a   ┆ b    ┆ c     ┆ 2a  │
            │ --- ┆ ---  ┆ ---   ┆ --- │
            │ i64 ┆ f64  ┆ bool  ┆ i64 │
            ╞═════╪══════╪═══════╪═════╡
            │ 1   ┆ 0.5  ┆ true  ┆ 2   │
            │ 2   ┆ 4.0  ┆ true  ┆ 4   │
            │ 3   ┆ 10.0 ┆ false ┆ 6   │
            │ 4   ┆ 13.0 ┆ true  ┆ 8   │
            └─────┴──────┴───────┴─────┘
        """
        return super().with_columns(*exprs, **named_exprs)

    def select(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        r"""
        Select columns from this LazyFrame.

        Arguments:
            *exprs: Column(s) to select, specified as positional arguments.
                     Accepts expression input. Strings are parsed as column names,
                     other non-expression inputs are parsed as literals.

            **named_exprs: Additional columns to select, specified as keyword arguments.
                            The columns will be renamed to the keyword used.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6, 7, 8],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)
            >>> lf_pl = pl.LazyFrame(df)

            Let's define a dataframe-agnostic function in which we pass the name of a
            column to select that column.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select("foo")

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo
            0    1
            1    2
            2    3
            >>> func(df_pl)
            shape: (3, 1)
            ┌─────┐
            │ foo │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            │ 2   │
            │ 3   │
            └─────┘
            >>> func(lf_pl).collect()
            shape: (3, 1)
            ┌─────┐
            │ foo │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            │ 2   │
            │ 3   │
            └─────┘

            Multiple columns can be selected by passing a list of column names.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(["foo", "bar"])
            >>> func(df_pd)
               foo  bar
            0    1    6
            1    2    7
            2    3    8
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 6   │
            │ 2   ┆ 7   │
            │ 3   ┆ 8   │
            └─────┴─────┘
            >>> func(lf_pl).collect()
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 6   │
            │ 2   ┆ 7   │
            │ 3   ┆ 8   │
            └─────┴─────┘

            Multiple columns can also be selected using positional arguments instead of a
            list. Expressions are also accepted.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(nw.col("foo"), nw.col("bar") + 1)
            >>> func(df_pd)
               foo  bar
            0    1    7
            1    2    8
            2    3    9
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 7   │
            │ 2   ┆ 8   │
            │ 3   ┆ 9   │
            └─────┴─────┘
            >>> func(lf_pl).collect()
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 7   │
            │ 2   ┆ 8   │
            │ 3   ┆ 9   │
            └─────┴─────┘

            Use keyword arguments to easily name your expression inputs.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.select(threshold=nw.col("foo") * 2)
            >>> func(df_pd)
               threshold
            0          2
            1          4
            2          6
            >>> func(df_pl)
            shape: (3, 1)
            ┌───────────┐
            │ threshold │
            │ ---       │
            │ i64       │
            ╞═══════════╡
            │ 2         │
            │ 4         │
            │ 6         │
            └───────────┘
            >>> func(lf_pl).collect()
            shape: (3, 1)
            ┌───────────┐
            │ threshold │
            │ ---       │
            │ i64       │
            ╞═══════════╡
            │ 2         │
            │ 4         │
            │ 6         │
            └───────────┘
        """
        return super().select(*exprs, **named_exprs)

    def rename(self, mapping: dict[str, str]) -> Self:
        r"""
        Rename column names.

        Arguments:
            mapping: Key value pairs that map from old name to new name, or a
                      function that takes the old name as input and returns the
                      new name.

        Notes:
            If existing names are swapped (e.g. 'A' points to 'B' and 'B'
             points to 'A'), polars will block projection and predicate
             pushdowns at this node.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> lf_pl = pl.LazyFrame(data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.rename({"foo": "apple"})

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               apple  bar ham
            0      1    6   a
            1      2    7   b
            2      3    8   c
            >>> func(lf_pl).collect()
            shape: (3, 3)
            ┌───────┬─────┬─────┐
            │ apple ┆ bar ┆ ham │
            │ ---   ┆ --- ┆ --- │
            │ i64   ┆ i64 ┆ str │
            ╞═══════╪═════╪═════╡
            │ 1     ┆ 6   ┆ a   │
            │ 2     ┆ 7   ┆ b   │
            │ 3     ┆ 8   ┆ c   │
            └───────┴─────┴─────┘
        """
        return super().rename(mapping)

    def head(self, n: int = 5) -> Self:
        r"""
        Get the first `n` rows.

        Arguments:
            n: Number of rows to return.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {
            ...     "a": [1, 2, 3, 4, 5, 6],
            ...     "b": [7, 8, 9, 10, 11, 12],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> lf_pl = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function that gets the first 3 rows.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.head(3)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a  b
            0  1  7
            1  2  8
            2  3  9
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 7   │
            │ 2   ┆ 8   │
            │ 3   ┆ 9   │
            └─────┴─────┘
            >>> func(lf_pl).collect()
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 7   │
            │ 2   ┆ 8   │
            │ 3   ┆ 9   │
            └─────┴─────┘
        """
        return super().head(n)

    def tail(self, n: int = 5) -> Self:
        r"""
        Get the last `n` rows.

        Arguments:
            n: Number of rows to return.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {
            ...     "a": [1, 2, 3, 4, 5, 6],
            ...     "b": [7, 8, 9, 10, 11, 12],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> lf_pl = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function that gets the last 3 rows.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.tail(3)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a   b
            3  4  10
            4  5  11
            5  6  12
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 4   ┆ 10  │
            │ 5   ┆ 11  │
            │ 6   ┆ 12  │
            └─────┴─────┘
            >>> func(lf_pl).collect()
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 4   ┆ 10  │
            │ 5   ┆ 11  │
            │ 6   ┆ 12  │
            └─────┴─────┘
        """
        return super().tail(n)

    def drop(self, *columns: str | Iterable[str], strict: bool = True) -> Self:
        r"""
        Remove columns from the LazyFrame.

        Arguments:
            *columns: Names of the columns that should be removed from the dataframe.
            strict: Validate that all column names exist in the schema and throw an
                exception if a column name does not exist in the schema.

        Warning:
            `strict` argument is ignored for `polars<1.0.0`.

            Please consider upgrading to a newer version or pass to eager mode.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(data)
            >>> lf_pl = pl.LazyFrame(data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.drop("ham")

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo  bar
            0    1  6.0
            1    2  7.0
            2    3  8.0
            >>> func(lf_pl).collect()
            shape: (3, 2)
            ┌─────┬─────┐
            │ foo ┆ bar │
            │ --- ┆ --- │
            │ i64 ┆ f64 │
            ╞═════╪═════╡
            │ 1   ┆ 6.0 │
            │ 2   ┆ 7.0 │
            │ 3   ┆ 8.0 │
            └─────┴─────┘

            Use positional arguments to drop multiple columns.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.drop("foo", "ham")

            >>> func(df_pd)
               bar
            0  6.0
            1  7.0
            2  8.0
            >>> func(lf_pl).collect()
            shape: (3, 1)
            ┌─────┐
            │ bar │
            │ --- │
            │ f64 │
            ╞═════╡
            │ 6.0 │
            │ 7.0 │
            │ 8.0 │
            └─────┘
        """
        return super().drop(*flatten(columns), strict=strict)

    def unique(
        self,
        subset: str | list[str] | None = None,
        *,
        keep: Literal["any", "first", "last", "none"] = "any",
        maintain_order: bool = False,
    ) -> Self:
        """
        Drop duplicate rows from this LazyFrame.

        Arguments:
            subset: Column name(s) to consider when identifying duplicate rows.
                     If set to `None`, use all columns.
            keep: {'first', 'last', 'any', 'none'}
                Which of the duplicate rows to keep.

                * 'any': Does not give any guarantee of which row is kept.
                        This allows more optimizations.
                * 'none': Don't keep duplicate rows.
                * 'first': Keep first unique row.
                * 'last': Keep last unique row.
            maintain_order: Keep the same order as the original DataFrame. This is more
                expensive to compute. Settings this to `True` blocks the possibility
                to run on the streaming engine for polars.

        Returns:
            LazyFrame: LazyFrame with unique rows.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {
            ...     "foo": [1, 2, 3, 1],
            ...     "bar": ["a", "a", "a", "a"],
            ...     "ham": ["b", "b", "b", "b"],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> lf_pl = pl.LazyFrame(data)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.unique(["bar", "ham"])

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo bar ham
            0    1   a   b
            >>> func(lf_pl).collect()
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ str ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ a   ┆ b   │
            └─────┴─────┴─────┘
        """
        return super().unique(subset, keep=keep, maintain_order=maintain_order)

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        r"""
        Filter the rows in the LazyFrame based on a predicate expression.

        The original order of the remaining rows is preserved.

        Arguments:
            *predicates: Expression that evaluates to a boolean Series.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> data = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6, 7, 8],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)
            >>> lf_pl = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function in which we filter on
            one condition.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.filter(nw.col("foo") > 1)

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               foo  bar ham
            1    2    7   b
            2    3    8   c
            >>> func(df_pl)
            shape: (2, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 2   ┆ 7   ┆ b   │
            │ 3   ┆ 8   ┆ c   │
            └─────┴─────┴─────┘
            >>> func(lf_pl).collect()
            shape: (2, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 2   ┆ 7   ┆ b   │
            │ 3   ┆ 8   ┆ c   │
            └─────┴─────┴─────┘

            Filter on multiple conditions:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.filter((nw.col("foo") < 3) & (nw.col("ham") == "a"))
            >>> func(df_pd)
               foo  bar ham
            0    1    6   a
            >>> func(df_pl)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            └─────┴─────┴─────┘
            >>> func(lf_pl).collect()
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            └─────┴─────┴─────┘

            Provide multiple filters using `*args` syntax:

            >>> @nw.narwhalify
            ... def func(df):
            ...     dframe = df.filter(
            ...         nw.col("foo") == 1,
            ...         nw.col("ham") == "a",
            ...     )
            ...     return dframe
            >>> func(df_pd)
               foo  bar ham
            0    1    6   a
            >>> func(df_pl)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            └─────┴─────┴─────┘
            >>> func(lf_pl).collect()
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            └─────┴─────┴─────┘

            Filter on an OR condition:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.filter((nw.col("foo") == 1) | (nw.col("ham") == "c"))
            >>> func(df_pd)
               foo  bar ham
            0    1    6   a
            2    3    8   c
            >>> func(df_pl)
            shape: (2, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            │ 3   ┆ 8   ┆ c   │
            └─────┴─────┴─────┘
            >>> func(lf_pl).collect()
            shape: (2, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            │ 3   ┆ 8   ┆ c   │
            └─────┴─────┴─────┘
        """
        return super().filter(*predicates)

    def group_by(self, *keys: str | Iterable[str]) -> LazyGroupBy[Self]:
        r"""
        Start a group by operation.

        Arguments:
            *keys:
                Column(s) to group by. Accepts expression input. Strings are
                parsed as column names.

        Examples:
            Group by one column and call `agg` to compute the grouped sum of
            another column.

            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {
            ...     "a": ["a", "b", "a", "b", "c"],
            ...     "b": [1, 2, 1, 3, 3],
            ...     "c": [5, 4, 3, 2, 1],
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)
            >>> lf_pl = pl.LazyFrame(df)

            Let's define a dataframe-agnostic function in which we group by one column
            and call `agg` to compute the grouped sum of another column.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.group_by("a").agg(nw.col("b").sum()).sort("a")

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
               a  b
            0  a  2
            1  b  5
            2  c  3
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ str ┆ i64 │
            ╞═════╪═════╡
            │ a   ┆ 2   │
            │ b   ┆ 5   │
            │ c   ┆ 3   │
            └─────┴─────┘
            >>> func(lf_pl).collect()
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ str ┆ i64 │
            ╞═════╪═════╡
            │ a   ┆ 2   │
            │ b   ┆ 5   │
            │ c   ┆ 3   │
            └─────┴─────┘

            Group by multiple columns by passing a list of column names.

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.group_by(["a", "b"]).agg(nw.max("c")).sort(["a", "b"])
            >>> func(df_pd)
               a  b  c
            0  a  1  5
            1  b  2  4
            2  b  3  2
            3  c  3  1
            >>> func(df_pl)
            shape: (4, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ a   ┆ 1   ┆ 5   │
            │ b   ┆ 2   ┆ 4   │
            │ b   ┆ 3   ┆ 2   │
            │ c   ┆ 3   ┆ 1   │
            └─────┴─────┴─────┘
            >>> func(lf_pl).collect()
            shape: (4, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ a   ┆ 1   ┆ 5   │
            │ b   ┆ 2   ┆ 4   │
            │ b   ┆ 3   ┆ 2   │
            │ c   ┆ 3   ┆ 1   │
            └─────┴─────┴─────┘
        """
        from narwhals.group_by import LazyGroupBy

        return LazyGroupBy(self, *flatten(keys))

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        r"""
        Sort the LazyFrame by the given columns.

        Arguments:
            by: Column(s) to sort by. Accepts expression input. Strings are
                 parsed as column names.

            *more_by: Additional columns to sort by, specified as positional
                       arguments.

            descending: Sort in descending order. When sorting by multiple
                         columns, can be specified per column by passing a
                         sequence of booleans.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {
            ...     "a": [1, 2, None],
            ...     "b": [6.0, 5.0, 4.0],
            ...     "c": ["a", "c", "b"],
            ... }
            >>> df_pd = pd.DataFrame(data)
            >>> df_lf = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function in which we sort by multiple
            columns in different orders

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.sort("c", "a", descending=[False, True])

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
                 a    b  c
            0  1.0  6.0  a
            2  NaN  4.0  b
            1  2.0  5.0  c
            >>> func(df_lf).collect()
            shape: (3, 3)
            ┌──────┬─────┬─────┐
            │ a    ┆ b   ┆ c   │
            │ ---  ┆ --- ┆ --- │
            │ i64  ┆ f64 ┆ str │
            ╞══════╪═════╪═════╡
            │ 1    ┆ 6.0 ┆ a   │
            │ null ┆ 4.0 ┆ b   │
            │ 2    ┆ 5.0 ┆ c   │
            └──────┴─────┴─────┘
        """
        return super().sort(by, *more_by, descending=descending)

    def join(
        self,
        other: Self,
        *,
        how: Literal["inner", "left", "cross", "semi", "anti"] = "inner",
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
    ) -> Self:
        r"""
        Add a join operation to the Logical Plan.

        Arguments:
            other: Lazy DataFrame to join with.

            how: Join strategy.

                  * *inner*: Returns rows that have matching values in both tables.
                  * *cross*: Returns the Cartesian product of rows from both tables.
                  * *semi*: Filter rows that have a match in the right table.
                  * *anti*: Filter rows that do not have a match in the right table.

            left_on: Join column of the left DataFrame.

            right_on: Join column of the right DataFrame.

        Returns:
            A new joined LazyFrame

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {
            ...     "foo": [1, 2, 3],
            ...     "bar": [6.0, 7.0, 8.0],
            ...     "ham": ["a", "b", "c"],
            ... }
            >>> data_other = {
            ...     "apple": ["x", "y", "z"],
            ...     "ham": ["a", "b", "d"],
            ... }

            >>> df_pd = pd.DataFrame(data)
            >>> other_pd = pd.DataFrame(data_other)

            >>> df_pl = pl.LazyFrame(data)
            >>> other_pl = pl.LazyFrame(data_other)

            Let's define a dataframe-agnostic function in which we join over "ham" column:

            >>> @nw.narwhalify
            ... def join_on_ham(df, other_any):
            ...     return df.join(other_any, left_on="ham", right_on="ham")

            We can now pass either pandas or Polars to the function:

            >>> join_on_ham(df_pd, other_pd)
               foo  bar ham apple
            0    1  6.0   a     x
            1    2  7.0   b     y

            >>> join_on_ham(df_pl, other_pl).collect()
            shape: (2, 4)
            ┌─────┬─────┬─────┬───────┐
            │ foo ┆ bar ┆ ham ┆ apple │
            │ --- ┆ --- ┆ --- ┆ ---   │
            │ i64 ┆ f64 ┆ str ┆ str   │
            ╞═════╪═════╪═════╪═══════╡
            │ 1   ┆ 6.0 ┆ a   ┆ x     │
            │ 2   ┆ 7.0 ┆ b   ┆ y     │
            └─────┴─────┴─────┴───────┘
        """
        return super().join(other, how=how, left_on=left_on, right_on=right_on)

    def clone(self) -> Self:
        r"""
        Create a copy of this DataFrame.

        >>> import narwhals as nw
        >>> import pandas as pd
        >>> import polars as pl
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> df_pd = pd.DataFrame(data)
        >>> df_pl = pl.LazyFrame(data)

        Let's define a dataframe-agnostic function in which we copy the DataFrame:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.clone()

        >>> func(df_pd)
           a  b
        0  1  3
        1  2  4

        >>> func(df_pl).collect()
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 2   ┆ 4   │
        └─────┴─────┘
        """
        return super().clone()

    def lazy(self) -> Self:
        """
        Lazify the DataFrame (if possible).

        If a library does not support lazy execution, then this is a no-op.

        Examples:
            Construct pandas and Polars objects:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.LazyFrame(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.lazy()

            Note that then, pandas dataframe stay eager, and the Polars LazyFrame stays lazy:

            >>> func(df_pd)
               foo  bar ham
            0    1  6.0   a
            1    2  7.0   b
            2    3  8.0   c
            >>> func(df_pl)
            <LazyFrame ...>
        """
        return super().lazy()  # type: ignore[return-value]

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        r"""
        Take every nth row in the DataFrame and return as a new DataFrame.

        Arguments:
            n: Gather every *n*-th row.
            offset: Starting index.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> data = {"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}
            >>> df_pd = pd.DataFrame(data)
            >>> lf_pl = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function in which gather every 2 rows,
            starting from a offset of 1:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.gather_every(n=2, offset=1)

            >>> func(df_pd)
               a  b
            1  2  6
            3  4  8

            >>> func(lf_pl).collect()
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 2   ┆ 6   │
            │ 4   ┆ 8   │
            └─────┴─────┘
        """
        return super().gather_every(n=n, offset=offset)
