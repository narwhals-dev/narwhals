from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Sequence
from typing import overload

from narwhals._pandas_like.dataframe import PandasDataFrame
from narwhals.dependencies import get_polars
from narwhals.dtypes import to_narwhals_dtype
from narwhals.translate import get_cudf
from narwhals.translate import get_modin
from narwhals.translate import get_pandas
from narwhals.utils import parse_version
from narwhals.utils import validate_same_library

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.group_by import GroupBy
    from narwhals.group_by import LazyGroupBy
    from narwhals.series import Series
    from narwhals.typing import IntoExpr


class BaseFrame:
    _dataframe: Any
    _is_polars: bool

    def __len__(self) -> Any:
        return self._dataframe.__len__()

    def __native_namespace__(self) -> Any:
        if self._is_polars:
            return get_polars()
        return self._dataframe.__native_namespace__()

    def __narwhals_namespace__(self) -> Any:
        if self._is_polars:
            return get_polars()
        return self._dataframe.__narwhals_namespace__()

    def _from_dataframe(self, df: Any) -> Self:
        # construct, preserving properties
        return self.__class__(  # type: ignore[call-arg]
            df,
            is_polars=self._is_polars,
        )

    def _flatten_and_extract(self, *args: Any, **kwargs: Any) -> Any:
        """Process `args` and `kwargs`, extracting underlying objects as we go."""
        from narwhals.utils import flatten

        args = [self._extract_native(v) for v in flatten(args)]  # type: ignore[assignment]
        kwargs = {k: self._extract_native(v) for k, v in kwargs.items()}
        return args, kwargs

    def _extract_native(self, arg: Any) -> Any:
        from narwhals.expression import Expr
        from narwhals.series import Series

        if isinstance(arg, BaseFrame):
            return arg._dataframe
        if isinstance(arg, Series):
            return arg._series
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
    def schema(self) -> dict[str, DType]:
        return {
            k: to_narwhals_dtype(v, is_polars=self._is_polars)
            for k, v in self._dataframe.schema.items()
        }

    def pipe(self, function: Callable[[Any], Self], *args: Any, **kwargs: Any) -> Self:
        return function(self, *args, **kwargs)

    def with_row_index(self, name: str = "index") -> Self:
        if self._is_polars and parse_version(get_polars().__version__) < parse_version(
            "0.20.4"
        ):  # pragma: no cover
            return self._from_dataframe(
                self._dataframe.with_row_count(name),
            )
        return self._from_dataframe(
            self._dataframe.with_row_index(name),
        )

    def drop_nulls(self) -> Self:
        return self._from_dataframe(
            self._dataframe.drop_nulls(),
        )

    @property
    def columns(self) -> list[str]:
        return self._dataframe.columns  # type: ignore[no-any-return]

    def lazy(self) -> LazyFrame:
        return LazyFrame(
            self._dataframe.lazy(),
        )

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        exprs, named_exprs = self._flatten_and_extract(*exprs, **named_exprs)
        return self._from_dataframe(
            self._dataframe.with_columns(*exprs, **named_exprs),
        )

    def select(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        exprs, named_exprs = self._flatten_and_extract(*exprs, **named_exprs)
        return self._from_dataframe(
            self._dataframe.select(*exprs, **named_exprs),
        )

    def rename(self, mapping: dict[str, str]) -> Self:
        return self._from_dataframe(self._dataframe.rename(mapping))

    def head(self, n: int) -> Self:
        return self._from_dataframe(self._dataframe.head(n))

    def drop(self, *columns: str | Iterable[str]) -> Self:
        return self._from_dataframe(self._dataframe.drop(*columns))

    def unique(self, subset: str | list[str]) -> Self:
        return self._from_dataframe(self._dataframe.unique(subset=subset))

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        predicates, _ = self._flatten_and_extract(*predicates)
        return self._from_dataframe(
            self._dataframe.filter(*predicates),
        )

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        return self._from_dataframe(
            self._dataframe.sort(by, *more_by, descending=descending)
        )

    def join(
        self,
        other: Self,
        *,
        how: Literal["inner"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> Self:
        if how != "inner":
            raise NotImplementedError("Only inner joins are supported for now")
        validate_same_library([self, other])
        return self._from_dataframe(
            self._dataframe.join(
                self._extract_native(other),
                how=how,
                left_on=left_on,
                right_on=right_on,
            )
        )


class DataFrame(BaseFrame):
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
        is_polars: bool = False,
    ) -> None:
        self._is_polars = is_polars
        if hasattr(df, "__narwhals_dataframe__"):
            self._dataframe: Any = df.__narwhals_dataframe__()
        elif is_polars or (
            (pl := get_polars()) is not None and isinstance(df, pl.DataFrame)
        ):
            self._dataframe = df
            self._is_polars = True
        elif (pl := get_polars()) is not None and isinstance(df, pl.LazyFrame):
            raise TypeError(
                "Can't instantiate DataFrame from Polars LazyFrame. Call `collect()` first, or use `narwhals.LazyFrame` if you don't specifically require eager execution."
            )
        elif (pd := get_pandas()) is not None and isinstance(df, pd.DataFrame):
            self._dataframe = PandasDataFrame(df, implementation="pandas")
        elif (mpd := get_modin()) is not None and isinstance(
            df, mpd.DataFrame
        ):  # pragma: no cover
            self._dataframe = PandasDataFrame(df, implementation="modin")
        elif (cudf := get_cudf()) is not None and isinstance(
            df, cudf.DataFrame
        ):  # pragma: no cover
            self._dataframe = PandasDataFrame(df, implementation="cudf")
        else:
            msg = f"Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: {type(df)}"
            raise TypeError(msg)

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._dataframe.to_numpy(*args, **kwargs)

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

    def to_pandas(self) -> Any:
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

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.to_pandas()
            ...     return df

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
        return self._dataframe.to_pandas()

    def to_numpy(self) -> Any:
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

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.to_numpy()
            ...     return df

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
        return self._dataframe.to_numpy()

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

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     return df.shape

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
            (5, 1)
            >>> func(df_pl)
            (5, 1)
        """
        return self._dataframe.shape  # type: ignore[no-any-return]

    @overload
    def __getitem__(self, item: str) -> Series: ...

    @overload
    def __getitem__(self, item: range | slice) -> DataFrame: ...

    def __getitem__(self, item: str | range | slice) -> Series | DataFrame:
        if isinstance(item, str):
            from narwhals.series import Series

            return Series(self._dataframe[item])

        elif isinstance(item, (range, slice)):
            return DataFrame(self._dataframe[item])

        else:
            msg = f"Expected str, range or slice, got: {type(item)}"
            raise TypeError(msg)

    def to_dict(self, *, as_series: bool = True) -> dict[str, Any]:
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
            ...    "A": [1, 2, 3, 4, 5],
            ...    "fruits": ["banana", "banana", "apple", "apple", "banana"],
            ...    "B": [5, 4, 3, 2, 1],
            ...    "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
            ...    "optional": [28, 300, None, 2, -30]
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.to_dict(as_series=False)
            ...     return df

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
            {'A': [1, 2, 3, 4, 5], 'fruits': ['banana', 'banana', 'apple', 'apple', 'banana'], 'B': [5, 4, 3, 2, 1], 'cars': ['beetle', 'audi', 'beetle', 'beetle', 'beetle'], 'optional': [28.0, 300.0, nan, 2.0, -30.0]}
            >>> func(df_pl)
            {'A': [1, 2, 3, 4, 5], 'fruits': ['banana', 'banana', 'apple', 'apple', 'banana'], 'B': [5, 4, 3, 2, 1], 'cars': ['beetle', 'audi', 'beetle', 'beetle', 'beetle'], 'optional': [28, 300, None, 2, -30]}
        """
        from narwhals.series import Series

        if as_series:
            return {
                key: Series(value)
                for key, value in self._dataframe.to_dict(as_series=as_series).items()
            }
        # TODO: overload return type
        return self._dataframe.to_dict(as_series=as_series)  # type: ignore[no-any-return]

    # inherited
    def pipe(self, function: Callable[[Any], Self], *args: Any, **kwargs: Any) -> Self:
        """
        Pipe function call.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {'a': [1,2,3], 'ba': [4,5,6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.pipe(lambda _df: _df.select([x for x in _df.columns if len(x) == 1]))
            ...     return nw.to_native(df)

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

    def drop_nulls(self) -> Self:
        """
        Drop null values.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {'a': [1., 2., None], 'ba': [1, None, 2.]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.drop_nulls()
            ...     return nw.to_native(df)

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
        return super().drop_nulls()

    def with_row_index(self, name: str = "index") -> Self:
        """
        Insert column which enumerates rows.

        Examples:
            Construct pandas as polars DataFrames:

            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {'a': [1,2,3], 'b': [4,5,6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            Let's define a dataframe-agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.with_row_index()
            ...     return nw.to_native(df)

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
    def schema(self) -> dict[str, DType]:
        r"""
        Get a dict[column name, DataType].

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {
            ...         "foo": [1, 2, 3],
            ...         "bar": [6.0, 7.0, 8.0],
            ...         "ham": ["a", "b", "c"],
            ...     }
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a library agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     return df.schema

            You can pass either pandas or Polars to `func`:

            >>> df_pd_schema = func(df_pd)
            >>> df_pd_schema
            {'foo': Int64, 'bar': Float64, 'ham': String}

            >>> df_pl_schema = func(df_pl)
            >>> df_pl_schema
            {'foo': Int64, 'bar': Float64, 'ham': String}

        """
        return super().schema

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

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     return df.columns

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)
            ['foo', 'bar', 'ham']
            >>> func(df_pl)
            ['foo', 'bar', 'ham']
        """
        return super().columns

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
            Pass an expression to add it as a new column.

            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "a": [1, 2, 3, 4],
            ...         "b": [0.5, 4, 10, 13],
            ...         "c": [True, True, False, True],
            ...     }
            ... )
            >>> df = nw.from_native(df_pl)
            >>> dframe = df.with_columns((nw.col("a") * 2).alias("a*2"))
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
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
            Pass the name of a column to select that column.

            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "foo": [1, 2, 3],
            ...         "bar": [6, 7, 8],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... )
            >>> df = nw.DataFrame(df_pl)
            >>> dframe = df.select("foo")
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
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

            >>> dframe = df.select(["foo", "bar"])
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
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

            >>> dframe = df.select(nw.col("foo"), nw.col("bar") + 1)
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
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

            >>> dframe = df.select(threshold=nw.col('foo')*2)
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
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

    def head(self, n: int) -> Self:
        """
        Get the first `n` rows.

        Arguments:
            n: Number of rows to return. If a negative value is passed, return all rows
                except the last `abs(n)`.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3, 4, 5], "bar": [6, 7, 8, 9, 10], "ham": ["a", "b", "c", "d", "e"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.head(3)
            ...     return nw.to_native(df)

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

    def drop(self, *columns: str | Iterable[str]) -> Self:
        """
        Remove columns from the dataframe.

        Arguments:
            *columns: Names of the columns that should be removed from the dataframe.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.drop("ham")
            ...     return nw.to_native(df)

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
        """
        return super().drop(*columns)

    def unique(self, subset: str | list[str]) -> Self:
        """
        Drop duplicate rows from this dataframe.

        Arguments:
            subset: Column name(s) to consider when identifying duplicate rows.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df = {"foo": [1, 2, 3, 1], "bar": ["a", "a", "a", "a"], "ham": ["b", "b", "b", "b"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)

            We define a library agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.unique(["bar", "ham"])
            ...     return nw.to_native(df)

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
        return super().unique(subset)

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        r"""
        Filter the rows in the DataFrame based on one or more predicate expressions.

        The original order of the remaining rows is preserved.

        Arguments:
            predicates: Expression(s) that evaluates to a boolean Series.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "foo": [1, 2, 3],
            ...         "bar": [6, 7, 8],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... )
            >>> df = nw.DataFrame(df_pl)
            >>> df
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘

            Filter on one condition:

            >>> dframe = df.filter(nw.col("foo") > 1)
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
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

            >>> dframe = df.filter((nw.col("foo") < 3) & (nw.col("ham") == "a"))
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            └─────┴─────┴─────┘

            >>> dframe = df.filter((nw.col("foo") == 1) | (nw.col("ham") == "c"))
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
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

            >>> dframe = df.filter(
            ...     nw.col("foo") <= 2,
            ...     ~nw.col("ham").is_in(["b", "c"]),
            ... )
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
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

    def group_by(self, *keys: str | Iterable[str]) -> GroupBy:
        r"""
        Start a group by operation.

        Arguments:
            *keys: Column(s) to group by. Accepts multiple columns names as a list.

        Returns:
            GroupBy: Object which can be used to perform aggregations.

        Examples:
            Group by one column and call `agg` to compute the grouped sum of another
             column.

            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "a": ["a", "b", "a", "b", "c"],
            ...         "b": [1, 2, 1, 3, 3],
            ...         "c": [5, 4, 3, 2, 1],
            ...     }
            ... )
            >>> df = nw.DataFrame(df_pl)
            >>> df
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> dframe = df.group_by("a").agg(nw.col("b").sum()).sort("a")
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
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

            >>> dframe = df.group_by(["a", "b"]).agg(nw.max("c")).sort("a", "b")
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe) # doctest: +SKIP
            shape: (4, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ b   ┆ 2   ┆ 4   │
            │ b   ┆ 3   ┆ 2   │
            │ c   ┆ 3   ┆ 1   │
            │ a   ┆ 1   ┆ 5   │
            └─────┴─────┴─────┘
        """
        from narwhals.group_by import GroupBy

        return GroupBy(self, *keys)

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
            Pass a single column name to sort by that column.

            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "a": [1, 2, None],
            ...         "b": [6.0, 5.0, 4.0],
            ...         "c": ["a", "c", "b"],
            ...     }
            ... )
            >>> df = nw.DataFrame(df_pl)
            >>> dframe = df.sort("a")
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
            shape: (3, 3)
            ┌──────┬─────┬─────┐
            │ a    ┆ b   ┆ c   │
            │ ---  ┆ --- ┆ --- │
            │ i64  ┆ f64 ┆ str │
            ╞══════╪═════╪═════╡
            │ null ┆ 4.0 ┆ b   │
            │ 1    ┆ 6.0 ┆ a   │
            │ 2    ┆ 5.0 ┆ c   │
            └──────┴─────┴─────┘

            Sort by multiple columns by passing a list of columns.

            >>> dframe = df.sort(["c", "a"], descending=True)
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
            shape: (3, 3)
            ┌──────┬─────┬─────┐
            │ a    ┆ b   ┆ c   │
            │ ---  ┆ --- ┆ --- │
            │ i64  ┆ f64 ┆ str │
            ╞══════╪═════╪═════╡
            │ 2    ┆ 5.0 ┆ c   │
            │ null ┆ 4.0 ┆ b   │
            │ 1    ┆ 6.0 ┆ a   │
            └──────┴─────┴─────┘

            Or use positional arguments to sort by multiple columns in the same way.

            >>> dframe = df.sort("c", "a", descending=[False, True])
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
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
        how: Literal["inner"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> Self:
        r"""
        Join in SQL-like fashion.

        Arguments:
            other: DataFrame to join with.

            how: {'inner'}
                  Join strategy.

                  * *inner*: Returns rows that have matching values in both
                              tables

            left_on: Name(s) of the left join column(s).

            right_on: Name(s) of the right join column(s).

        Returns:
            A new joined DataFrame

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "foo": [1, 2, 3],
            ...         "bar": [6.0, 7.0, 8.0],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... )
            >>> other_df_pl = pl.DataFrame(
            ...     {
            ...         "apple": ["x", "y", "z"],
            ...         "ham": ["a", "b", "d"],
            ...     }
            ... )
            >>> df = nw.DataFrame(df_pl)
            >>> other_df = nw.DataFrame(other_df_pl)
            >>> dframe = df.join(other_df, left_on="ham", right_on="ham")
            >>> dframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(dframe)
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

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     duplicated = df.is_duplicated()
            ...     return nw.to_native(duplicated)

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

        return Series(self._dataframe.is_duplicated())

    def is_empty(self: Self) -> bool:
        r"""
        Check if the dataframe is empty.

        Examples:
            >>> import narwhals as nw
            >>> import pandas as pd
            >>> import polars as pl

            Let's define a dataframe-agnostic function that filters rows in which "foo"
            values are greater than 10, and then checks if the result is empty or not:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     return df.filter(nw.col("foo")>10).is_empty()

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

        return self._dataframe.is_empty()  # type: ignore[no-any-return]

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

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     unique = df.is_unique()
            ...     return nw.to_native(unique)

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

        return Series(self._dataframe.is_unique())

    def null_count(self: Self) -> DataFrame:
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

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     null_counts = df.null_count()
            ...     return nw.to_native(null_counts)

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

        return DataFrame(self._dataframe.null_count())


class LazyFrame(BaseFrame):
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
        is_polars: bool = False,
    ) -> None:
        self._is_polars = is_polars
        if hasattr(df, "__narwhals_lazyframe__"):
            self._dataframe: Any = df.__narwhals_lazyframe__()
        elif is_polars or (
            (pl := get_polars()) is not None
            and isinstance(df, (pl.DataFrame, pl.LazyFrame))
        ):
            self._dataframe = df.lazy()
            self._is_polars = True
        elif (pd := get_pandas()) is not None and isinstance(df, pd.DataFrame):
            self._dataframe = PandasDataFrame(df, implementation="pandas")
        elif (mpd := get_modin()) is not None and isinstance(
            df, mpd.DataFrame
        ):  # pragma: no cover
            self._dataframe = PandasDataFrame(df, implementation="modin")
        elif (cudf := get_cudf()) is not None and isinstance(
            df, cudf.DataFrame
        ):  # pragma: no cover
            self._dataframe = PandasDataFrame(df, implementation="cudf")
        else:
            msg = f"Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: {type(df)}"
            raise TypeError(msg)

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

    def __getitem__(self, item: str | range | slice) -> Series | DataFrame:
        raise TypeError("Slicing is not supported on LazyFrame")

    def collect(self) -> DataFrame:
        r"""
        Materialize this LazyFrame into a DataFrame.

        Returns:
            DataFrame

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "a": ["a", "b", "a", "b", "b", "c"],
            ...         "b": [1, 2, 3, 4, 5, 6],
            ...         "c": [6, 5, 4, 3, 2, 1],
            ...     }
            ... )
            >>> lf = nw.LazyFrame(lf_pl)
            >>> lf
            ┌───────────────────────────────────────────────┐
            | Narwhals LazyFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> df = lf.group_by("a").agg(nw.all().sum()).collect()
            >>> df
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
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
            self._dataframe.collect(),
        )

    # inherited
    def pipe(self, function: Callable[[Any], Self], *args: Any, **kwargs: Any) -> Self:
        """
        Pipe function call.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {'a': [1,2,3], 'ba': [4,5,6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.pipe(lambda _df: _df.select([x for x in _df.columns if len(x) == 1]))
            ...     return nw.to_native(df)

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

    def drop_nulls(self) -> Self:
        """
        Drop null values.

        Notes:
            pandas and Polars handle null values differently. Polars distinguishes
            between NaN and Null, whereas pandas doesn't.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {'a': [1., 2., None], 'ba': [1, None, 2.]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.drop_nulls()
            ...     return nw.to_native(df)

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
        return super().drop_nulls()

    def with_row_index(self, name: str = "index") -> Self:
        """
        Insert column which enumerates rows.

        Examples:
            >>> import polars as pl
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> data = {'a': [1,2,3], 'b': [4,5,6]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.LazyFrame(data)

            Let's define a dataframe-agnostic function:

            >>> def func(df_any):
            ...     df = nw.from_native(df_any)
            ...     df = df.with_row_index()
            ...     return nw.to_native(df)

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
    def schema(self) -> dict[str, DType]:
        r"""
        Get a dict[column name, DType].

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
            >>> lf = nw.LazyFrame(lf_pl)
            >>> lf.schema # doctest: +SKIP
            OrderedDict({'foo': Int64, 'bar': Float64, 'ham': String})
        """
        return super().schema

    @property
    def columns(self) -> list[str]:
        r"""
        Get column names.

        Examples:

            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "foo": [1, 2, 3],
            ...         "bar": [6, 7, 8],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... ).select("foo", "bar")
            >>> lf = nw.LazyFrame(lf_pl)
            >>> lf.columns
            ['foo', 'bar']
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
            Pass an expression to add it as a new column.

            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "a": [1, 2, 3, 4],
            ...         "b": [0.5, 4, 10, 13],
            ...         "c": [True, True, False, True],
            ...     }
            ... )
            >>> lf = nw.LazyFrame(lf_pl)
            >>> lframe = lf.with_columns((nw.col("a") * 2).alias("2a")).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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
            Pass the name of a column to select that column.

            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "foo": [1, 2, 3],
            ...         "bar": [6, 7, 8],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... )
            >>> lf = nw.LazyFrame(lf_pl)
            >>> lframe = lf.select("foo").collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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

            >>> lframe = lf.select(["foo", "bar"]).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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

            >>> lframe = lf.select(nw.col("foo"), nw.col("bar") + 1).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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

            >>> lframe = lf.select(threshold=nw.col('foo')*2).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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
            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "foo": [1, 2, 3],
            ...         "bar": [6, 7, 8],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... )
            >>> lf = nw.LazyFrame(lf_pl)
            >>> lframe = lf.rename({"foo": "apple"}).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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

    def head(self, n: int) -> Self:
        r"""
        Get the first `n` rows.

        Arguments:
            n: Number of rows to return.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "a": [1, 2, 3, 4, 5, 6],
            ...         "b": [7, 8, 9, 10, 11, 12],
            ...     }
            ... )
            >>> lf = nw.LazyFrame(lf_pl)
            >>> lframe = lf.head(5).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
            shape: (5, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 7   │
            │ 2   ┆ 8   │
            │ 3   ┆ 9   │
            │ 4   ┆ 10  │
            │ 5   ┆ 11  │
            └─────┴─────┘
            >>> lframe = lf.head(2).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
            shape: (2, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ i64 ┆ i64 │
            ╞═════╪═════╡
            │ 1   ┆ 7   │
            │ 2   ┆ 8   │
            └─────┴─────┘
        """
        return super().head(n)

    def drop(self, *columns: str | Iterable[str]) -> Self:
        r"""
        Remove columns from the LazyFrame.

        Arguments:
            *columns: Names of the columns that should be removed from the
                      dataframe. Accepts column selector input.

        Examples:
            Drop a single column by passing the name of that column.

            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "foo": [1, 2, 3],
            ...         "bar": [6.0, 7.0, 8.0],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... )
            >>> lf = nw.LazyFrame(lf_pl)
            >>> lframe = lf.drop("ham").collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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

            >>> lframe = lf.drop("foo", "ham").collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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
        return super().drop(*columns)

    def unique(self, subset: str | list[str]) -> Self:
        """
        Drop duplicate rows from this LazyFrame.

        Arguments:
            subset: Column name(s) to consider when identifying duplicate rows.
                     If set to `None`, use all columns.

        Returns:
            LazyFrame: LazyFrame with unique rows.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "foo": [1, 2, 3, 1],
            ...         "bar": ["a", "a", "a", "a"],
            ...         "ham": ["b", "b", "b", "b"],
            ...     }
            ... )
            >>> lf = nw.LazyFrame(lf_pl)
            >>> lframe = lf.unique(None).collect().sort("foo")
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
            shape: (3, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ str ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ a   ┆ b   │
            │ 2   ┆ a   ┆ b   │
            │ 3   ┆ a   ┆ b   │
            └─────┴─────┴─────┘
            >>> lframe = lf.unique(subset=["bar", "ham"]).collect().sort("foo")
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ str ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ a   ┆ b   │
            └─────┴─────┴─────┘
        """
        return super().unique(subset)

    def filter(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> Self:
        r"""
        Filter the rows in the LazyFrame based on a predicate expression.

        The original order of the remaining rows is preserved.

        Arguments:
            *predicates: Expression that evaluates to a boolean Series.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "foo": [1, 2, 3],
            ...         "bar": [6, 7, 8],
            ...         "ham": ["a", "b", "c"],
            ...     }
            ... )

            Filter on one condition:

            >>> lf = nw.LazyFrame(lf_pl)
            >>> lframe = lf.filter(nw.col("foo") > 1).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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

            >>> lframe = lf.filter((nw.col("foo") < 3) & (nw.col("ham") == "a")).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            └─────┴─────┴─────┘

            Provide multiple filters using `*args` syntax:

            >>> lframe = lf.filter(
            ...     nw.col("foo") == 1,
            ...     nw.col("ham") == "a",
            ... ).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ foo ┆ bar ┆ ham │
            │ --- ┆ --- ┆ --- │
            │ i64 ┆ i64 ┆ str │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 6   ┆ a   │
            └─────┴─────┴─────┘

            Filter on an OR condition:

            >>> lframe = lf.filter((nw.col("foo") == 1) | (nw.col("ham") == "c")).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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

    def group_by(self, *keys: str | Iterable[str]) -> LazyGroupBy:
        r"""
        Start a group by operation.

        Arguments:
            *keys:
                Column(s) to group by. Accepts expression input. Strings are
                parsed as column names.

        Examples:
            Group by one column and call `agg` to compute the grouped sum of
            another column.

            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "a": ["a", "b", "a", "b", "c"],
            ...         "b": [1, 2, 1, 3, 3],
            ...         "c": [5, 4, 3, 2, 1],
            ...     }
            ... )
            >>> lf = nw.LazyFrame(lf_pl)
            >>> lframe = lf.group_by("a").agg(nw.col("b").sum()).collect().sort("a")
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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

            >>> lframe = lf.group_by(["a", "b"]).agg(nw.max("c")).collect().sort(["a", "b"])
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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

        return LazyGroupBy(self, *keys)

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
            Pass a single column name to sort by that column.

            >>> import polars as pl
            >>> import narwhals as nw
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "a": [1, 2, None],
            ...         "b": [6.0, 5.0, 4.0],
            ...         "c": ["a", "c", "b"],
            ...     }
            ... )
            >>> lf = nw.LazyFrame(lf_pl)
            >>> lframe = lf.sort("a").collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
            shape: (3, 3)
            ┌──────┬─────┬─────┐
            │ a    ┆ b   ┆ c   │
            │ ---  ┆ --- ┆ --- │
            │ i64  ┆ f64 ┆ str │
            ╞══════╪═════╪═════╡
            │ null ┆ 4.0 ┆ b   │
            │ 1    ┆ 6.0 ┆ a   │
            │ 2    ┆ 5.0 ┆ c   │
            └──────┴─────┴─────┘

            Sort by multiple columns by passing a list of columns.

            >>> lframe = lf.sort(["c", "a"], descending=True).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
            shape: (3, 3)
            ┌──────┬─────┬─────┐
            │ a    ┆ b   ┆ c   │
            │ ---  ┆ --- ┆ --- │
            │ i64  ┆ f64 ┆ str │
            ╞══════╪═════╪═════╡
            │ 2    ┆ 5.0 ┆ c   │
            │ null ┆ 4.0 ┆ b   │
            │ 1    ┆ 6.0 ┆ a   │
            └──────┴─────┴─────┘

            Or use positional arguments to sort by multiple columns in the same way.

            >>> lframe = lf.sort("c", "a", descending=[False, True]).collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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
        how: Literal["inner"] = "inner",
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> Self:
        r"""
        Add a join operation to the Logical Plan.

        Arguments:
            other: Lazy DataFrame to join with.

            how: {'inner'}
                  Join strategy.

                  * *inner*: Returns rows that have matching values in both
                              tables

            left_on: Join column of the left DataFrame.

            right_on: Join column of the right DataFrame.

        Returns:
            A new joined LazyFrame

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
            >>> other_lf_pl = pl.LazyFrame(
            ...     {
            ...         "apple": ["x", "y", "z"],
            ...         "ham": ["a", "b", "d"],
            ...     }
            ... )
            >>> lf = nw.LazyFrame(lf_pl)
            >>> other_lf = nw.LazyFrame(other_lf_pl)
            >>> lframe = lf.join(other_lf, left_on="ham", right_on="ham").collect()
            >>> lframe
            ┌───────────────────────────────────────────────┐
            | Narwhals DataFrame                            |
            | Use `narwhals.to_native` to see native output |
            └───────────────────────────────────────────────┘
            >>> nw.to_native(lframe)
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
