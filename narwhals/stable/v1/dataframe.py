from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast
from typing import overload

from narwhals.dataframe import DataFrame as NwDataFrame
from narwhals.dataframe import LazyFrame as NwLazyFrame
from narwhals.dependencies import get_polars
from narwhals.exceptions import InvalidIntoExprError
from narwhals.typing import IntoDataFrameT
from narwhals.typing import IntoFrameT
from narwhals.utils import Implementation
from narwhals.utils import inherit_doc

if TYPE_CHECKING:
    from types import ModuleType

    from typing_extensions import ParamSpec
    from typing_extensions import Self
    from typing_extensions import TypeVar

    from narwhals.dataframe import MultiColSelector
    from narwhals.dataframe import MultiIndexSelector
    from narwhals.stable.v1.series import Series
    from narwhals.typing import IntoSeries
    from narwhals.typing import SingleColSelector
    from narwhals.typing import SingleIndexSelector

    FrameT = TypeVar("FrameT", "DataFrame[Any]", "LazyFrame[Any]")
    DataFrameT = TypeVar("DataFrameT", bound="DataFrame[Any]")
    LazyFrameT = TypeVar("LazyFrameT", bound="LazyFrame[Any]")
    SeriesT = TypeVar("SeriesT", bound="Series[Any]")
    IntoSeriesT = TypeVar("IntoSeriesT", bound="IntoSeries", default=Any)
    T = TypeVar("T", default=Any)
    P = ParamSpec("P")
    R = TypeVar("R")
else:
    from typing import TypeVar

    IntoSeriesT = TypeVar("IntoSeriesT", bound="IntoSeries")
    T = TypeVar("T")


class DataFrame(NwDataFrame[IntoDataFrameT]):
    @inherit_doc(NwDataFrame)
    def __init__(self, df: Any, *, level: Literal["full", "lazy", "interchange"]) -> None:
        super().__init__(df, level=level)

    # We need to override any method which don't return Self so that type
    # annotations are correct.

    @property
    def _series(self) -> type[Series[Any]]:
        from narwhals.stable.v1.series import Series

        return cast("type[Series[Any]]", Series)

    @property
    def _lazyframe(self) -> type[LazyFrame[Any]]:
        return cast("type[LazyFrame[Any]]", LazyFrame)

    @overload
    def __getitem__(self, item: tuple[SingleIndexSelector, SingleColSelector]) -> Any: ...

    @overload
    def __getitem__(  # type: ignore[overload-overlap]
        self, item: str | tuple[MultiIndexSelector, SingleColSelector]
    ) -> Series[Any]: ...

    @overload
    def __getitem__(
        self,
        item: (
            SingleIndexSelector
            | MultiIndexSelector
            | MultiColSelector
            | tuple[SingleIndexSelector, MultiColSelector]
            | tuple[MultiIndexSelector, MultiColSelector]
        ),
    ) -> Self: ...
    def __getitem__(
        self,
        item: (
            SingleIndexSelector
            | SingleColSelector
            | MultiColSelector
            | MultiIndexSelector
            | tuple[SingleIndexSelector, SingleColSelector]
            | tuple[SingleIndexSelector, MultiColSelector]
            | tuple[MultiIndexSelector, SingleColSelector]
            | tuple[MultiIndexSelector, MultiColSelector]
        ),
    ) -> Series[Any] | Self | Any:
        return super().__getitem__(item)

    def lazy(
        self,
        backend: ModuleType | Implementation | str | None = None,
    ) -> LazyFrame[Any]:
        return super().lazy(backend=backend)  # type: ignore[return-value]

    # Not sure what mypy is complaining about, probably some fancy
    # thing that I need to understand category theory for
    @overload  # type: ignore[override]
    def to_dict(self, *, as_series: Literal[True] = ...) -> dict[str, Series[Any]]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, Series[Any]] | dict[str, list[Any]]: ...
    def to_dict(
        self, *, as_series: bool = True
    ) -> dict[str, Series[Any]] | dict[str, list[Any]]:
        return super().to_dict(as_series=as_series)  # type: ignore[return-value]

    def is_duplicated(self) -> Series[Any]:
        return super().is_duplicated()  # type: ignore[return-value]

    def is_unique(self) -> Series[Any]:
        return super().is_unique()  # type: ignore[return-value]

    def _l1_norm(self) -> Self:
        """Private, just used to test the stable API.

        Returns:
            A new DataFrame.
        """
        from narwhals.stable.v1.functions import all

        return self.select(all()._l1_norm())


class LazyFrame(NwLazyFrame[IntoFrameT]):
    @inherit_doc(NwLazyFrame)
    def __init__(self, df: Any, *, level: Literal["full", "lazy", "interchange"]) -> None:
        super().__init__(df, level=level)

    @property
    def _dataframe(self) -> type[DataFrame[Any]]:
        return DataFrame

    def _extract_compliant(self, arg: Any) -> Any:
        # After v1, we raise when passing order-dependent or length-changing
        # expressions to LazyFrame
        from narwhals.dataframe import BaseFrame
        from narwhals.expr import Expr
        from narwhals.series import Series

        if isinstance(arg, BaseFrame):
            return arg._compliant_frame
        if isinstance(arg, Series):  # pragma: no cover
            msg = "Mixing Series with LazyFrame is not supported."
            raise TypeError(msg)
        if isinstance(arg, Expr):
            # After stable.v1, we raise for order-dependent exprs or filtrations
            return arg._to_compliant_expr(self.__narwhals_namespace__())
        if isinstance(arg, str):
            plx = self.__narwhals_namespace__()
            return plx.col(arg)
        if get_polars() is not None and "polars" in str(type(arg)):  # pragma: no cover
            msg = (
                f"Expected Narwhals object, got: {type(arg)}.\n\n"
                "Perhaps you:\n"
                "- Forgot a `nw.from_native` somewhere?\n"
                "- Used `pl.col` instead of `nw.col`?"
            )
            raise TypeError(msg)
        raise InvalidIntoExprError.from_invalid_type(type(arg))

    def collect(
        self,
        backend: ModuleType | Implementation | str | None = None,
        **kwargs: Any,
    ) -> DataFrame[Any]:
        return super().collect(backend=backend, **kwargs)  # type: ignore[return-value]

    def _l1_norm(self) -> Self:
        """Private, just used to test the stable API.

        Returns:
            A new lazyframe.
        """
        from narwhals.stable.v1.functions import all

        return self.select(all()._l1_norm())

    def tail(self, n: int = 5) -> Self:  # pragma: no cover
        r"""Get the last `n` rows.

        Arguments:
            n: Number of rows to return.

        Returns:
            A subset of the LazyFrame of shape (n, n_columns).
        """
        return super().tail(n)

    def gather_every(self, n: int, offset: int = 0) -> Self:
        r"""Take every nth row in the DataFrame and return as a new DataFrame.

        Arguments:
            n: Gather every *n*-th row.
            offset: Starting index.

        Returns:
            The LazyFrame containing only the selected rows.
        """
        return self._with_compliant(
            self._compliant_frame.gather_every(n=n, offset=offset)
        )
