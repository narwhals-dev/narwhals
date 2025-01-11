from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Sequence

from narwhals._expression_parsing import extract_compliant
from narwhals.dtypes import _validate_dtype
from narwhals.expr_cat import ExprCatNamespace
from narwhals.expr_dt import ExprDateTimeNamespace
from narwhals.expr_list import ExprListNamespace
from narwhals.expr_name import ExprNameNamespace
from narwhals.expr_str import ExprStringNamespace
from narwhals.utils import _validate_rolling_arguments
from narwhals.utils import flatten

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.typing import CompliantExpr
    from narwhals.typing import CompliantNamespace
    from narwhals.typing import IntoExpr


class Expr:
    def __init__(self, to_compliant_expr: Callable[[Any], Any]) -> None:
        # callable from CompliantNamespace to CompliantExpr
        self._to_compliant_expr = to_compliant_expr

    def _taxicab_norm(self) -> Self:
        # This is just used to test out the stable api feature in a realistic-ish way.
        # It's not intended to be used.
        return self.__class__(lambda plx: self._to_compliant_expr(plx).abs().sum())

    # --- convert ---
    def alias(self, name: str) -> Self:
        """Rename the expression.

        Arguments:
            name: The new name.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).alias(name))

    def pipe(self, function: Callable[[Any], Self], *args: Any, **kwargs: Any) -> Self:
        """Pipe function call.

        Arguments:
            function: Function to apply.
            args: Positional arguments to pass to function.
            kwargs: Keyword arguments to pass to function.

        Returns:
            A new expression.
        """
        return function(self, *args, **kwargs)

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        """Redefine an object's data type.

        Arguments:
            dtype: Data type that the object will be cast into.

        Returns:
            A new expression.
        """
        _validate_dtype(dtype)
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cast(dtype),
        )

    # --- binary ---
    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__eq__(extract_compliant(plx, other))
        )

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__ne__(extract_compliant(plx, other))
        )

    def __and__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__and__(
                extract_compliant(plx, other)
            )
        )

    def __rand__(self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(extract_compliant(plx, other), dtype=None).__and__(
                extract_compliant(plx, self)
            )

        return self.__class__(func)

    def __or__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__or__(extract_compliant(plx, other))
        )

    def __ror__(self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(extract_compliant(plx, other), dtype=None).__or__(
                extract_compliant(plx, self)
            )

        return self.__class__(func)

    def __add__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__add__(
                extract_compliant(plx, other)
            )
        )

    def __radd__(self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(extract_compliant(plx, other), dtype=None).__add__(
                extract_compliant(plx, self)
            )

        return self.__class__(func)

    def __sub__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__sub__(
                extract_compliant(plx, other)
            )
        )

    def __rsub__(self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(extract_compliant(plx, other), dtype=None).__sub__(
                extract_compliant(plx, self)
            )

        return self.__class__(func)

    def __truediv__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__truediv__(
                extract_compliant(plx, other)
            )
        )

    def __rtruediv__(self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(extract_compliant(plx, other), dtype=None).__truediv__(
                extract_compliant(plx, self)
            )

        return self.__class__(func)

    def __mul__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__mul__(
                extract_compliant(plx, other)
            )
        )

    def __rmul__(self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(extract_compliant(plx, other), dtype=None).__mul__(
                extract_compliant(plx, self)
            )

        return self.__class__(func)

    def __le__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__le__(extract_compliant(plx, other))
        )

    def __lt__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__lt__(extract_compliant(plx, other))
        )

    def __gt__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__gt__(extract_compliant(plx, other))
        )

    def __ge__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__ge__(extract_compliant(plx, other))
        )

    def __pow__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__pow__(
                extract_compliant(plx, other)
            )
        )

    def __rpow__(self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(extract_compliant(plx, other), dtype=None).__pow__(
                extract_compliant(plx, self)
            )

        return self.__class__(func)

    def __floordiv__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__floordiv__(
                extract_compliant(plx, other)
            )
        )

    def __rfloordiv__(self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(extract_compliant(plx, other), dtype=None).__floordiv__(
                extract_compliant(plx, self)
            )

        return self.__class__(func)

    def __mod__(self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__mod__(
                extract_compliant(plx, other)
            )
        )

    def __rmod__(self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(extract_compliant(plx, other), dtype=None).__mod__(
                extract_compliant(plx, self)
            )

        return self.__class__(func)

    # --- unary ---
    def __invert__(self) -> Self:
        return self.__class__(lambda plx: self._to_compliant_expr(plx).__invert__())

    def any(self) -> Self:
        """Return whether any of the values in the column are `True`.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).any())

    def all(self) -> Self:
        """Return whether all values in the column are `True`.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).all())

    def ewm_mean(
        self: Self,
        *,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        min_periods: int = 1,
        ignore_nulls: bool = False,
    ) -> Self:
        r"""Compute exponentially-weighted moving average.

        !!! warning
            This functionality is considered **unstable**. It may be changed at any point
            without it being considered a breaking change.

        Arguments:
            com: Specify decay in terms of center of mass, $\gamma$, with <br> $\alpha = \frac{1}{1+\gamma}\forall\gamma\geq0$
            span: Specify decay in terms of span, $\theta$, with <br> $\alpha = \frac{2}{\theta + 1} \forall \theta \geq 1$
            half_life: Specify decay in terms of half-life, $\tau$, with <br> $\alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \tau } \right\} \forall \tau > 0$
            alpha: Specify smoothing factor alpha directly, $0 < \alpha \leq 1$.
            adjust: Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings

                - When `adjust=True` (the default) the EW function is calculated
                  using weights $w_i = (1 - \alpha)^i$
                - When `adjust=False` the EW function is calculated recursively by
                  $$
                  y_0=x_0
                  $$
                  $$
                  y_t = (1 - \alpha)y_{t - 1} + \alpha x_t
                  $$
            min_periods: Minimum number of observations in window required to have a value, (otherwise result is null).
            ignore_nulls: Ignore missing values when calculating weights.

                - When `ignore_nulls=False` (default), weights are based on absolute
                  positions.
                  For example, the weights of $x_0$ and $x_2$ used in
                  calculating the final weighted average of $[x_0, None, x_2]$ are
                  $(1-\alpha)^2$ and $1$ if `adjust=True`, and
                  $(1-\alpha)^2$ and $\alpha$ if `adjust=False`.
                - When `ignore_nulls=True`, weights are based
                  on relative positions. For example, the weights of
                  $x_0$ and $x_2$ used in calculating the final weighted
                  average of $[x_0, None, x_2]$ are
                  $1-\alpha$ and $1$ if `adjust=True`,
                  and $1-\alpha$ and $\alpha$ if `adjust=False`.

        Returns:
            Expr
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).ewm_mean(
                com=com,
                span=span,
                half_life=half_life,
                alpha=alpha,
                adjust=adjust,
                min_periods=min_periods,
                ignore_nulls=ignore_nulls,
            )
        )

    def mean(self) -> Self:
        """Get mean value.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).mean())

    def median(self) -> Self:
        """Get median value.

        Returns:
            A new expression.

        Notes:
            Results might slightly differ across backends due to differences in the underlying algorithms used to compute the median.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).median())

    def std(self, *, ddof: int = 1) -> Self:
        """Get standard deviation.

        Arguments:
            ddof: "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof,
                where N represents the number of elements. By default ddof is 1.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).std(ddof=ddof))

    def var(self, *, ddof: int = 1) -> Self:
        """Get variance.

        Arguments:
            ddof: "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof,
                     where N represents the number of elements. By default ddof is 1.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).var(ddof=ddof))

    def map_batches(
        self,
        function: Callable[[Any], Self],
        return_dtype: DType | None = None,
    ) -> Self:
        """Apply a custom python function to a whole Series or sequence of Series.

        The output of this custom function is presumed to be either a Series,
        or a NumPy array (in which case it will be automatically converted into
        a Series).

        Arguments:
            function: Function to apply to Series.
            return_dtype: Dtype of the output Series.
                If not set, the dtype will be inferred based on the first non-null value
                that is returned by the function.

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).map_batches(
                function=function, return_dtype=return_dtype
            )
        )

    def skew(self: Self) -> Self:
        """Calculate the sample skewness of a column.

        Returns:
            An expression representing the sample skewness of the column.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).skew())

    def sum(self) -> Expr:
        """Return the sum value.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).sum())

    def min(self) -> Self:
        """Returns the minimum value(s) from a column(s).

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).min())

    def max(self) -> Self:
        """Returns the maximum value(s) from a column(s).

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).max())

    def arg_min(self) -> Self:
        """Returns the index of the minimum value.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).arg_min())

    def arg_max(self) -> Self:
        """Returns the index of the maximum value.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).arg_max())

    def count(self) -> Self:
        """Returns the number of non-null elements in the column.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).count())

    def n_unique(self) -> Self:
        """Returns count of unique values.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).n_unique())

    def unique(self, *, maintain_order: bool = False) -> Self:
        """Return unique values of this expression.

        Arguments:
            maintain_order: Keep the same order as the original expression. This may be more
                expensive to compute. Settings this to `True` blocks the possibility
                to run on the streaming engine for Polars.

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).unique(maintain_order=maintain_order)
        )

    def abs(self) -> Self:
        """Return absolute value of each element.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).abs())

    def cum_sum(self: Self, *, reverse: bool = False) -> Self:
        """Return cumulative sum.

        Arguments:
            reverse: reverse the operation

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cum_sum(reverse=reverse)
        )

    def diff(self) -> Self:
        """Returns the difference between each element and the previous one.

        Returns:
            A new expression.

        Notes:
            pandas may change the dtype here, for example when introducing missing
            values in an integer column. To ensure, that the dtype doesn't change,
            you may want to use `fill_null` and `cast`. For example, to calculate
            the diff and fill missing values with `0` in a Int64 column, you could
            do:

                nw.col("a").diff().fill_null(0).cast(nw.Int64)
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).diff())

    def shift(self, n: int) -> Self:
        """Shift values by `n` positions.

        Arguments:
            n: Number of positions to shift values by.

        Returns:
            A new expression.

        Notes:
            pandas may change the dtype here, for example when introducing missing
            values in an integer column. To ensure, that the dtype doesn't change,
            you may want to use `fill_null` and `cast`. For example, to shift
            and fill missing values with `0` in a Int64 column, you could
            do:

                nw.col("a").shift(1).fill_null(0).cast(nw.Int64)
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).shift(n))

    def replace_strict(
        self,
        old: Sequence[Any] | Mapping[Any, Any],
        new: Sequence[Any] | None = None,
        *,
        return_dtype: DType | type[DType] | None = None,
    ) -> Self:
        """Replace all values by different values.

        This function must replace all non-null input values (else it raises an error).

        Arguments:
            old: Sequence of values to replace. It also accepts a mapping of values to
                their replacement as syntactic sugar for
                `replace_all(old=list(mapping.keys()), new=list(mapping.values()))`.
            new: Sequence of values to replace by. Length must match the length of `old`.
            return_dtype: The data type of the resulting expression. If set to `None`
                (default), the data type is determined automatically based on the other
                inputs.

        Returns:
            A new expression.
        """
        if new is None:
            if not isinstance(old, Mapping):
                msg = "`new` argument is required if `old` argument is not a Mapping type"
                raise TypeError(msg)

            new = list(old.values())
            old = list(old.keys())

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).replace_strict(
                old, new, return_dtype=return_dtype
            )
        )

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        """Sort this column. Place null values first.

        Arguments:
            descending: Sort in descending order.
            nulls_last: Place null values last instead of first.

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).sort(
                descending=descending, nulls_last=nulls_last
            )
        )

    # --- transform ---
    def is_between(
        self: Self,
        lower_bound: Any | IntoExpr,
        upper_bound: Any | IntoExpr,
        closed: Literal["left", "right", "none", "both"] = "both",
    ) -> Self:
        """Check if this expression is between the given lower and upper bounds.

        Arguments:
            lower_bound: Lower bound value.
            upper_bound: Upper bound value.
            closed: Define which sides of the interval are closed (inclusive).

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).is_between(
                extract_compliant(plx, lower_bound),
                extract_compliant(plx, upper_bound),
                closed,
            )
        )

    def is_in(self, other: Any) -> Self:
        """Check if elements of this expression are present in the other iterable.

        Arguments:
            other: iterable

        Returns:
            A new expression.
        """
        if isinstance(other, Iterable) and not isinstance(other, (str, bytes)):
            return self.__class__(
                lambda plx: self._to_compliant_expr(plx).is_in(
                    extract_compliant(plx, other)
                )
            )
        else:
            msg = "Narwhals `is_in` doesn't accept expressions as an argument, as opposed to Polars. You should provide an iterable instead."
            raise NotImplementedError(msg)

    def filter(self, *predicates: Any) -> Self:
        """Filters elements based on a condition, returning a new expression.

        Arguments:
            predicates: Conditions to filter by (which get ANDed together).

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).filter(
                *[extract_compliant(plx, pred) for pred in flatten(predicates)],
            )
        )

    def is_null(self) -> Self:
        """Returns a boolean Series indicating which values are null.

        Returns:
            A new expression.

        Notes:
            pandas handles null values differently from Polars and PyArrow.
            See [null_handling](../pandas_like_concepts/null_handling.md/)
            for reference.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).is_null())

    def is_nan(self) -> Self:
        """Indicate which values are NaN.

        Returns:
            A new expression.

        Notes:
            pandas handles null values differently from Polars and PyArrow.
            See [null_handling](../pandas_like_concepts/null_handling.md/)
            for reference.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).is_nan())

    def arg_true(self) -> Self:
        """Find elements where boolean expression is True.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).arg_true())

    def fill_null(
        self,
        value: Any | None = None,
        strategy: Literal["forward", "backward"] | None = None,
        limit: int | None = None,
    ) -> Self:
        """Fill null values with given value.

        Arguments:
            value: Value used to fill null values.
            strategy: Strategy used to fill null values.
            limit: Number of consecutive null values to fill when using the 'forward' or 'backward' strategy.

        Returns:
            A new expression.

        Notes:
            pandas handles null values differently from Polars and PyArrow.
            See [null_handling](../pandas_like_concepts/null_handling.md/)
            for reference.
        """
        if value is not None and strategy is not None:
            msg = "cannot specify both `value` and `strategy`"
            raise ValueError(msg)
        if value is None and strategy is None:
            msg = "must specify either a fill `value` or `strategy`"
            raise ValueError(msg)
        if strategy is not None and strategy not in {"forward", "backward"}:
            msg = f"strategy not supported: {strategy}"
            raise ValueError(msg)
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).fill_null(
                value=value, strategy=strategy, limit=limit
            )
        )

    # --- partial reduction ---
    def drop_nulls(self) -> Self:
        """Drop null values.

        Returns:
            A new expression.

        Notes:
            pandas handles null values differently from Polars and PyArrow.
            See [null_handling](../pandas_like_concepts/null_handling.md/)
            for reference.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).drop_nulls())

    def sample(
        self: Self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        seed: int | None = None,
    ) -> Self:
        """Sample randomly from this expression.

        Arguments:
            n: Number of items to return. Cannot be used with fraction.
            fraction: Fraction of items to return. Cannot be used with n.
            with_replacement: Allow values to be sampled more than once.
            seed: Seed for the random number generator. If set to None (default), a random
                seed is generated for each sample operation.

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).sample(
                n, fraction=fraction, with_replacement=with_replacement, seed=seed
            )
        )

    def over(self, *keys: str | Iterable[str]) -> Self:
        """Compute expressions over the given groups.

        Arguments:
            keys: Names of columns to compute window expression over.
                  Must be names of columns, as opposed to expressions -
                  so, this is a bit less flexible than Polars' `Expr.over`.

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).over(flatten(keys))
        )

    def is_duplicated(self) -> Self:
        r"""Return a boolean mask indicating duplicated values.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).is_duplicated())

    def is_unique(self) -> Self:
        r"""Return a boolean mask indicating unique values.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).is_unique())

    def null_count(self) -> Self:
        r"""Count null values.

        Returns:
            A new expression.

        Notes:
            pandas handles null values differently from Polars and PyArrow.
            See [null_handling](../pandas_like_concepts/null_handling.md/)
            for reference.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).null_count())

    def is_first_distinct(self) -> Self:
        r"""Return a boolean mask indicating the first occurrence of each distinct value.

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).is_first_distinct()
        )

    def is_last_distinct(self) -> Self:
        r"""Return a boolean mask indicating the last occurrence of each distinct value.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).is_last_distinct())

    def quantile(
        self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        r"""Get quantile value.

        Arguments:
            quantile: Quantile between 0.0 and 1.0.
            interpolation: Interpolation method.

        Returns:
            A new expression.

        Note:
            - pandas and Polars may have implementation differences for a given interpolation method.
            - [dask](https://docs.dask.org/en/stable/generated/dask.dataframe.Series.quantile.html) has
                its own method to approximate quantile and it doesn't implement 'nearest', 'higher',
                'lower', 'midpoint' as interpolation method - use 'linear' which is closest to the
                native 'dask' - method.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).quantile(quantile, interpolation)
        )

    def head(self, n: int = 10) -> Self:
        r"""Get the first `n` rows.

        Arguments:
            n: Number of rows to return.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).head(n))

    def tail(self, n: int = 10) -> Self:
        r"""Get the last `n` rows.

        Arguments:
            n: Number of rows to return.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).tail(n))

    def round(self, decimals: int = 0) -> Self:
        r"""Round underlying floating point data by `decimals` digits.

        Arguments:
            decimals: Number of decimals to round by.

        Returns:
            A new expression.


        Notes:
            For values exactly halfway between rounded decimal values pandas behaves differently than Polars and Arrow.

            pandas rounds to the nearest even value (e.g. -0.5 and 0.5 round to 0.0, 1.5 and 2.5 round to 2.0, 3.5 and
            4.5 to 4.0, etc..).

            Polars and Arrow round away from 0 (e.g. -0.5 to -1.0, 0.5 to 1.0, 1.5 to 2.0, 2.5 to 3.0, etc..).
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).round(decimals))

    def len(self) -> Self:
        r"""Return the number of elements in the column.

        Null values count towards the total.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).len())

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        r"""Take every nth value in the Series and return as new Series.

        Arguments:
            n: Gather every *n*-th row.
            offset: Starting index.

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).gather_every(n=n, offset=offset)
        )

    # need to allow numeric typing
    # TODO @aivanoved: make type alias for numeric type
    def clip(
        self,
        lower_bound: IntoExpr | Any | None = None,
        upper_bound: IntoExpr | Any | None = None,
    ) -> Self:
        r"""Clip values in the Series.

        Arguments:
            lower_bound: Lower bound value.
            upper_bound: Upper bound value.

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).clip(
                extract_compliant(plx, lower_bound),
                extract_compliant(plx, upper_bound),
            )
        )

    def mode(self: Self) -> Self:
        r"""Compute the most occurring value(s).

        Can return multiple values.

        Returns:
            A new expression.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).mode())

    def is_finite(self: Self) -> Self:
        """Returns boolean values indicating which original values are finite.

        Warning:
            Different backend handle null values differently. `is_finite` will return
            False for NaN and Null's in the Dask and pandas non-nullable backend, while
            for Polars, PyArrow and pandas nullable backends null values are kept as such.

        Returns:
            Expression of `Boolean` data type.
        """
        return self.__class__(lambda plx: self._to_compliant_expr(plx).is_finite())

    def cum_count(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative count of the non-null values in the column.

        Arguments:
            reverse: reverse the operation

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cum_count(reverse=reverse)
        )

    def cum_min(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative min of the non-null values in the column.

        Arguments:
            reverse: reverse the operation

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cum_min(reverse=reverse)
        )

    def cum_max(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative max of the non-null values in the column.

        Arguments:
            reverse: reverse the operation

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cum_max(reverse=reverse)
        )

    def cum_prod(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative product of the non-null values in the column.

        Arguments:
            reverse: reverse the operation

        Returns:
            A new expression.
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cum_prod(reverse=reverse)
        )

    def rolling_sum(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None = None,
        center: bool = False,
    ) -> Self:
        """Apply a rolling sum (moving sum) over the values.

        !!! warning
            This functionality is considered **unstable**. It may be changed at any point
            without it being considered a breaking change.

        A window of length `window_size` will traverse the values. The resulting values
        will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Arguments:
            window_size: The length of the window in number of elements. It must be a
                strictly positive integer.
            min_periods: The number of values in the window that should be non-null before
                computing a result. If set to `None` (default), it will be set equal to
                `window_size`. If provided, it must be a strictly positive integer, and
                less than or equal to `window_size`
            center: Set the labels at the center of the window.

        Returns:
            A new expression.
        """
        window_size, min_periods = _validate_rolling_arguments(
            window_size=window_size, min_periods=min_periods
        )

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).rolling_sum(
                window_size=window_size,
                min_periods=min_periods,
                center=center,
            )
        )

    def rolling_mean(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None = None,
        center: bool = False,
    ) -> Self:
        """Apply a rolling mean (moving mean) over the values.

        !!! warning
            This functionality is considered **unstable**. It may be changed at any point
            without it being considered a breaking change.

        A window of length `window_size` will traverse the values. The resulting values
        will be aggregated to their mean.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Arguments:
            window_size: The length of the window in number of elements. It must be a
                strictly positive integer.
            min_periods: The number of values in the window that should be non-null before
                computing a result. If set to `None` (default), it will be set equal to
                `window_size`. If provided, it must be a strictly positive integer, and
                less than or equal to `window_size`
            center: Set the labels at the center of the window.

        Returns:
            A new expression.
        """
        window_size, min_periods = _validate_rolling_arguments(
            window_size=window_size, min_periods=min_periods
        )

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).rolling_mean(
                window_size=window_size,
                min_periods=min_periods,
                center=center,
            )
        )

    def rolling_var(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None = None,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        """Apply a rolling variance (moving variance) over the values.

        !!! warning
            This functionality is considered **unstable**. It may be changed at any point
            without it being considered a breaking change.

        A window of length `window_size` will traverse the values. The resulting values
        will be aggregated to their variance.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Arguments:
            window_size: The length of the window in number of elements. It must be a
                strictly positive integer.
            min_periods: The number of values in the window that should be non-null before
                computing a result. If set to `None` (default), it will be set equal to
                `window_size`. If provided, it must be a strictly positive integer, and
                less than or equal to `window_size`.
            center: Set the labels at the center of the window.
            ddof: Delta Degrees of Freedom; the divisor for a length N window is N - ddof.

        Returns:
            A new expression.
        """
        window_size, min_periods = _validate_rolling_arguments(
            window_size=window_size, min_periods=min_periods
        )

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).rolling_var(
                window_size=window_size, min_periods=min_periods, center=center, ddof=ddof
            )
        )

    def rolling_std(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None = None,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        """Apply a rolling standard deviation (moving standard deviation) over the values.

        !!! warning
            This functionality is considered **unstable**. It may be changed at any point
            without it being considered a breaking change.

        A window of length `window_size` will traverse the values. The resulting values
        will be aggregated to their standard deviation.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Arguments:
            window_size: The length of the window in number of elements. It must be a
                strictly positive integer.
            min_periods: The number of values in the window that should be non-null before
                computing a result. If set to `None` (default), it will be set equal to
                `window_size`. If provided, it must be a strictly positive integer, and
                less than or equal to `window_size`.
            center: Set the labels at the center of the window.
            ddof: Delta Degrees of Freedom; the divisor for a length N window is N - ddof.

        Returns:
            A new expression.
        """
        window_size, min_periods = _validate_rolling_arguments(
            window_size=window_size, min_periods=min_periods
        )

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).rolling_std(
                window_size=window_size,
                min_periods=min_periods,
                center=center,
                ddof=ddof,
            )
        )

    def rank(
        self: Self,
        method: Literal["average", "min", "max", "dense", "ordinal"] = "average",
        *,
        descending: bool = False,
    ) -> Self:
        """Assign ranks to data, dealing with ties appropriately.

        Notes:
            The resulting dtype may differ between backends.

        Arguments:
            method: The method used to assign ranks to tied elements.
                The following methods are available (default is 'average'):

                - 'average' : The average of the ranks that would have been assigned to
                  all the tied values is assigned to each value.
                - 'min' : The minimum of the ranks that would have been assigned to all
                    the tied values is assigned to each value. (This is also referred to
                    as "competition" ranking.)
                - 'max' : The maximum of the ranks that would have been assigned to all
                    the tied values is assigned to each value.
                - 'dense' : Like 'min', but the rank of the next highest element is
                   assigned the rank immediately after those assigned to the tied
                   elements.
                - 'ordinal' : All values are given a distinct rank, corresponding to the
                    order that the values occur in the Series.

            descending: Rank in descending order.

        Returns:
            A new expression with rank data.
        """
        supported_rank_methods = {"average", "min", "max", "dense", "ordinal"}
        if method not in supported_rank_methods:
            msg = (
                "Ranking method must be one of {'average', 'min', 'max', 'dense', 'ordinal'}. "
                f"Found '{method}'"
            )
            raise ValueError(msg)

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).rank(
                method=method, descending=descending
            )
        )

    @property
    def str(self: Self) -> ExprStringNamespace[Self]:
        return ExprStringNamespace(self)

    @property
    def dt(self: Self) -> ExprDateTimeNamespace[Self]:
        return ExprDateTimeNamespace(self)

    @property
    def cat(self: Self) -> ExprCatNamespace[Self]:
        return ExprCatNamespace(self)

    @property
    def name(self: Self) -> ExprNameNamespace[Self]:
        return ExprNameNamespace(self)

    @property
    def list(self: Self) -> ExprListNamespace[Self]:
        return ExprListNamespace(self)


__all__ = [
    "Expr",
]
