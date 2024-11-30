from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import Literal
from typing import Sequence
from typing import overload

from narwhals._arrow.utils import cast_for_truediv
from narwhals._arrow.utils import floordiv_compat
from narwhals._arrow.utils import narwhals_to_native_dtype
from narwhals._arrow.utils import native_to_narwhals_dtype
from narwhals._arrow.utils import parse_datetime_format
from narwhals._arrow.utils import validate_column_comparand
from narwhals.utils import Implementation
from narwhals.utils import generate_temporary_column_name

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes


def maybe_extract_py_scalar(value: Any, return_py_scalar: bool) -> Any:  # noqa: FBT001
    if return_py_scalar:
        return getattr(value, "as_py", lambda: value)()
    return value


class ArrowSeries:
    def __init__(
        self: Self,
        native_series: pa.ChunkedArray,
        *,
        name: str,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> None:
        self._name = name
        self._native_series = native_series
        self._implementation = Implementation.PYARROW
        self._backend_version = backend_version
        self._dtypes = dtypes

    def _change_dtypes(self: Self, dtypes: DTypes) -> Self:
        return self.__class__(
            self._native_series,
            name=self._name,
            backend_version=self._backend_version,
            dtypes=dtypes,
        )

    def _from_native_series(self: Self, series: pa.ChunkedArray | pa.Array) -> Self:
        import pyarrow as pa  # ignore-banned-import()

        if isinstance(series, pa.Array):
            series = pa.chunked_array([series])
        return self.__class__(
            series,
            name=self._name,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    @classmethod
    def _from_iterable(
        cls: type[Self],
        data: Iterable[Any],
        name: str,
        *,
        backend_version: tuple[int, ...],
        dtypes: DTypes,
    ) -> Self:
        import pyarrow as pa  # ignore-banned-import()

        return cls(
            pa.chunked_array([data]),
            name=name,
            backend_version=backend_version,
            dtypes=dtypes,
        )

    def __narwhals_namespace__(self: Self) -> ArrowNamespace:
        from narwhals._arrow.namespace import ArrowNamespace

        return ArrowNamespace(backend_version=self._backend_version, dtypes=self._dtypes)

    def __len__(self: Self) -> int:
        return len(self._native_series)

    def __eq__(self: Self, other: object) -> Self:  # type: ignore[override]
        import pyarrow.compute as pc

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.equal(ser, other))

    def __ne__(self: Self, other: object) -> Self:  # type: ignore[override]
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.not_equal(ser, other))

    def __ge__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.greater_equal(ser, other))

    def __gt__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.greater(ser, other))

    def __le__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.less_equal(ser, other))

    def __lt__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.less(ser, other))

    def __and__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.and_kleene(ser, other))

    def __rand__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.and_kleene(other, ser))

    def __or__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.or_kleene(ser, other))

    def __ror__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.or_kleene(other, ser))

    def __add__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.add(ser, other))

    def __radd__(self: Self, other: Any) -> Self:
        return self + other  # type: ignore[no-any-return]

    def __sub__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.subtract(ser, other))

    def __rsub__(self: Self, other: Any) -> Self:
        return (self - other) * (-1)  # type: ignore[no-any-return]

    def __mul__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.multiply(ser, other))

    def __rmul__(self: Self, other: Any) -> Self:
        return self * other  # type: ignore[no-any-return]

    def __pow__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.power(ser, other))

    def __rpow__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(pc.power(other, ser))

    def __floordiv__(self: Self, other: Any) -> Self:
        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(floordiv_compat(ser, other))

    def __rfloordiv__(self: Self, other: Any) -> Self:
        ser, other = validate_column_comparand(self, other, self._backend_version)
        return self._from_native_series(floordiv_compat(other, ser))

    def __truediv__(self: Self, other: Any) -> Self:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        if not isinstance(other, (pa.Array, pa.ChunkedArray)):
            # scalar
            other = pa.scalar(other)
        return self._from_native_series(pc.divide(*cast_for_truediv(ser, other)))

    def __rtruediv__(self: Self, other: Any) -> Self:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        ser, other = validate_column_comparand(self, other, self._backend_version)
        if not isinstance(other, (pa.Array, pa.ChunkedArray)):
            # scalar
            other = pa.scalar(other)
        return self._from_native_series(pc.divide(*cast_for_truediv(other, ser)))

    def __mod__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        floor_div = (self // other)._native_series
        ser, other = validate_column_comparand(self, other, self._backend_version)
        res = pc.subtract(ser, pc.multiply(floor_div, other))
        return self._from_native_series(res)

    def __rmod__(self: Self, other: Any) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        floor_div = (other // self)._native_series
        ser, other = validate_column_comparand(self, other, self._backend_version)
        res = pc.subtract(other, pc.multiply(floor_div, ser))
        return self._from_native_series(res)

    def __invert__(self: Self) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(pc.invert(self._native_series))

    def len(self: Self, *, _return_py_scalar: bool = True) -> int:
        return maybe_extract_py_scalar(len(self._native_series), _return_py_scalar)  # type: ignore[no-any-return]

    def filter(self: Self, other: Any) -> Self:
        if not (isinstance(other, list) and all(isinstance(x, bool) for x in other)):
            ser, other = validate_column_comparand(self, other, self._backend_version)
        else:
            ser = self._native_series
        return self._from_native_series(ser.filter(other))

    def mean(self: Self, *, _return_py_scalar: bool = True) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        return maybe_extract_py_scalar(pc.mean(self._native_series), _return_py_scalar)  # type: ignore[no-any-return]

    def median(self: Self, *, _return_py_scalar: bool = True) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        from narwhals.exceptions import InvalidOperationError

        if not self.dtype.is_numeric():
            msg = "`median` operation not supported for non-numeric input type."
            raise InvalidOperationError(msg)

        return maybe_extract_py_scalar(  # type: ignore[no-any-return]
            pc.approximate_median(self._native_series), _return_py_scalar
        )

    def min(self: Self, *, _return_py_scalar: bool = True) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        return maybe_extract_py_scalar(pc.min(self._native_series), _return_py_scalar)  # type: ignore[no-any-return]

    def max(self: Self, *, _return_py_scalar: bool = True) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        return maybe_extract_py_scalar(pc.max(self._native_series), _return_py_scalar)  # type: ignore[no-any-return]

    def sum(self: Self, *, _return_py_scalar: bool = True) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        return maybe_extract_py_scalar(pc.sum(self._native_series), _return_py_scalar)  # type: ignore[no-any-return]

    def drop_nulls(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(pc.drop_null(self._native_series))

    def shift(self: Self, n: int) -> Self:
        import pyarrow as pa  # ignore-banned-import()

        ca = self._native_series

        if n > 0:
            result = pa.concat_arrays([pa.nulls(n, ca.type), *ca[:-n].chunks])
        elif n < 0:
            result = pa.concat_arrays([*ca[-n:].chunks, pa.nulls(-n, ca.type)])
        else:
            result = ca
        return self._from_native_series(result)

    def std(self: Self, ddof: int, *, _return_py_scalar: bool = True) -> float:
        import pyarrow.compute as pc  # ignore-banned-import()

        return maybe_extract_py_scalar(  # type: ignore[no-any-return]
            pc.stddev(self._native_series, ddof=ddof), _return_py_scalar
        )

    def skew(self: Self, *, _return_py_scalar: bool = True) -> float | None:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        ser_not_null = pc.drop_null(ser)
        if len(ser_not_null) == 0:
            return None
        elif len(ser_not_null) == 1:
            return float("nan")
        elif len(ser_not_null) == 2:
            return 0.0
        else:
            m = pc.subtract(ser_not_null, pc.mean(ser_not_null))
            m2 = pc.mean(pc.power(m, 2))
            m3 = pc.mean(pc.power(m, 3))
            # Biased population skewness
            return maybe_extract_py_scalar(  # type: ignore[no-any-return]
                pc.divide(m3, pc.power(m2, 1.5)), _return_py_scalar
            )

    def count(self: Self, *, _return_py_scalar: bool = True) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        return maybe_extract_py_scalar(pc.count(self._native_series), _return_py_scalar)  # type: ignore[no-any-return]

    def n_unique(self: Self, *, _return_py_scalar: bool = True) -> int:
        import pyarrow.compute as pc  # ignore-banned-import()

        unique_values = pc.unique(self._native_series)
        return maybe_extract_py_scalar(  # type: ignore[no-any-return]
            pc.count(unique_values, mode="all"), _return_py_scalar
        )

    def __native_namespace__(self: Self) -> ModuleType:
        if self._implementation is Implementation.PYARROW:
            return self._implementation.to_native_namespace()

        msg = f"Expected pyarrow, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    @property
    def name(self: Self) -> str:
        return self._name

    def __narwhals_series__(self: Self) -> Self:
        return self

    @overload
    def __getitem__(self: Self, idx: int) -> Any: ...

    @overload
    def __getitem__(self: Self, idx: slice | Sequence[int]) -> Self: ...

    def __getitem__(self: Self, idx: int | slice | Sequence[int]) -> Any | Self:
        if isinstance(idx, int):
            return self._native_series[idx]
        if isinstance(idx, Sequence):
            return self._from_native_series(self._native_series.take(idx))
        return self._from_native_series(self._native_series[idx])

    def scatter(self: Self, indices: int | Sequence[int], values: Any) -> Self:
        import numpy as np  # ignore-banned-import
        import pyarrow as pa  # ignore-banned-import
        import pyarrow.compute as pc  # ignore-banned-import

        mask = np.zeros(self.len(), dtype=bool)
        mask[indices] = True
        if isinstance(values, self.__class__):
            ser, values = validate_column_comparand(self, values, self._backend_version)
        else:
            ser = self._native_series
        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        if not isinstance(values, pa.Array):
            values = pa.array(values)
        result = pc.replace_with_mask(ser, mask, values.take(indices))
        return self._from_native_series(result)

    def to_list(self: Self) -> list[Any]:
        return self._native_series.to_pylist()  # type: ignore[no-any-return]

    def __array__(self: Self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        return self._native_series.__array__(dtype=dtype, copy=copy)

    def to_numpy(self: Self) -> np.ndarray:
        return self._native_series.to_numpy()

    def alias(self: Self, name: str) -> Self:
        return self.__class__(
            self._native_series,
            name=name,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    @property
    def dtype(self: Self) -> DType:
        return native_to_narwhals_dtype(self._native_series.type, self._dtypes)

    def abs(self: Self) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(pc.abs(self._native_series))

    def cum_sum(self: Self, *, reverse: bool) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        native_series = self._native_series
        result = (
            pc.cumulative_sum(native_series, skip_nulls=True)
            if not reverse
            else pc.cumulative_sum(native_series[::-1], skip_nulls=True)[::-1]
        )
        return self._from_native_series(result)

    def round(self: Self, decimals: int) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(
            pc.round(self._native_series, decimals, round_mode="half_towards_infinity")
        )

    def diff(self: Self) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(
            pc.pairwise_diff(self._native_series.combine_chunks())
        )

    def any(self: Self, *, _return_py_scalar: bool = True) -> bool:
        import pyarrow.compute as pc  # ignore-banned-import()

        return maybe_extract_py_scalar(pc.any(self._native_series), _return_py_scalar)  # type: ignore[no-any-return]

    def all(self: Self, *, _return_py_scalar: bool = True) -> bool:
        import pyarrow.compute as pc  # ignore-banned-import()

        return maybe_extract_py_scalar(pc.all(self._native_series), _return_py_scalar)  # type: ignore[no-any-return]

    def is_between(
        self, lower_bound: Any, upper_bound: Any, closed: str = "both"
    ) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        if closed == "left":
            ge = pc.greater_equal(ser, lower_bound)
            lt = pc.less(ser, upper_bound)
            res = pc.and_kleene(ge, lt)
        elif closed == "right":
            gt = pc.greater(ser, lower_bound)
            le = pc.less_equal(ser, upper_bound)
            res = pc.and_kleene(gt, le)
        elif closed == "none":
            gt = pc.greater(ser, lower_bound)
            lt = pc.less(ser, upper_bound)
            res = pc.and_kleene(gt, lt)
        elif closed == "both":
            ge = pc.greater_equal(ser, lower_bound)
            le = pc.less_equal(ser, upper_bound)
            res = pc.and_kleene(ge, le)
        else:  # pragma: no cover
            raise AssertionError
        return self._from_native_series(res)

    def is_empty(self: Self) -> bool:
        return len(self) == 0

    def is_null(self: Self) -> Self:
        ser = self._native_series
        return self._from_native_series(ser.is_null())

    def cast(self: Self, dtype: DType) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        dtype = narwhals_to_native_dtype(dtype, self._dtypes)
        return self._from_native_series(pc.cast(ser, dtype))

    def null_count(self: Self, *, _return_py_scalar: bool = True) -> int:
        return maybe_extract_py_scalar(self._native_series.null_count, _return_py_scalar)  # type: ignore[no-any-return]

    def head(self: Self, n: int) -> Self:
        ser = self._native_series
        if n >= 0:
            return self._from_native_series(ser.slice(0, n))
        else:
            num_rows = len(ser)
            return self._from_native_series(ser.slice(0, max(0, num_rows + n)))

    def tail(self: Self, n: int) -> Self:
        ser = self._native_series
        if n >= 0:
            num_rows = len(ser)
            return self._from_native_series(ser.slice(max(0, num_rows - n)))
        else:
            return self._from_native_series(ser.slice(abs(n)))

    def is_in(self: Self, other: Any) -> Self:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        value_set = pa.array(other)
        ser = self._native_series
        return self._from_native_series(pc.is_in(ser, value_set=value_set))

    def arg_true(self: Self) -> Self:
        import numpy as np  # ignore-banned-import

        ser = self._native_series
        res = np.flatnonzero(ser)
        return self._from_iterable(
            res,
            name=self.name,
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        )

    def item(self: Self, index: int | None = None) -> Any:
        if index is None:
            if len(self) != 1:
                msg = (
                    "can only call '.item()' if the Series is of length 1,"
                    f" or an explicit index is provided (Series is of length {len(self)})"
                )
                raise ValueError(msg)
            return maybe_extract_py_scalar(self._native_series[0], return_py_scalar=True)
        return maybe_extract_py_scalar(self._native_series[index], return_py_scalar=True)

    def value_counts(
        self: Self,
        *,
        sort: bool = False,
        parallel: bool = False,
        name: str | None = None,
        normalize: bool = False,
    ) -> ArrowDataFrame:
        """Parallel is unused, exists for compatibility."""
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        from narwhals._arrow.dataframe import ArrowDataFrame

        index_name_ = "index" if self._name is None else self._name
        value_name_ = name or ("proportion" if normalize else "count")

        val_count = pc.value_counts(self._native_series)
        values = val_count.field("values")
        counts = val_count.field("counts")

        if normalize:
            counts = pc.divide(*cast_for_truediv(counts, pc.sum(counts)))

        val_count = pa.Table.from_arrays(
            [values, counts], names=[index_name_, value_name_]
        )

        if sort:
            val_count = val_count.sort_by([(value_name_, "descending")])

        return ArrowDataFrame(
            val_count, backend_version=self._backend_version, dtypes=self._dtypes
        )

    def zip_with(self: Self, mask: Self, other: Self) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import()

        mask = mask._native_series.combine_chunks()
        return self._from_native_series(
            pc.if_else(
                mask,
                self._native_series,
                other._native_series,
            )
        )

    def sample(
        self: Self,
        n: int | None,
        *,
        fraction: float | None,
        with_replacement: bool,
        seed: int | None,
    ) -> Self:
        import numpy as np  # ignore-banned-import
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        num_rows = len(self)

        if n is None and fraction is not None:
            n = int(num_rows * fraction)

        rng = np.random.default_rng(seed=seed)
        idx = np.arange(0, num_rows)
        mask = rng.choice(idx, size=n, replace=with_replacement)

        return self._from_native_series(pc.take(ser, mask))

    def fill_null(
        self: Self,
        value: Any | None,
        strategy: Literal["forward", "backward"] | None,
        limit: int | None,
    ) -> Self:
        import numpy as np  # ignore-banned-import
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        def fill_aux(
            arr: pa.Array,
            limit: int,
            direction: Literal["forward", "backward"] | None = None,
        ) -> pa.Array:
            # this algorithm first finds the indices of the valid values to fill all the null value positions
            # then it calculates the distance of each new index and the original index
            # if the distance is equal to or less than the limit and the original value is null, it is replaced
            valid_mask = pc.is_valid(arr)
            indices = pa.array(np.arange(len(arr)), type=pa.int64())
            if direction == "forward":
                valid_index = np.maximum.accumulate(np.where(valid_mask, indices, -1))
                distance = indices - valid_index
            else:
                valid_index = np.minimum.accumulate(
                    np.where(valid_mask[::-1], indices[::-1], len(arr))
                )[::-1]
                distance = valid_index - indices
            return pc.if_else(
                pc.and_(
                    pc.is_null(arr),
                    pc.less_equal(distance, pa.scalar(limit)),
                ),
                arr.take(valid_index),
                arr,
            )

        ser = self._native_series
        dtype = ser.type

        if value is not None:
            res_ser = self._from_native_series(pc.fill_null(ser, pa.scalar(value, dtype)))
        elif limit is None:
            fill_func = (
                pc.fill_null_forward if strategy == "forward" else pc.fill_null_backward
            )
            res_ser = self._from_native_series(fill_func(ser))
        else:
            res_ser = self._from_native_series(fill_aux(ser, limit, strategy))

        return res_ser

    def to_frame(self: Self) -> ArrowDataFrame:
        import pyarrow as pa  # ignore-banned-import()

        from narwhals._arrow.dataframe import ArrowDataFrame

        df = pa.Table.from_arrays([self._native_series], names=[self.name])
        return ArrowDataFrame(
            df, backend_version=self._backend_version, dtypes=self._dtypes
        )

    def to_pandas(self: Self) -> pd.Series:
        import pandas as pd  # ignore-banned-import()

        return pd.Series(self._native_series, name=self.name)

    def is_duplicated(self: Self) -> ArrowSeries:
        return self.to_frame().is_duplicated().alias(self.name)

    def is_unique(self: Self) -> ArrowSeries:
        return self.to_frame().is_unique().alias(self.name)

    def is_first_distinct(self: Self) -> Self:
        import numpy as np  # ignore-banned-import
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        row_number = pa.array(np.arange(len(self)))
        col_token = generate_temporary_column_name(n_bytes=8, columns=[self.name])
        first_distinct_index = (
            pa.Table.from_arrays([self._native_series], names=[self.name])
            .append_column(col_token, row_number)
            .group_by(self.name)
            .aggregate([(col_token, "min")])
            .column(f"{col_token}_min")
        )

        return self._from_native_series(pc.is_in(row_number, first_distinct_index))

    def is_last_distinct(self: Self) -> Self:
        import numpy as np  # ignore-banned-import
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        row_number = pa.array(np.arange(len(self)))
        col_token = generate_temporary_column_name(n_bytes=8, columns=[self.name])
        last_distinct_index = (
            pa.Table.from_arrays([self._native_series], names=[self.name])
            .append_column(col_token, row_number)
            .group_by(self.name)
            .aggregate([(col_token, "max")])
            .column(f"{col_token}_max")
        )

        return self._from_native_series(pc.is_in(row_number, last_distinct_index))

    def is_sorted(self: Self, *, descending: bool) -> bool:
        if not isinstance(descending, bool):
            msg = f"argument 'descending' should be boolean, found {type(descending)}"
            raise TypeError(msg)
        import pyarrow.compute as pc  # ignore-banned-import()

        ser = self._native_series
        if descending:
            result = pc.all(pc.greater_equal(ser[:-1], ser[1:]))
        else:
            result = pc.all(pc.less_equal(ser[:-1], ser[1:]))
        return maybe_extract_py_scalar(result, return_py_scalar=True)  # type: ignore[no-any-return]

    def unique(self: Self, *, maintain_order: bool) -> ArrowSeries:
        # The param `maintain_order` is only here for compatibility with the Polars API
        # and has no effect on the output.
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._from_native_series(pc.unique(self._native_series))

    def replace_strict(
        self, old: Sequence[Any], new: Sequence[Any], *, return_dtype: DType | None
    ) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import
        import pyarrow.compute as pc  # ignore-banned-import

        # https://stackoverflow.com/a/79111029/4451315
        idxs = pc.index_in(self._native_series, pa.array(old))
        result_native = pc.take(pa.array(new), idxs)
        if return_dtype is not None:
            result_native.cast(narwhals_to_native_dtype(return_dtype, self._dtypes))
        result = self._from_native_series(result_native)
        if result.is_null().sum() != self.is_null().sum():
            msg = (
                "replace_strict did not replace all non-null values.\n\n"
                "The following did not get replaced: "
                f"{self.filter(~self.is_null() & result.is_null()).unique(maintain_order=False).to_list()}"
            )
            raise ValueError(msg)
        return result

    def sort(self: Self, *, descending: bool, nulls_last: bool) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        series = self._native_series
        order = "descending" if descending else "ascending"
        null_placement = "at_end" if nulls_last else "at_start"
        sorted_indices = pc.array_sort_indices(
            series, order=order, null_placement=null_placement
        )

        return self._from_native_series(pc.take(series, sorted_indices))

    def to_dummies(self: Self, *, separator: str, drop_first: bool) -> ArrowDataFrame:
        import numpy as np  # ignore-banned-import
        import pyarrow as pa  # ignore-banned-import()

        from narwhals._arrow.dataframe import ArrowDataFrame

        series = self._native_series
        name = self._name
        da = series.dictionary_encode(null_encoding="encode").combine_chunks()

        columns = np.zeros((len(da.dictionary), len(da)), np.int8)
        columns[da.indices, np.arange(len(da))] = 1
        null_col_pa, null_col_pl = f"{name}{separator}None", f"{name}{separator}null"
        cols = [
            {null_col_pa: null_col_pl}.get(
                f"{name}{separator}{v}", f"{name}{separator}{v}"
            )
            for v in da.dictionary
        ]

        output_order = (
            [
                null_col_pl,
                *sorted([c for c in cols if c != null_col_pl])[int(drop_first) :],
            ]
            if null_col_pl in cols
            else sorted(cols)[int(drop_first) :]
        )
        return ArrowDataFrame(
            pa.Table.from_arrays(columns, names=cols),
            backend_version=self._backend_version,
            dtypes=self._dtypes,
        ).select(*output_order)

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
        *,
        _return_py_scalar: bool = True,
    ) -> Any:
        import pyarrow.compute as pc  # ignore-banned-import()

        return maybe_extract_py_scalar(
            pc.quantile(self._native_series, q=quantile, interpolation=interpolation)[0],
            _return_py_scalar,
        )

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        return self._from_native_series(self._native_series[offset::n])

    def clip(self: Self, lower_bound: Any | None, upper_bound: Any | None) -> Self:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._native_series
        arr = pc.max_element_wise(arr, pa.scalar(lower_bound, type=arr.type))
        arr = pc.min_element_wise(arr, pa.scalar(upper_bound, type=arr.type))

        return self._from_native_series(arr)

    def to_arrow(self: Self) -> pa.Array:
        return self._native_series.combine_chunks()

    def mode(self: Self) -> ArrowSeries:
        plx = self.__narwhals_namespace__()
        col_token = generate_temporary_column_name(n_bytes=8, columns=[self.name])
        return self.value_counts(name=col_token, normalize=False).filter(
            plx.col(col_token) == plx.col(col_token).max()
        )[self.name]

    def is_finite(self: Self) -> Self:
        import pyarrow.compute as pc  # ignore-banned-import

        return self._from_native_series(pc.is_finite(self._native_series))

    def cum_count(self: Self, *, reverse: bool) -> Self:
        return (~self.is_null()).cast(self._dtypes.UInt32()).cum_sum(reverse=reverse)

    def cum_min(self: Self, *, reverse: bool) -> Self:
        if self._backend_version < (13, 0, 0):
            msg = "cum_min method is not supported for pyarrow < 13.0.0"
            raise NotImplementedError(msg)

        import pyarrow.compute as pc  # ignore-banned-import

        native_series = self._native_series

        result = (
            pc.cumulative_min(native_series, skip_nulls=True)
            if not reverse
            else pc.cumulative_min(native_series[::-1], skip_nulls=True)[::-1]
        )
        return self._from_native_series(result)

    def cum_max(self: Self, *, reverse: bool) -> Self:
        if self._backend_version < (13, 0, 0):
            msg = "cum_max method is not supported for pyarrow < 13.0.0"
            raise NotImplementedError(msg)

        import pyarrow.compute as pc  # ignore-banned-import

        native_series = self._native_series

        result = (
            pc.cumulative_max(native_series, skip_nulls=True)
            if not reverse
            else pc.cumulative_max(native_series[::-1], skip_nulls=True)[::-1]
        )
        return self._from_native_series(result)

    def cum_prod(self: Self, *, reverse: bool) -> Self:
        if self._backend_version < (13, 0, 0):
            msg = "cum_max method is not supported for pyarrow < 13.0.0"
            raise NotImplementedError(msg)

        import pyarrow.compute as pc  # ignore-banned-import

        native_series = self._native_series

        result = (
            pc.cumulative_prod(native_series, skip_nulls=True)
            if not reverse
            else pc.cumulative_prod(native_series[::-1], skip_nulls=True)[::-1]
        )
        return self._from_native_series(result)

    def rolling_sum(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None,
        center: bool,
    ) -> Self:
        import pyarrow as pa  # ignore-banned-import
        import pyarrow.compute as pc  # ignore-banned-import

        min_periods = min_periods if min_periods is not None else window_size
        if center:
            offset_left = window_size // 2
            offset_right = offset_left - (
                window_size % 2 == 0
            )  # subtract one if window_size is even

            native_series = self._native_series

            pad_left = pa.array([None] * offset_left, type=native_series.type)
            pad_right = pa.array([None] * offset_right, type=native_series.type)
            padded_arr = self._from_native_series(
                pa.concat_arrays([pad_left, native_series.combine_chunks(), pad_right])
            )
        else:
            padded_arr = self

        cum_sum = padded_arr.cum_sum(reverse=False).fill_null(
            value=None, strategy="forward", limit=None
        )
        rolling_sum = (
            cum_sum
            - cum_sum.shift(window_size).fill_null(value=0, strategy=None, limit=None)
            if window_size != 0
            else cum_sum
        )

        valid_count = padded_arr.cum_count(reverse=False)
        count_in_window = valid_count - valid_count.shift(window_size).fill_null(
            value=0, strategy=None, limit=None
        )

        result = self._from_native_series(
            pc.if_else(
                (count_in_window >= min_periods)._native_series,
                rolling_sum._native_series,
                None,
            )
        )
        if center:
            result = result[offset_left + offset_right :]
        return result

    def rolling_mean(
        self: Self,
        window_size: int,
        *,
        min_periods: int | None,
        center: bool,
    ) -> Self:
        import pyarrow as pa  # ignore-banned-import
        import pyarrow.compute as pc  # ignore-banned-import

        min_periods = min_periods if min_periods is not None else window_size
        if center:
            offset_left = window_size // 2
            offset_right = offset_left - (
                window_size % 2 == 0
            )  # subtract one if window_size is even

            native_series = self._native_series

            pad_left = pa.array([None] * offset_left, type=native_series.type)
            pad_right = pa.array([None] * offset_right, type=native_series.type)
            padded_arr = self._from_native_series(
                pa.concat_arrays([pad_left, native_series.combine_chunks(), pad_right])
            )
        else:
            padded_arr = self

        cum_sum = padded_arr.cum_sum(reverse=False).fill_null(
            value=None, strategy="forward", limit=None
        )
        rolling_sum = (
            cum_sum
            - cum_sum.shift(window_size).fill_null(value=0, strategy=None, limit=None)
            if window_size != 0
            else cum_sum
        )

        valid_count = padded_arr.cum_count(reverse=False)
        count_in_window = valid_count - valid_count.shift(window_size).fill_null(
            value=0, strategy=None, limit=None
        )

        result = (
            self._from_native_series(
                pc.if_else(
                    (count_in_window >= min_periods)._native_series,
                    rolling_sum._native_series,
                    None,
                )
            )
            / count_in_window
        )
        if center:
            result = result[offset_left + offset_right :]
        return result

    def __iter__(self: Self) -> Iterator[Any]:
        yield from (
            maybe_extract_py_scalar(x, return_py_scalar=True)
            for x in self._native_series.__iter__()
        )

    @property
    def shape(self: Self) -> tuple[int]:
        return (len(self._native_series),)

    @property
    def dt(self: Self) -> ArrowSeriesDateTimeNamespace:
        return ArrowSeriesDateTimeNamespace(self)

    @property
    def cat(self: Self) -> ArrowSeriesCatNamespace:
        return ArrowSeriesCatNamespace(self)

    @property
    def str(self: Self) -> ArrowSeriesStringNamespace:
        return ArrowSeriesStringNamespace(self)


class ArrowSeriesDateTimeNamespace:
    def __init__(self: Self, series: ArrowSeries) -> None:
        self._arrow_series = series

    def to_string(self: Self, format: str) -> ArrowSeries:  # noqa: A002
        import pyarrow.compute as pc  # ignore-banned-import()

        # PyArrow differs from other libraries in that %S also prints out
        # the fractional part of the second...:'(
        # https://arrow.apache.org/docs/python/generated/pyarrow.compute.strftime.html
        format = format.replace("%S.%f", "%S").replace("%S%.f", "%S")
        return self._arrow_series._from_native_series(
            pc.strftime(self._arrow_series._native_series, format)
        )

    def replace_time_zone(self: Self, time_zone: str | None) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        if time_zone is not None:
            result = pc.assume_timezone(
                pc.local_timestamp(self._arrow_series._native_series), time_zone
            )
        else:
            result = pc.local_timestamp(self._arrow_series._native_series)
        return self._arrow_series._from_native_series(result)

    def convert_time_zone(self: Self, time_zone: str) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import

        if self._arrow_series.dtype.time_zone is None:  # type: ignore[attr-defined]
            result = self.replace_time_zone("UTC")._native_series.cast(
                pa.timestamp(self._arrow_series._native_series.type.unit, time_zone)
            )
        else:
            result = self._arrow_series._native_series.cast(
                pa.timestamp(self._arrow_series._native_series.type.unit, time_zone)
            )

        return self._arrow_series._from_native_series(result)

    def timestamp(self: Self, time_unit: Literal["ns", "us", "ms"] = "us") -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import
        import pyarrow.compute as pc  # ignore-banned-import

        s = self._arrow_series._native_series
        dtype = self._arrow_series.dtype
        if dtype == self._arrow_series._dtypes.Datetime:
            unit = dtype.time_unit  # type: ignore[attr-defined]
            s_cast = s.cast(pa.int64())
            if unit == "ns":
                if time_unit == "ns":
                    result = s_cast
                elif time_unit == "us":
                    result = floordiv_compat(s_cast, 1_000)
                else:
                    result = floordiv_compat(s_cast, 1_000_000)
            elif unit == "us":
                if time_unit == "ns":
                    result = pc.multiply(s_cast, 1_000)
                elif time_unit == "us":
                    result = s_cast
                else:
                    result = floordiv_compat(s_cast, 1_000)
            elif unit == "ms":
                if time_unit == "ns":
                    result = pc.multiply(s_cast, 1_000_000)
                elif time_unit == "us":
                    result = pc.multiply(s_cast, 1_000)
                else:
                    result = s_cast
            elif unit == "s":
                if time_unit == "ns":
                    result = pc.multiply(s_cast, 1_000_000_000)
                elif time_unit == "us":
                    result = pc.multiply(s_cast, 1_000_000)
                else:
                    result = pc.multiply(s_cast, 1_000)
            else:  # pragma: no cover
                msg = f"unexpected time unit {unit}, please report an issue at https://github.com/narwhals-dev/narwhals"
                raise AssertionError(msg)
        elif dtype == self._arrow_series._dtypes.Date:
            time_s = pc.multiply(s.cast(pa.int32()), 86400)
            if time_unit == "ns":
                result = pc.multiply(time_s, 1_000_000_000)
            elif time_unit == "us":
                result = pc.multiply(time_s, 1_000_000)
            else:
                result = pc.multiply(time_s, 1_000)
        else:
            msg = "Input should be either of Date or Datetime type"
            raise TypeError(msg)
        return self._arrow_series._from_native_series(result)

    def date(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            self._arrow_series._native_series.cast(pa.date32())
        )

    def year(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.year(self._arrow_series._native_series)
        )

    def month(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.month(self._arrow_series._native_series)
        )

    def day(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.day(self._arrow_series._native_series)
        )

    def hour(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.hour(self._arrow_series._native_series)
        )

    def minute(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.minute(self._arrow_series._native_series)
        )

    def second(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.second(self._arrow_series._native_series)
        )

    def millisecond(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.millisecond(self._arrow_series._native_series)
        )

    def microsecond(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        result = pc.add(pc.multiply(pc.millisecond(arr), 1000), pc.microsecond(arr))

        return self._arrow_series._from_native_series(result)

    def nanosecond(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        result = pc.add(
            pc.multiply(self.microsecond()._native_series, 1000), pc.nanosecond(arr)
        )
        return self._arrow_series._from_native_series(result)

    def ordinal_day(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.day_of_year(self._arrow_series._native_series)
        )

    def total_minutes(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        unit = arr.type.unit

        unit_to_minutes_factor = {
            "s": 60,  # seconds
            "ms": 60 * 1e3,  # milli
            "us": 60 * 1e6,  # micro
            "ns": 60 * 1e9,  # nano
        }

        factor = pa.scalar(unit_to_minutes_factor[unit], type=pa.int64())
        return self._arrow_series._from_native_series(
            pc.cast(pc.divide(arr, factor), pa.int64())
        )

    def total_seconds(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        unit = arr.type.unit

        unit_to_seconds_factor = {
            "s": 1,  # seconds
            "ms": 1e3,  # milli
            "us": 1e6,  # micro
            "ns": 1e9,  # nano
        }
        factor = pa.scalar(unit_to_seconds_factor[unit], type=pa.int64())

        return self._arrow_series._from_native_series(
            pc.cast(pc.divide(arr, factor), pa.int64())
        )

    def total_milliseconds(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        unit = arr.type.unit

        unit_to_milli_factor = {
            "s": 1e3,  # seconds
            "ms": 1,  # milli
            "us": 1e3,  # micro
            "ns": 1e6,  # nano
        }

        factor = pa.scalar(unit_to_milli_factor[unit], type=pa.int64())

        if unit == "s":
            return self._arrow_series._from_native_series(
                pc.cast(pc.multiply(arr, factor), pa.int64())
            )

        return self._arrow_series._from_native_series(
            pc.cast(pc.divide(arr, factor), pa.int64())
        )

    def total_microseconds(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        unit = arr.type.unit

        unit_to_micro_factor = {
            "s": 1e6,  # seconds
            "ms": 1e3,  # milli
            "us": 1,  # micro
            "ns": 1e3,  # nano
        }

        factor = pa.scalar(unit_to_micro_factor[unit], type=pa.int64())

        if unit in {"s", "ms"}:
            return self._arrow_series._from_native_series(
                pc.cast(pc.multiply(arr, factor), pa.int64())
            )
        return self._arrow_series._from_native_series(
            pc.cast(pc.divide(arr, factor), pa.int64())
        )

    def total_nanoseconds(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()
        import pyarrow.compute as pc  # ignore-banned-import()

        arr = self._arrow_series._native_series
        unit = arr.type.unit

        unit_to_nano_factor = {
            "s": 1e9,  # seconds
            "ms": 1e6,  # milli
            "us": 1e3,  # micro
            "ns": 1,  # nano
        }

        factor = pa.scalar(unit_to_nano_factor[unit], type=pa.int64())

        return self._arrow_series._from_native_series(
            pc.cast(pc.multiply(arr, factor), pa.int64())
        )


class ArrowSeriesCatNamespace:
    def __init__(self: Self, series: ArrowSeries) -> None:
        self._arrow_series = series

    def get_categories(self: Self) -> ArrowSeries:
        import pyarrow as pa  # ignore-banned-import()

        ca = self._arrow_series._native_series
        # TODO(Unassigned): this looks potentially expensive - is there no better way?
        # https://github.com/narwhals-dev/narwhals/issues/464
        out = pa.chunked_array(
            [pa.concat_arrays([x.dictionary for x in ca.chunks]).unique()]
        )
        return self._arrow_series._from_native_series(out)


class ArrowSeriesStringNamespace:
    def __init__(self: Self, series: ArrowSeries) -> None:
        self._arrow_series = series

    def len_chars(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.utf8_length(self._arrow_series._native_series)
        )

    def replace(
        self: Self, pattern: str, value: str, *, literal: bool, n: int
    ) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        method = "replace_substring" if literal else "replace_substring_regex"
        return self._arrow_series._from_native_series(
            getattr(pc, method)(
                self._arrow_series._native_series,
                pattern=pattern,
                replacement=value,
                max_replacements=n,
            )
        )

    def replace_all(
        self: Self, pattern: str, value: str, *, literal: bool
    ) -> ArrowSeries:
        return self.replace(pattern, value, literal=literal, n=-1)

    def strip_chars(self: Self, characters: str | None) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        whitespace = " \t\n\r\v\f"
        return self._arrow_series._from_native_series(
            pc.utf8_trim(
                self._arrow_series._native_series,
                characters or whitespace,
            )
        )

    def starts_with(self: Self, prefix: str) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.equal(self.slice(0, len(prefix))._native_series, prefix)
        )

    def ends_with(self: Self, suffix: str) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.equal(self.slice(-len(suffix), None)._native_series, suffix)
        )

    def contains(self: Self, pattern: str, *, literal: bool) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        check_func = pc.match_substring if literal else pc.match_substring_regex
        return self._arrow_series._from_native_series(
            check_func(self._arrow_series._native_series, pattern)
        )

    def slice(self: Self, offset: int, length: int | None) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        stop = offset + length if length is not None else None
        return self._arrow_series._from_native_series(
            pc.utf8_slice_codeunits(
                self._arrow_series._native_series, start=offset, stop=stop
            ),
        )

    def to_datetime(self: Self, format: str | None) -> ArrowSeries:  # noqa: A002
        import pyarrow.compute as pc  # ignore-banned-import()

        if format is None:
            format = parse_datetime_format(self._arrow_series._native_series)

        return self._arrow_series._from_native_series(
            pc.strptime(self._arrow_series._native_series, format=format, unit="us")
        )

    def to_uppercase(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.utf8_upper(self._arrow_series._native_series),
        )

    def to_lowercase(self: Self) -> ArrowSeries:
        import pyarrow.compute as pc  # ignore-banned-import()

        return self._arrow_series._from_native_series(
            pc.utf8_lower(self._arrow_series._native_series),
        )
