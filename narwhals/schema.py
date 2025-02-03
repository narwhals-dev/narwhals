"""Schema.

Adapted from Polars implementation at:
https://github.com/pola-rs/polars/blob/main/py-polars/polars/schema.py.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING
from typing import Iterable
from typing import Mapping

from narwhals.utils import Implementation
from narwhals.utils import Version
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Any

    import polars as pl
    import pyarrow as pa
    from typing_extensions import Self

    from narwhals.dtypes import DType

    BaseSchema = OrderedDict[str, DType]
else:
    # Python 3.8 does not support generic OrderedDict at runtime
    BaseSchema = OrderedDict

__all__ = ["Schema"]


class Schema(BaseSchema):
    """Ordered mapping of column names to their data type.

    Arguments:
        schema: Mapping[str, DType] | Iterable[tuple[str, DType]] | None
            The schema definition given by column names and their associated.
            *instantiated* Narwhals data type. Accepts a mapping or an iterable of tuples.

    Examples:
        Define a schema by passing *instantiated* data types.

        >>> import narwhals as nw
        >>> schema = nw.Schema({"foo": nw.Int8(), "bar": nw.String()})
        >>> schema
        Schema({'foo': Int8, 'bar': String})

        Access the data type associated with a specific column name.

        >>> schema["foo"]
        Int8

        Access various schema properties using the `names`, `dtypes`, and `len` methods.

        >>> schema.names()
        ['foo', 'bar']
        >>> schema.dtypes()
        [Int8, String]
        >>> schema.len()
        2
    """

    def __init__(
        self: Self,
        schema: Mapping[str, DType] | Iterable[tuple[str, DType]] | None = None,
    ) -> None:
        schema = schema or {}
        super().__init__(schema)

    def names(self: Self) -> list[str]:
        """Get the column names of the schema.

        Returns:
            Column names.
        """
        return list(self.keys())

    def dtypes(self: Self) -> list[DType]:
        """Get the data types of the schema.

        Returns:
            Data types of schema.
        """
        return list(self.values())

    def len(self: Self) -> int:
        """Get the number of columns in the schema.

        Returns:
            Number of columns.
        """
        return len(self)

    def to_native(
        self: Self, *, native_namespace: ModuleType, dtype_backend: str | None = None
    ) -> dict[str, Any] | pl.Schema | pa.Schema:
        implementation = Implementation.from_native_namespace(native_namespace)
        version = Version.MAIN
        if implementation is Implementation.POLARS:
            return self.to_polars(backend=implementation, version=version)
        elif implementation.is_pandas_like():
            return self.to_pandas(
                backend=implementation, version=version, dtype_backend=dtype_backend
            )
        elif implementation is Implementation.PYARROW:
            return self.to_arrow(backend=implementation, version=version)

        raise NotImplementedError

    def to_arrow(
        self: Self, *, backend: ModuleType | Implementation | str, version: Version
    ) -> pa.Schema:
        from narwhals._arrow.utils import narwhals_to_native_dtype

        implementation = Implementation.from_backend(backend)
        schema: pa.Schema = implementation.to_native_namespace().schema(
            (name, narwhals_to_native_dtype(dtype, version))
            for name, dtype in self.items()
        )
        return schema

    def to_pandas(
        self: Self,
        *,
        backend: ModuleType | Implementation | str,
        version: Version,
        dtype_backend: str | None = None,
    ) -> dict[str, Any]:
        from narwhals._pandas_like.utils import narwhals_to_native_dtype

        implementation = Implementation.from_backend(backend)
        backend_version = parse_version(implementation.to_native_namespace().__version__)
        return {
            name: narwhals_to_native_dtype(
                dtype=dtype,
                dtype_backend=dtype_backend,
                implementation=implementation,
                backend_version=backend_version,
                version=version,
            )
            for name, dtype in self.items()
        }

    def to_polars(
        self: Self, *, backend: ModuleType | Implementation | str, version: Version
    ) -> pl.Schema:
        from narwhals._polars.utils import narwhals_to_native_dtype

        implementation = Implementation.from_backend(backend)
        schema: pl.Schema = implementation.to_native_namespace().Schema(
            (name, narwhals_to_native_dtype(dtype, version))
            for name, dtype in self.items()
        )
        return schema
