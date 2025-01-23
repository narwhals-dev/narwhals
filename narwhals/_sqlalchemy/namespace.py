from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from sqlalchemy import Column
from sqlalchemy import literal

from narwhals._sqlalchemy.expr import SQLAlchemyExpr
from narwhals.typing import CompliantNamespace

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._sqlalchemy.dataframe import SQLAlchemyLazyFrame
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class SQLAlchemyNamespace(CompliantNamespace["Column"]):
    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._version = version

    def all(self: Self) -> SQLAlchemyExpr:
        def _all(df: SQLAlchemyLazyFrame) -> list[Column]:
            return [Column(col_name) for col_name in df.columns]

        return SQLAlchemyExpr(  # type: ignore[abstract]
            call=_all,
            depth=0,
            function_name="all",
            root_names=None,
            output_names=None,
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def col(self: Self, *column_names: str) -> SQLAlchemyExpr:
        return SQLAlchemyExpr.from_column_names(
            *column_names, backend_version=self._backend_version, version=self._version
        )

    def lit(self: Self, value: Any, dtype: DType | None) -> SQLAlchemyExpr:
        def func(_df: SQLAlchemyLazyFrame) -> list[Column]:
            if dtype is not None:
                msg = "todo"
                raise NotImplementedError(msg)

            return [literal(value)]

        return SQLAlchemyExpr(  # type: ignore[abstract]
            func,
            depth=0,
            function_name="lit",
            root_names=None,
            output_names=["literal"],
            returns_scalar=True,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )
