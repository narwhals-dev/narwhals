from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from narwhals.expr import Expr as NwExpr
from narwhals.utils import find_stacklevel

if TYPE_CHECKING:
    from typing_extensions import Self


class Expr(NwExpr):
    def _l1_norm(self) -> Self:
        return super()._taxicab_norm()

    def head(self, n: int = 10) -> Self:
        r"""Get the first `n` rows.

        Arguments:
            n: Number of rows to return.

        Returns:
            A new expression.
        """
        return self._with_orderable_filtration(
            lambda plx: self._to_compliant_expr(plx).head(n)
        )

    def tail(self, n: int = 10) -> Self:
        r"""Get the last `n` rows.

        Arguments:
            n: Number of rows to return.

        Returns:
            A new expression.
        """
        return self._with_orderable_filtration(
            lambda plx: self._to_compliant_expr(plx).tail(n)
        )

    def gather_every(self, n: int, offset: int = 0) -> Self:
        r"""Take every nth value in the Series and return as new Series.

        Arguments:
            n: Gather every *n*-th row.
            offset: Starting index.

        Returns:
            A new expression.
        """
        return self._with_orderable_filtration(
            lambda plx: self._to_compliant_expr(plx).gather_every(n=n, offset=offset)
        )

    def unique(self, *, maintain_order: bool | None = None) -> Self:
        """Return unique values of this expression.

        Arguments:
            maintain_order: Keep the same order as the original expression.
                This is deprecated and will be removed in a future version,
                but will still be kept around in `narwhals.stable.v1`.

        Returns:
            A new expression.
        """
        if maintain_order is not None:
            msg = (
                "`maintain_order` has no effect and is only kept around for backwards-compatibility. "
                "You can safely remove this argument."
            )
            warn(message=msg, category=UserWarning, stacklevel=find_stacklevel())
        return self._with_filtration(lambda plx: self._to_compliant_expr(plx).unique())

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        """Sort this column. Place null values first.

        Arguments:
            descending: Sort in descending order.
            nulls_last: Place null values last instead of first.

        Returns:
            A new expression.
        """
        return self._with_unorderable_window(
            lambda plx: self._to_compliant_expr(plx).sort(
                descending=descending, nulls_last=nulls_last
            )
        )

    def arg_true(self) -> Self:
        """Find elements where boolean expression is True.

        Returns:
            A new expression.
        """
        return self._with_orderable_filtration(
            lambda plx: self._to_compliant_expr(plx).arg_true(),
        )

    def sample(
        self,
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
        return self._with_filtration(
            lambda plx: self._to_compliant_expr(plx).sample(
                n, fraction=fraction, with_replacement=with_replacement, seed=seed
            )
        )
