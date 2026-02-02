from __future__ import annotations

import pyarrow.compute as pc  # ignore-banned-import

__all__ = ["coalesce", "max_horizontal", "min_horizontal"]

# TODO @dangotbanned: Wrap horizontal functions with correct typing
# Should only return scalar if all elements are as well
# NOTE: Changing typing will propagate to a lot of places (so be careful!):
# - `_round.{clip,clip_lower,clip_upper}`
# - `acero.join_asof_tables`
# - `ArrowNamespace.{min,max}_horizontal`
# - `ArrowNamespace.coalesce`
# - `ArrowSeries.rolling_var`
min_horizontal = pc.min_element_wise
max_horizontal = pc.max_element_wise
coalesce = pc.coalesce
