from __future__ import annotations

import pyarrow.compute as pc  # ignore-banned-import

__all__ = ["max_horizontal", "min_horizontal"]

# TODO @dangotbanned: Wrap horizontal functions with correct typing
# Should only return scalar if all elements are as well
min_horizontal = pc.min_element_wise
max_horizontal = pc.max_element_wise
