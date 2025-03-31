"""Narwhals-level equivalent of `CompliantNamespace`.

Aiming to solve 2 distinct issues.

### 1. A unified entry point for creating a `CompliantNamespace`

Currently lots of ways we do this:
- Most recently `nw.utils._into_compliant_namespace`
- Creating an object, then using `__narwhals_namespace__`
- Generally repeating logic in multiple places


### 2. Typing and no `lambda`s for `nw.(expr|functions)`

Lacking a better alternative, the current pattern is:

    lambda plx: plx.all()
    lambda plx: apply_n_ary_operation(
        plx, lambda x, y: x - y, self, other, str_as_lit=True
    )

If this can *also* get those parts typed - then 🎉
"""

from __future__ import annotations
