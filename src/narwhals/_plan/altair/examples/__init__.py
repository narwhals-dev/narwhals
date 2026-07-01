"""A subset of [Example Gallery], rewritten using Narwhals expressions.

## Implementation constraints
These rules help to keep the scope reasonable, but reviewing them
at a later date could lead to something more useful:

- Zero changes are allowed within `narwhals._plan`
- Zero new Altair API
    - (A subset of) the existing API is mocked
    - `narwhals._plan.Expr`s are intercepted for translation
- Expressions are not evaluated against a backend/dataframe
- **Strings passed to Altair API must continue using `utils.parse_shorthand`**
    - But any inputs to Narwhals methods will follow Narwhals behavior

[Example Gallery]: https://altair-viz.github.io/gallery/index.html
"""

from __future__ import annotations
