"""Experimental translation from `ExprIR` to [Vega Expressions].

Fleshing out the idea from ([c17329f974de8307a4bc17daa2c1268dd4100d9c]) to see how far it can be taken.

## Important
- None of this code is intended to stay in Narwhals
    - but it depends on things that haven't hit `main`
    - so it's easiest to write it here.
- Hoping to reveal what needs exposing to make writing something similar in `altair` possible
    - purely through reusing our representation, which is an artifact of using the public API

[Vega Expressions]: https://vega.github.io/vega/docs/expressions/
[c17329f974de8307a4bc17daa2c1268dd4100d9c]: https://github.com/narwhals-dev/narwhals/commit/c17329f974de8307a4bc17daa2c1268dd4100d9c

See Also:
    [Altair docs](https://altair-viz.github.io/user_guide/interactions/expressions.html)
"""

from __future__ import annotations

from importlib.util import find_spec as _find_spec

if _find_spec("altair") is None:
    msg = "`altair` is required to convert `ExprIR` to altair"
    raise ModuleNotFoundError(msg)
