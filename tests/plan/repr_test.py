from __future__ import annotations

import narwhals._plan as nwp


def test_repr() -> None:
    nwp.col("a").meta.as_selector()
    expr = nwp.col("a")
    selector = expr.meta.as_selector()

    expr_repr_html = expr._repr_html_()
    expr_ir_repr_html = expr._ir._repr_html_()
    selector_repr_html = selector._repr_html_()
    selector_ir_repr_html = selector._ir._repr_html_()
    expr_repr = expr.__repr__()
    expr_ir_repr = expr._ir.__repr__()
    selector_repr = selector.__repr__()
    selector_ir_repr = selector._ir.__repr__()

    # In a notebook, both `Expr` and `ExprIR` are displayed the same
    assert expr_repr_html == expr_ir_repr_html
    # The actual repr (for debugging) has more information
    assert expr_repr != expr_repr_html
    # Currently, all extra information is *before* the part which matches
    assert expr_repr.endswith(expr_repr_html)
    # But these guys should not deviate
    assert expr_ir_repr == expr_ir_repr_html
    # The same invariants should hold for `Selector` and `SelectorIR`
    assert selector_repr_html == selector_ir_repr_html
    assert selector_repr != selector_repr_html
    assert selector_repr.endswith(selector_repr_html)
    assert selector_ir_repr == selector_ir_repr_html
    # But they must still be visually different from `Expr` and `ExprIR`
    assert selector_repr_html != expr_repr_html
    assert selector_repr != expr_repr
