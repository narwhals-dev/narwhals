from __future__ import annotations


class ExprIR: ...


class Function(ExprIR):
    """Shared by expr functions and namespace functions."""


class FunctionExpr(ExprIR): ...
