from __future__ import annotations

from queries import q12

from . import IO_FUNCS
from . import line_item
from . import orders

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q12.query(fn(line_item), fn(orders)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q12.query(fn(line_item), fn(orders)).collect())

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(q12.query(fn(line_item), fn(orders)))
