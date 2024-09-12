from queries import q12

from . import IO_FUNCS
from . import line_item
from . import orders

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q12.query(fn(line_item), fn(orders)))

tool = "polars[eager]"
fn = IO_FUNCS[tool]
print(q12.query(fn(line_item), fn(orders)))

tool = "pyarrow"
fn = IO_FUNCS[tool]
print(q12.query(fn(line_item), fn(orders)))
