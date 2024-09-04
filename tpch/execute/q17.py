from queries import q17

from . import IO_FUNCS
from . import lineitem
from . import part

tool = "pandas"
fn = IO_FUNCS[tool]
print(q17.query(fn(lineitem), fn(part)))

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q17.query(fn(lineitem), fn(part)))

tool = "polars[eager]"
fn = IO_FUNCS[tool]
print(q17.query(fn(lineitem), fn(part)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q17.query(fn(lineitem), fn(part)).collect())
