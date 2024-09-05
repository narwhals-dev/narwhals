from queries import q15

from . import IO_FUNCS
from . import lineitem
from . import supplier

tool = "pandas"
fn = IO_FUNCS[tool]
print(q15.query(fn(lineitem), fn(supplier)))

tool = "pandas[pyarrow]"
fn = IO_FUNCS[tool]
print(q15.query(fn(lineitem), fn(supplier)))

tool = "polars[eager]"
fn = IO_FUNCS[tool]
print(q15.query(fn(lineitem), fn(supplier)))

tool = "polars[lazy]"
fn = IO_FUNCS[tool]
print(q15.query(fn(lineitem), fn(supplier)).collect())
